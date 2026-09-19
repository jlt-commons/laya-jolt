(ns laya.email
  "Email helpers, mirroring email_utils.py: clean raw emails into a compact
  state and a ready-made set of email questions.

  Jev-style models lose accuracy on long, noisy state, and the model reads
  at most max_len (512) tokens, so strip quoted replies, signatures and
  disclaimers in code before asking questions.

  The Python module is regex-based with `re.I` and Unicode \\s / \\w. jolt's
  irregex has ASCII-only classes, so the matchers are written out by hand
  with Python's semantics: \\s is str.isspace(), \\w is str.isalnum() or _,
  strip() removes str.isspace() chars."
  (:require [clojure.string :as str]
            [laya.sequence :as seq]
            [laya.tokenizer :as tk]))

;; --- python string semantics --------------------------------------------------

(def ^:private py-space-cps
  "str.isspace(): bidi WS/B/S or category Zs."
  (into #{0x20 0x85 0xA0 0x1680 0x2028 0x2029 0x202F 0x205F 0x3000}
        (concat (range 0x09 0x0E) (range 0x1C 0x20) (range 0x2000 0x200B))))

(defn- py-space? [c] (contains? py-space-cps (int c)))

(defn- word-char?
  "re \\w for str patterns: str.isalnum() (\\p{L} | \\p{N}) or underscore."
  [c]
  (or (= c \_) (tk/letter? c) (tk/number? c)))

(defn- py-lstrip [s] (apply str (drop-while py-space? s)))
(defn- py-rstrip [s] (apply str (reverse (drop-while py-space? (reverse s)))))
(defn- py-strip [s] (py-lstrip (py-rstrip s)))

(defn- lower [s] (str/lower-case s))

(defn- after-prefix
  "The rest of s after a case-insensitive prefix, or nil."
  [s prefix]
  (when (and (>= (count s) (count prefix))
             (= (lower (subs s 0 (count prefix))) (lower prefix)))
    (subs s (count prefix))))

(defn- all? [pred s] (every? pred s))

(defn- run-count [pred s] (count (take-while pred s)))

;; --- the line patterns (each is re.match, i.e. anchored at the start) -------------

(defn- on-wrote?
  "^\\s*On .{0,300}wrote:\\s*$  (re.I). `.` never sees a newline here: the
  input was split on \\n before matching."
  [line]
  (when-let [rest (after-prefix (py-lstrip line) "On ")]
    (let [n (count rest)
          low (lower rest)]
      (some (fn [j]
              (and (str/starts-with? (subs low j) "wrote:")
                   (all? py-space? (subs rest (+ j 6)))))
            (range 0 (inc (min 300 (- n 6))))))))

(defn- original-message?
  "^\\s*-{2,}\\s*(Original|Forwarded) Message\\s*-{2,}  (re.I)"
  [line]
  (let [s (py-lstrip line)
        d1 (run-count #(= % \-) s)]
    (when (>= d1 2)
      (let [s (py-lstrip (subs s d1))]
        (when-let [rest (or (after-prefix s "Original Message")
                            (after-prefix s "Forwarded Message"))]
          (>= (run-count #(= % \-) (py-lstrip rest)) 2))))))

(defn- underscore-rule?
  "^\\s*_{8,}\\s*$"
  [line]
  (let [s (py-lstrip line)
        u (run-count #(= % \_) s)]
    (and (>= u 8) (all? py-space? (subs s u)))))

(defn- from-header?
  "^\\s*From:\\s.+$  (re.I): a whitespace char after the colon, then at
  least one more character."
  [line]
  (when-let [rest (after-prefix (py-lstrip line) "From:")]
    (and (>= (count rest) 2) (py-space? (first rest)))))

(def ^:private quote-header? (some-fn on-wrote? original-message? underscore-rule? from-header?))

(defn- dash-dash?
  "^\\s*--\\s*$"
  [line]
  (let [s (py-lstrip line)]
    (and (str/starts-with? s "--") (all? py-space? (subs s 2)))))

(def ^:private sign-offs
  ["best" "kind" "warm" "many thanks" "thanks" "thank you" "regards" "cheers" "sincerely"])

(defn- sign-off?
  "^\\s*(best|kind|...|sincerely)[\\w ,!.]*$  (re.I). Every alternative is
  itself made of class characters, so: starts with one, and the whole rest
  of the line stays inside [\\w ,!.]."
  [line]
  (let [s (py-lstrip line)]
    (and (some #(after-prefix s %) sign-offs)
         (all? (fn [c] (or (word-char? c) (#{\space \, \! \.} c))) s))))

(defn- sent-from?
  "^\\s*sent from my (iphone|android|mobile|ipad)  (re.I)"
  [line]
  (when-let [rest (after-prefix (py-lstrip line) "sent from my ")]
    (boolean (some #(after-prefix rest %) ["iphone" "android" "mobile" "ipad"]))))

(def ^:private signature-marker? (some-fn dash-dash? sign-off? sent-from?))

(def ^:private disclaimer-phrases
  "Every expansion of the optional groups in
   (confidential|intended (solely )?for the (use of the )?(named )?(addressee|recipient)|
    if you (have )?received this (e-?mail|message) in error)"
  (concat ["confidential"]
          (for [solely ["" "solely "] use ["" "use of the "] named ["" "named "]
                who ["addressee" "recipient"]]
            (str "intended " solely "for the " use named who))
          (for [have ["" "have "] what ["e-mail" "email" "message"]]
            (str "if you " have "received this " what " in error"))))

(defn- disclaimer?
  "_DISCLAIMER.search(p) with re.I"
  [paragraph]
  (let [low (lower paragraph)]
    (boolean (some #(str/includes? low %) disclaimer-phrases))))

;; --- re.split(r\"\\n\\s*\\n\", text) ------------------------------------------------

(defn- split-blank-lines
  "Split on a newline, optional whitespace (\\s includes newlines), and a
  newline: greedy with backtracking, so each separator is the longest
  whitespace run that both starts and ends with a newline."
  [s]
  (let [n (count s)]
    (loop [i 0 start 0 out []]
      (if (>= i n)
        (conj out (subs s start n))
        (if (= (nth s i) \newline)
          (let [j (+ i (run-count py-space? (subs s i)))          ; end of the ws run
                k (loop [k (dec j)] (if (> k i) (if (= (nth s k) \newline) k (recur (dec k))) nil))]
            (if k
              (recur (inc k) (inc k) (conj out (subs s start i)))
              (recur (inc i) start out)))
          (recur (inc i) start out))))))

;; --- public API -------------------------------------------------------------------

(defn clean-email-body
  "Remove quoted history, signature and legal disclaimer; collapse
  whitespace; truncate to max-chars (default 3000) codepoints."
  ([body] (clean-email-body body 3000))
  ([body max-chars]
   (let [text (-> (or body "")
                  (str/replace "\r\n" "\n")
                  (str/replace "\r" "\n")
                  (str/replace "\\n" "\n"))
         lines (loop [ls (str/split text #"\n" -1) out []]
                 (if (empty? ls)
                   out
                   (let [line (first ls)]
                     (cond
                       ;; everything below a quote header is the previous thread
                       (and (quote-header? line) (seq out)) out
                       (str/starts-with? (py-lstrip line) ">") (recur (rest ls) out)
                       :else (recur (rest ls) (conj out (py-rstrip line)))))))
         n (count lines)
         ;; a sign-off only counts near the end (last 40%, or last 8 lines of
         ;; a short email) and must be a short line
         from (max 1 (min (long (Math/floor (* n 0.6))) (- n 8)))
         cut (or (some (fn [i]
                         (when (and (<= (count (py-strip (nth lines i))) 40)
                                    (signature-marker? (nth lines i)))
                           i))
                       (range from n))
                 n)
         lines (subvec lines 0 cut)
         paragraphs (remove disclaimer? (split-blank-lines (str/join "\n" lines)))
         text (->> paragraphs
                   (map py-strip)
                   (remove #(= % ""))
                   (str/join "\n\n"))
         text (str/replace text #"[ \t]+" " ")]
     (subs text 0 (min max-chars (count text))))))

(defn email-state
  "Build the state map the email questions refer to (`subject`, `body`,
  optional `from`), keyed by strings like the Python dict.
  Options: :sender, :clean (default true), :extra — a map merged in with
  nil values dropped."
  [subject body & {:keys [sender clean extra] :or {clean true}}]
  (seq/ordered-map
   (concat [["subject" (py-strip (or subject ""))]
            ["body" (if clean (clean-email-body body) (or body ""))]]
           (when (and sender (not= sender "")) [["from" sender]])
           (remove (fn [[_ v]] (nil? v)) extra))))

(def default-categories
  (array-map "billing" "invoices, payments, refunds"
             "technical" "bugs, outages, integrations"
             "sales" "pricing, demos, new purchases"
             "account" "login, access, profile changes"
             "hr" "hiring, leave, payroll"
             "other" "none of the above"))

(defn email-questions
  "A default fan-out of email questions, in the Jev request shape that
  agent/system-one accepts. `categories` = {key description} for your own
  routing labels (an ordered map: option order is part of the input)."
  ([] (email-questions nil))
  ([categories]
   (let [categories (if (seq categories) categories default-categories)]
     (array-map
      "category" (array-map "type" "choice"
                            "instructions" "Which team should handle the email in `body`?"
                            "criteria" categories)
      "is_spam" (array-map "type" "noul"
                           "instructions" "Is this email unsolicited spam or bulk marketing?")
      "is_phishing" (array-map "type" "noul"
                               "instructions" "Is this email a phishing or scam attempt to steal money, credentials, or personal data?"
                               "criteria" (array-map "true" "phishing, scam, or fraud"
                                                     "false" "a legitimate email"))
      "urgency" (array-map "type" "score"
                           "instructions" "How urgent is the issue described in `body`?"
                           "criteria" ["no time pressure" "needs attention soon"
                                       "blocking issue or hard deadline"])
      "needs_reply" (array-map "type" "noul"
                               "instructions" "Does the sender expect a reply?")
      "sentiment" (array-map "type" "score"
                             "instructions" "What is the sender's tone in `body`?"
                             "criteria" ["angry or very negative" "negative" "neutral" "positive"])))))
