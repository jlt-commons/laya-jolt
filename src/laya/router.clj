(ns laya.router
  "Route a request to the Laya checkpoint best suited to it (the port of laya
  0.3.0 router.py; decisions pinned by golden/router.edn).

  Three checkpoints, all in the Hub repo convaiinnovations/laya:

    english          root                ModernBERT-large 421M, 512 tokens
    multilingual     multilingual/       mmBERT-base 322M, 1024 tokens, 100+ languages
    typed-decisions  typed-decisions/    ModernBERT-large 421M, 1024 tokens, fine-tuned
                                         on the four typed-decisions workflows

  The English checkpoint does not degrade gracefully off English, it
  collapses while staying confident, so script detection (laya.lang) is the
  primary routing signal. typed-decisions is never picked automatically
  unless :auto-task-detection is on or task=\"typed_decisions\" is passed.

  Precedence: explicit model > explicit task > detected workflow (opt-in) >
  explicit lang > detected script/language > default.

  A router holds prepared data directories per checkpoint ({name dir}, by
  default data/, data/multilingual, data/typed-decisions), loads an agent on
  first use and keeps :max-loaded of them resident, evicting the least
  recently used: all three together are ~4.6 GB of f32."
  (:require [clojure.string :as str]
            [laya.agent :as ag]
            [laya.lang :as lang]
            [laya.sequence :as seq]))

(def bundle-repo "convaiinnovations/laya")

(def repos
  "Checkpoint name -> where it lives on the Hub (repo or repo/subfolder)."
  (seq/ordered-map [["english" bundle-repo]
                    ["multilingual" (str bundle-repo "/multilingual")]
                    ["typed-decisions" (str bundle-repo "/typed-decisions")]]))

(def names ["english" "multilingual" "typed-decisions"])

(def aliases
  "Spellings people are likely to type."
  {"en" "english" "laya" "english" "default" "english"
   "multi" "multilingual" "ml" "multilingual" "laya-multilingual" "multilingual"
   "typed" "typed-decisions" "typed_decisions" "typed-decisions"
   "laya-typed-decisions" "typed-decisions" "decisions" "typed-decisions"})

(def typed-decisions-workflows
  "Question-id signatures of the four typed-decisions workflows, used only
  when :auto-task-detection is on."
  {"agent_trace_observability" #{"action" "needs_review" "outcome" "risk" "urgency"}
   "customer_service" #{"action" "category" "churn_risk" "needs_human" "urgency"}
   "invoice_processing" #{"discrepancy_severity" "disposition" "duplicate" "matches_order" "urgency"}
   "security_incidents" #{"credential_compromise" "disposition" "severity" "true_positive" "urgency"}})

(defn default-models
  "The bundle layout jolt prepare writes under one data root."
  [data-root]
  (seq/ordered-map [["english" data-root]
                    ["multilingual" (str data-root "/multilingual")]
                    ["typed-decisions" (str data-root "/typed-decisions")]]))

(defn normalise-name
  "Canonical checkpoint name for a name or alias, case- and space-insensitive.
  Throws {:type :unknown-model} otherwise."
  [name]
  (let [key (str/lower-case (str/trim (str name)))
        key (get aliases key key)]
    (when-not (contains? repos key)
      (throw (ex-info (str "unknown model " (pr-str name) "; choose one of " (str/join ", " names)
                          " (or an alias: " (str/join ", " (sort (keys aliases))) ")")
                      {:type :unknown-model :model name :known names :aliases (sort (keys aliases))})))
    key))

(defn match-typed-decisions-workflow
  "Name of the typed-decisions workflow whose question ids these are exactly,
  else nil. A schema that merely contains \"urgency\" is never captured."
  [questions]
  (let [ids (set (map #(if (keyword? %) (name %) (str %)) (keys (or questions {}))))]
    (some (fn [[wf sig]] (when (= ids sig) wf)) (sort typed-decisions-workflows))))

;; --- the router -----------------------------------------------------------------------

(defn load-prepared
  "The default loader: laya.agent/load-agent on a prepared data directory.
  Throws {:type :model-unavailable} when there is nothing there."
  [name dir]
  (when-not (and dir (.exists (clojure.java.io/file dir "manifest.edn")))
    (throw (ex-info (str "checkpoint " name " is not prepared (no " dir "/manifest.edn); "
                         "run jolt prepare for it or point :models at its data directory")
                    {:type :model-unavailable :model name :dir dir})))
  (ag/load-agent dir))

(defn make-router
  "Options: :models {name data-dir} (default (default-models \"data\")),
  :data root for default-models, :max-loaded (default 1), :default
  checkpoint (\"english\"), :auto-task-detection (false), :loader
  (fn [name dir] agent) for tests (default laya.agent/load-agent on dir)."
  [{:keys [models data max-loaded default auto-task-detection loader]
    :or {max-loaded 1 default "english" auto-task-detection false}}]
  {:models (into (default-models (or data "data"))
                 (map (fn [[k v]] [(normalise-name k) v])) models)
   :max-loaded (max 1 (long max-loaded))
   :default (normalise-name default)
   :auto-task-detection (boolean auto-task-detection)
   :loader (or loader load-prepared)
   :agents (atom {})
   :order (atom [])})            ; least recently used first

(defn preloaded
  "A router that already holds `agent` as checkpoint `name` (default
  english): what (laya.server/handler agent opts) builds, and what a test
  with one loaded agent wants."
  ([agent] (preloaded agent "english"))
  ([agent name] (preloaded agent name {}))
  ([agent name opts]
   (let [r (make-router opts)
         key (normalise-name name)]
     (swap! (:agents r) assoc key agent)
     (swap! (:order r) conj key)
     r)))

(defn router?
  [x]
  (and (map? x) (contains? x :agents) (contains? x :loader)))

(defn available?
  "Is this checkpoint's data directory prepared?"
  [router name]
  (let [dir (get (:models router) (normalise-name name))]
    (boolean (and dir (.exists (clojure.java.io/file dir "manifest.edn"))))))

(defn loaded
  "Resident checkpoints, least recently used first."
  [router]
  @(:order router))

(defn- touch! [router key]
  (swap! (:order router) #(conj (vec (remove #{key} %)) key)))

(defn- evict! [{:keys [agents order max-loaded]}]
  (while (> (count @order) max-loaded)
    (let [victim (first @order)]
      (swap! order subvec 1)
      (swap! agents dissoc victim))))

(defn load-model
  "The agent for `name`, loading it on first use (and evicting the least
  recently used past :max-loaded)."
  [router name]
  (let [key (normalise-name name)]
    (if-let [agent (get @(:agents router) key)]
      (do (touch! router key) agent)
      (let [dir (get (:models router) key)]
        (let [agent ((:loader router) key dir)]
          (swap! (:agents router) assoc key agent)
          (touch! router key)
          (evict! router)
          agent)))))

(defn unload
  "Free one checkpoint, or all of them."
  ([router]
   (reset! (:agents router) {})
   (reset! (:order router) []))
  ([router name]
   (let [key (normalise-name name)]
     (swap! (:agents router) dissoc key)
     (swap! (:order router) #(vec (remove #{key} %))))))

;; --- routing --------------------------------------------------------------------------

(defn- py-repr
  "Python %r of a string argument, as the reasons quote it."
  [s]
  (str "'" s "'"))

(defn- decision [model reason detection workflow]
  (seq/ordered-map [["model" model] ["repo" (get repos model)] ["reason" reason]
                    ["detection" detection] ["workflow" workflow]]))

(defn route
  "Decide which checkpoint to use, without loading or running anything:
  {model repo reason detection workflow}, keyed like the Python RouteDecision.
  Keyword options :model :task :lang, all optional."
  [router state questions & {:keys [model task lang]}]
  (cond
    (some? model)
    (decision (normalise-name model) (str "explicit model=" (py-repr model)) nil nil)

    (some? task)
    (let [key (normalise-name (if (= "typed_decisions" (str/replace (str/lower-case (str task)) "-" "_"))
                                "typed-decisions" task))]
      (decision key (str "explicit task=" (py-repr task)) nil nil))

    :else
    (let [workflow (match-typed-decisions-workflow questions)]
      (cond
        (and workflow (:auto-task-detection router))
        (decision "typed-decisions"
                  (str "question ids match the " (py-repr workflow) " typed-decisions workflow")
                  nil workflow)

        (some? lang)
        (let [key (if (contains? #{"en" "eng" "english"} (first (str/split (str/lower-case (str lang)) #"-")))
                    "english" "multilingual")]
          (decision key (str "explicit lang=" (py-repr lang)) nil workflow))

        :else
        (let [det (lang/analyse state)
              script (get det "script")]
          (cond
            (= script "unknown")
            (decision (:default router)
                      (str "no letters detected in state; using default (" (:default router) ")")
                      det workflow)

            (not= script "latin")
            (decision "multilingual"
                      (format "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"
                              script (* 100.0 (double (get det "non_latin_fraction"))))
                      det workflow)

            (not (get det "is_english"))
            (decision "multilingual"
                      (str "Latin script but language looks like " (py-repr (get det "language")) ", not English")
                      det workflow)

            :else
            (decision "english" "English Latin text" det workflow)))))))

(defn predict
  "Route, then answer every question in one forward pass on the chosen
  checkpoint: the system-one map plus a \"routing\" key with the decision."
  [router state questions & {:keys [model task lang]}]
  (let [d (route router state questions :model model :task task :lang lang)
        agent (load-model router (get d "model"))]
    (assoc (ag/system-one agent state questions) "routing" d)))
