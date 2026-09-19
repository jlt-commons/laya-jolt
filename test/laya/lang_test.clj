(ns laya.lang-test
  "laya lang.py parity: script detection, the Latin-script language guess and
  state flattening, against golden/lang.edn (analyse() on ~35 texts)."
  (:require [clojure.test :refer [deftest is testing]]
            [laya.lang :as lang]
            [laya.test-util :as tu]))

(def golden (delay (tu/read-golden "lang")))

(deftest analyse-matches-python
  (doseq [{:keys [text script profile language is-english non-latin-fraction guess]} (:cases @golden)]
    (let [det (lang/analyse text)]
      (testing (pr-str text)
        (is (= script (get det "script")))
        (is (= (into {} profile)
               (into {} (map (fn [[k v]] [k (/ (Math/round (* 1e6 (double v))) 1e6)]) (get det "script_profile")))))
        (is (= language (get det "language")))
        (is (= is-english (get det "is_english")))
        (is (= non-latin-fraction (get det "non_latin_fraction")))
        (is (= guess (lang/guess-latin-language text)))))))

(deftest detection-keys-are-ordered-like-python
  (is (= ["script" "script_profile" "language" "is_english" "non_latin_fraction"]
         (keys (lang/analyse "hello there friend")))))

(deftest state-text-flattens-like-python
  (testing "string leaves only, keys ignored, depth capped at 6, 4000 chars"
    (doseq [[state want] (:state-text @golden)]
      (is (= want (lang/state-text state)) (pr-str state)))))

(deftest is-english
  (is (true? (lang/is-english? "Please refund the duplicate charge on invoice 4411 today.")))
  (is (true? (lang/is-english? "refund me")))
  (is (false? (lang/is-english? "मुझसे दो बार शुल्क लिया गया")))
  (is (false? (lang/is-english? {"body" "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die Rechnung die nicht korrekt ist"})))
  (is (true? (lang/is-english? "")) "no letters: the English checkpoint is the default")
  (is (= "unknown" (lang/detect-script "12345 6789")))
  (is (= "hangul" (lang/detect-script "고객이 두 번 청구되어 환불을 원합니다"))))
