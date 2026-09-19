(ns workflows.presets-test
  "The bundled workflows that port the upstream presets.py: their question
  maps must serialize byte-for-byte like json.dumps of the Python presets
  (golden/presets.edn)."
  (:require [clojure.test :refer [deftest is testing]]
            [lev.sequence :as seq]
            [lev.test-util :as tu]
            [lev.workflows :as wf]))

(def golden (delay (tu/read-golden "presets")))
(def bundled (delay (wf/load-workflows ["workflows"])))

(deftest presets-match-python
  (doseq [[golden-key name] {:triage "triage" :email "email" :guard "guard"
                             :moderation "moderation" :router "llm-router"}]
    (testing name
      (is (contains? @bundled name))
      (is (= (get @golden golden-key) (seq/json-str (wf/questions (get @bundled name))))))))

(deftest security-preset-from-von
  ;; not an upstream preset: von's security_preset (github.com/wfzyx/von),
  ;; with constraints the upstream presets have no equivalent of
  (let [w (get @bundled "security")]
    (is (= ["event_type" "is_threat" "severity"] (keys (wf/questions w))))
    (is (= 5 (count (get-in (wf/questions w) ["event_type" "criteria"]))))
    (is (= 4 (count (get-in (wf/questions w) ["severity" "criteria"]))))
    (is (= 2 (count (wf/constraints w))))))

(deftest presets-have-descriptions-and-state-keys
  (doseq [[name key] {"triage" "message" "guard" "prompt" "moderation" "post" "llm-router" "request" "security" "event"}]
    (testing name
      (let [w (get @bundled name)]
        (is (string? (:doc w)))
        (is (= {key "hello"} (wf/state w "hello")) "a bare string becomes the field the instructions name")
        (is (= {key "hi" "extra" 1} (wf/state w {key "hi" "extra" 1})) "a map passes through")))))
