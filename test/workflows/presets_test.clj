(ns workflows.presets-test
  "The bundled workflows that port laya 0.3.0 presets.py: their question
  maps must serialize byte-for-byte like json.dumps of the Python presets
  (golden/presets.edn)."
  (:require [clojure.test :refer [deftest is testing]]
            [laya.sequence :as seq]
            [laya.test-util :as tu]
            [laya.workflows :as wf]))

(def golden (delay (tu/read-golden "presets")))
(def bundled (delay (wf/load-workflows ["workflows"])))

(deftest presets-match-python
  (doseq [[golden-key name] {:triage "triage" :email "email" :guard "guard"
                             :moderation "moderation" :router "llm-router"}]
    (testing name
      (is (contains? @bundled name))
      (is (= (get @golden golden-key) (seq/json-str (wf/questions (get @bundled name))))))))

(deftest presets-have-descriptions-and-state-keys
  (doseq [[name key] {"triage" "message" "guard" "prompt" "moderation" "post" "llm-router" "request"}]
    (testing name
      (let [w (get @bundled name)]
        (is (string? (:doc w)))
        (is (= {key "hello"} (wf/state w "hello")) "a bare string becomes the field the instructions name")
        (is (= {key "hi" "extra" 1} (wf/state w {key "hi" "extra" 1})) "a map passes through")))))
