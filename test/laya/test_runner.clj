(ns laya.test-runner
  (:require [clojure.test :as t])
  (:gen-class))

(def test-namespaces
  '[laya.tensors-test
    laya.tokenizer-test
    laya.sequence-test
    laya.json-test
    workflows.email-test
    laya.config-test
    laya.workflows-test
    laya.run-test
    laya.prepare-test
    laya.agent-test
    laya.server-test])

(defn -main
  [& _]
  (doseq [ns test-namespaces] (require ns))
  (let [{:keys [fail error]} (apply t/run-tests test-namespaces)]
    (System/exit (if (zero? (+ fail error)) 0 1))))
