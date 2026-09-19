(ns laya.test-runner
  (:require [clojure.test :as t])
  (:gen-class))

(def test-namespaces
  '[laya.tensors-test
    laya.tokenizer-test
    laya.sequence-test
    laya.email-test
    laya.prepare-test
    laya.agent-test])

(defn -main
  [& _]
  (doseq [ns test-namespaces] (require ns))
  (let [{:keys [fail error]} (apply t/run-tests test-namespaces)]
    (System/exit (if (zero? (+ fail error)) 0 1))))
