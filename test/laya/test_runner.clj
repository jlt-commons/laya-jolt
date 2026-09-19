(ns laya.test-runner
  (:require [clojure.test :as t])
  (:gen-class))

(def test-namespaces
  '[laya.tensors-test
    laya.tokenizer-test
    laya.sequence-test
    laya.json-test
    laya.constraints-test
    workflows.email-test
    workflows.presets-test
    laya.lang-test
    laya.router-test
    laya.config-test
    laya.workflows-test
    laya.run-test
    laya.prepare-test
    laya.agent-test
    laya.server-test
    laya.checkpoints-test])

(defn -main
  "jolt -M:test [namespace ...]: the whole suite, or only the namespaces
  named (CI runs laya.checkpoints-test one checkpoint at a time, each in
  its own process: three loaded checkpoints do not fit the mac runner)."
  [& names]
  (let [nss (if (seq names) (map symbol names) test-namespaces)]
    (doseq [ns nss] (require ns))
    (let [{:keys [fail error]} (apply t/run-tests nss)]
      (System/exit (if (zero? (+ fail error)) 0 1)))))
