(ns snake.test-runner
  (:require [clojure.test :as t])
  (:gen-class))

(def test-namespaces
  '[snake.game-test
    snake.policy-test])

(defn -main
  "jolt -M:test [namespace ...]: the suite, or only the namespaces named."
  [& names]
  (let [nss (if (seq names) (map symbol names) test-namespaces)]
    (doseq [ns nss] (require ns))
    (let [{:keys [fail error]} (apply t/run-tests nss)]
      (System/exit (if (zero? (+ fail error)) 0 1)))))
