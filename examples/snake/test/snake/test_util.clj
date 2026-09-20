(ns snake.test-util
  "The english checkpoint for the policy smoke test, loaded once."
  (:require [lev.agent :as ag]))

(def data-dir
  (or (System/getenv "LEV_DATA") "../../data"))

(def agent
  (delay (ag/load-agent data-dir)))
