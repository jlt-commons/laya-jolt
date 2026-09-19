(ns laya.router-test
  "laya router.py parity: which checkpoint a request goes to and why
  (golden/router.edn: Router.route decisions), plus lazy loading with LRU
  eviction and predict = system-one + routing."
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [laya.router :as router]
            [laya.sequence :as seq]
            [laya.test-util :as tu]))

(def golden (delay (tu/read-golden "router")))

(defn- fake-loader
  "Stands in for load-agent: records the order of loads."
  [log]
  (fn [name dir] (swap! log conj name) {:fake name :dir dir}))

(deftest route-decisions-match-python
  (doseq [{:keys [name state questions kw auto-task default decision]} (:cases @golden)]
    (let [r (router/make-router {:auto-task-detection auto-task :default default
                                 :loader (fake-loader (atom []))})
          want (json/read-str decision)
          ;; upstream puts the raw (repo, subfolder) pair in the workflow
          ;; branch where every other branch has "repo/subfolder"; we say the
          ;; string everywhere
          want (if (sequential? (get want "repo")) (update want "repo" #(clojure.string/join "/" %)) want)
          got (json/read-str (seq/json-str (router/route r state questions
                                                         :model (get kw "model") :task (get kw "task") :lang (get kw "lang"))))]
      (testing name
        (is (= want got))))))

(deftest decision-keys-are-ordered-like-python
  (let [r (router/make-router {:loader (fake-loader (atom []))})]
    (is (= ["model" "repo" "reason" "detection" "workflow"] (keys (router/route r "hello" {}))))))

(deftest names-and-aliases
  (let [aliases (json/read-str (:aliases @golden))]
    (doseq [[alias canonical] aliases]
      (is (= canonical (router/normalise-name alias)) alias))
    (is (= "english" (router/normalise-name "  English ")))
    (is (= (json/read-str (:default-models @golden)) router/repos))
    (let [[bad msg] (first (:normalise-errors @golden))
          e (try (router/normalise-name bad) nil (catch Exception e e))]
      (is (some? e))
      (is (= :unknown-model (:type (ex-data e))))
      (is (= ["english" "multilingual" "typed-decisions"] (:known (ex-data e))))
      (is (some? msg) "python raises too"))))

(deftest typed-decisions-workflows
  (let [wfs (json/read-str (:workflows @golden))]
    (is (= (set (keys wfs)) (set (keys router/typed-decisions-workflows))))
    (doseq [[wf ids] wfs]
      (is (= wf (router/match-typed-decisions-workflow (zipmap ids (repeat {})))) wf)
      (is (nil? (router/match-typed-decisions-workflow (zipmap (conj ids "extra") (repeat {})))) "superset is not a match"))
    (is (nil? (router/match-typed-decisions-workflow {})))
    (is (nil? (router/match-typed-decisions-workflow nil)))))

(deftest lazy-loading-with-lru-eviction
  (let [log (atom [])
        r (router/make-router {:loader (fake-loader log) :max-loaded 1
                               :models {"english" "d/en" "multilingual" "d/ml" "typed-decisions" "d/td"}})]
    (is (= [] (router/loaded r)) "nothing is loaded until a request needs it")
    (is (= {:fake "english" :dir "d/en"} (router/load-model r "english")))
    (is (= ["english"] (router/loaded r)))
    (router/load-model r "en")
    (is (= ["english"] @log) "a second load of the same model is a cache hit")
    (router/load-model r "multilingual")
    (is (= ["multilingual"] (router/loaded r)) "max-loaded 1 evicts the least recently used")
    (is (= ["english" "multilingual"] @log))
    (testing "max-loaded 2 keeps two hot and evicts the older one"
      (let [log (atom [])
            r (router/make-router {:loader (fake-loader log) :max-loaded 2})]
        (router/load-model r "english")
        (router/load-model r "multilingual")
        (is (= ["english" "multilingual"] (router/loaded r)))
        (router/load-model r "english")               ; touch: english is now most recent
        (router/load-model r "typed-decisions")
        (is (= ["english" "typed-decisions"] (router/loaded r)))))
    (testing "unload one or all"
      (router/load-model r "english")
      (router/unload r "english")
      (is (= [] (router/loaded r)))
      (router/load-model r "english")
      (router/unload r)
      (is (= [] (router/loaded r))))
    (testing "max-loaded is at least 1"
      (is (= 1 (:max-loaded (router/make-router {:loader (fake-loader (atom [])) :max-loaded 0})))))))

(deftest predict-is-system-one-plus-routing
  (let [r (router/make-router {:models {"english" tu/data-dir "multilingual" "target/not-prepared"}
                               ;; the suite's one english agent; other names go through the
                               ;; default loader so the not-prepared path is still exercised
                               :loader (fn [name dir] (if (= name "english") @tu/agent (router/load-prepared name dir)))})
        cases (tu/read-golden "cases")
        out (router/predict r (:readme-state cases) (:readme-questions cases))]
    (is (= ["model" "answers" "usage" "routing"] (keys out)))
    (tu/answers-match (:system-one (tu/read-golden "readme")) (seq/json-str (dissoc out "routing")))
    (is (= "english" (get-in out ["routing" "model"])))
    (is (= "English Latin text" (get-in out ["routing" "reason"])))
    (is (= ["english"] (router/loaded r)))
    (testing "a checkpoint with no prepared data says so instead of crashing"
      (let [e (try (router/predict r {"body" "मुझसे दो बार शुल्क लिया गया"} (:readme-questions cases)) nil
                   (catch Exception e e))]
        (is (= :model-unavailable (:type (ex-data e))))
        (is (= "multilingual" (:model (ex-data e))))))))
