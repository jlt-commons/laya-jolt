(ns lev.router-test
  "laya router.py parity: which checkpoint a request goes to and why
  (golden/router.edn: Router.route decisions), plus lazy loading with LRU
  eviction and predict = system-one + routing."
  (:require [clojure.data.json :as json]
            [clojure.test :refer [deftest is testing]]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.test-util :as tu]))

(def golden (delay (tu/read-golden "router")))

(defn- fake-loader
  "Stands in for load-agent: records the order of loads."
  [log]
  (fn [name dir & _] (swap! log conj name) {:fake name :dir dir}))

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
                               :loader (fn [name dir limits] (if (= name "english") @tu/agent (router/load-prepared name dir limits)))})
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

(deftest limits-reach-the-loader
  (let [seen (atom nil)
        r (router/make-router {:limits {:max-len 768}
                               :checkpoints {"multilingual" {:max-len 2048 :head-max-len 300}}
                               :loader (fn [name dir limits] (reset! seen [name dir limits]) {:fake name})})]
    (router/load-model r "english")
    (is (= ["english" "data" {:max-len 768}] @seen))
    (router/load-model r "ml")
    (is (= ["multilingual" "data/multilingual" {:max-len 2048 :head-max-len 300}] @seen))
    (testing "effective limits for the listing: the checkpoint's config.edn plus overrides"
      (is (= {:max-len 768 :head-max-len 192} (router/effective-limits r "english")))
      (is (= {:max-len 2048 :head-max-len 300} (router/effective-limits r "multilingual")))
      (is (nil? (router/effective-limits (router/make-router {:models {"english" "target/nowhere"} :loader (fn [& _])}) "english"))))))

(deftest thinkers-are-models-of-their-own
  (let [log (atom [])
        fake-thinker (fn [name cfg] (swap! log conj [:thinker name]) {:kind :thinker :name name :cfg cfg
                                                                     :decide (fn [& _]) :count-tokens (fn [& _] 0)})
        r (router/make-router {:models {"english" tu/data-dir}
                               :thinkers {"minicpm5" {:model "target/no-such.gguf" :thinking true}
                                          "direct" {:model "target/no-such.gguf" :thinking false}}
                               :loader (fake-loader log)
                               :thinker-loader fake-thinker})]
    (testing "names: a thinker is a known model, not an alias of a checkpoint"
      (is (= "minicpm5" (router/normalise-name r "minicpm5")))
      (is (= "minicpm5" (router/normalise-name r " MiniCPM5 ")))
      (is (thrown-with-msg? Exception #"unknown model" (router/normalise-name r "gpt")))
      (is (= ["english" "multilingual" "typed-decisions"] router/names))
      (is (= ["direct" "minicpm5"] (router/thinker-names r)))
      (is (router/thinker? r "minicpm5"))
      (is (not (router/thinker? r "english"))))
    (testing "explicit only: content routing never picks a thinker"
      (let [d (router/route r "Refund me." {} :model "minicpm5")]
        (is (= "minicpm5" (get d "model")))
        (is (nil? (get d "repo")))
        (is (= "explicit model='minicpm5'" (get d "reason"))))
      (is (= "english" (get (router/route r "Refund me." {}) "model"))))
    (testing "loaded through the thinker loader, kept in their own slot, evicted least recently used past :max-thinkers (1)"
      (let [a (router/load-model r "minicpm5")]
        (is (= :thinker (:kind a)))
        (is (= [[:thinker "minicpm5"]] @log))
        (is (= ["minicpm5"] (router/loaded-thinkers r)))
        (router/load-model r "english")
        (is (= ["english"] (router/loaded r)) "a checkpoint does not evict a thinker")
        (is (= ["minicpm5"] (router/loaded-thinkers r)))
        (router/load-model r "direct")
        (is (= ["direct"] (router/loaded-thinkers r)))
        (is (= [[:thinker "minicpm5"] "english" [:thinker "direct"]] @log))))
    (testing "availability: the file must exist and the llm native be built"
      (is (false? (router/available? r "minicpm5"))))
    (testing "predict hands :thinking and :thought to the thinker"
      (let [seen (atom nil)
            r2 (router/make-router {:thinkers {"t" {:model "x.gguf"}}
                                    :loader (fake-loader (atom []))
                                    :thinker-loader (fn [name cfg]
                                                      {:kind :thinker :name name :cfg cfg
                                                       :decide (fn [prompt options opts]
                                                                 (reset! seen [options opts])
                                                                 {:logp (vec (repeat (count options) -1.0)) :thought "" :tokens 0})
                                                       :count-tokens (fn [_] 1)})})
            out (router/predict r2 "s" {"q" {"type" "noul" "instructions" "?"}} :model "t" :thinking false)]
        (is (= ["true" "false"] (first @seen)))
        (is (= 0 (:think-max (second @seen))))
        (is (= "t" (get out "model")))
        (is (= "t" (get-in out ["routing" "model"])))))))
