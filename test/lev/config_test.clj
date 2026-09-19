(ns lev.config-test
  "~/.config/lev/config.edn and the CLI > env > config > default precedence.
  The pure functions take explicit values so the suite never reads the
  developer's real config or environment."
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [lev.config :as cfg]))

(def tmp "target/config-test")

(defn- fresh! []
  (jolt.host/delete-tree! tmp)
  (io/make-parents (io/file tmp "x")))

(deftest config-dir-resolution
  (testing "LEV_CONFIG_DIR wins, then XDG_CONFIG_HOME/lev, then ~/.config/lev"
    (is (= "/tmp/cfg" (cfg/config-dir {"LEV_CONFIG_DIR" "/tmp/cfg" "XDG_CONFIG_HOME" "/x" "HOME" "/h"})))
    (is (= "/x/lev" (cfg/config-dir {"XDG_CONFIG_HOME" "/x" "HOME" "/h"})))
    (is (= "/h/.config/lev" (cfg/config-dir {"HOME" "/h"})))))

(deftest load-config-reads-edn-or-nothing
  (fresh!)
  (testing "no file -> empty map, not an error"
    (is (= {} (cfg/load-config tmp))))
  (testing "config.edn is read as a map"
    (spit (str tmp "/config.edn") "{:data \"/models/laya-data\" :port 9000 :workflow-dirs [\"/extra\"]}")
    (is (= {:data "/models/laya-data" :port 9000 :workflow-dirs ["/extra"]} (cfg/load-config tmp))))
  (testing "a non-map or unreadable file is a clear error"
    (spit (str tmp "/config.edn") "[1 2 3]")
    (is (thrown-with-msg? Exception #"config.edn" (cfg/load-config tmp)))
    (spit (str tmp "/config.edn") "{:data")
    (is (thrown-with-msg? Exception #"config.edn" (cfg/load-config tmp))))
  (jolt.host/delete-tree! tmp))

(deftest setting-precedence
  (let [config {:data "/from-config" :port 9000}
        env {"LEV_DATA" "/from-env"}]
    (testing "CLI flag beats everything"
      (is (= "/from-cli" (cfg/setting {:opts {"--data" "/from-cli"} :env env :config config}
                                      "--data" "LEV_DATA" :data "data"))))
    (testing "then the environment"
      (is (= "/from-env" (cfg/setting {:opts {} :env env :config config} "--data" "LEV_DATA" :data "data"))))
    (testing "then config.edn"
      (is (= "/from-config" (cfg/setting {:opts {} :env {} :config config} "--data" "LEV_DATA" :data "data")))
      (is (= 9000 (cfg/setting {:opts {} :env {} :config config} "--port" "PORT" :port 8080))))
    (testing "then the default"
      (is (= "data" (cfg/setting {:opts {} :env {} :config {}} "--data" "LEV_DATA" :data "data")))
      (is (nil? (cfg/setting {:opts {} :env {} :config {}} "--api-key" "LEV_API_KEY" :api-key nil))))
    (testing "a bare --flag (true) is not a value"
      (is (= "/from-env" (cfg/setting {:opts {"--data" true} :env env :config config} "--data" "LEV_DATA" :data "data"))))))

(deftest workflow-dirs-composition
  (testing "defaults: ./workflows (bundled), config :workflow-dirs, then the user's dir; later wins"
    (is (= ["workflows" "/extra" "/h/.config/lev/workflows"]
           (cfg/workflow-dirs {:opts {} :env {"HOME" "/h"} :config {:workflow-dirs ["/extra"]}})))
    (is (= ["workflows" "/cfg/workflows"]
           (cfg/workflow-dirs {:opts {} :env {"LEV_CONFIG_DIR" "/cfg"} :config {}}))))
  (testing "LEV_WORKFLOWS=dir[:dir] scans exactly those directories"
    (is (= ["/a" "/b"]
           (cfg/workflow-dirs {:opts {} :env {"LEV_WORKFLOWS" "/a:/b" "HOME" "/h"} :config {:workflow-dirs ["/extra"]}})))
    (is (= ["/only"]
           (cfg/workflow-dirs {:opts {} :env {"LEV_WORKFLOWS" "/only"} :config {}}))))
  (testing "--workflows DIR beats the environment"
    (is (= ["/cli"]
           (cfg/workflow-dirs {:opts {"--workflows" "/cli"} :env {"LEV_WORKFLOWS" "/a"} :config {}})))))

(deftest sequence-limits
  (let [config {:max-len 768 :head-max-len 200
                :checkpoints {"multilingual" {:max-len 2048} "typed-decisions" {:head-max-len 300}}}]
    (testing "top-level keys apply to every checkpoint, :checkpoints entries win per name"
      (is (= {:max-len 768 :head-max-len 200} (cfg/limits {:opts {} :env {} :config config} "english")))
      (is (= {:max-len 2048 :head-max-len 200} (cfg/limits {:opts {} :env {} :config config} "multilingual")))
      (is (= {:max-len 768 :head-max-len 300} (cfg/limits {:opts {} :env {} :config config} "typed-decisions"))))
    (testing "env and CLI beat config, for every checkpoint"
      (is (= {:max-len 1024 :head-max-len 200}
             (cfg/limits {:opts {} :env {"LEV_MAX_LEN" "1024"} :config config} "multilingual")))
      (is (= {:max-len 640 :head-max-len 128}
             (cfg/limits {:opts {"--max-len" "640" "--head-max-len" "128"} :env {"LEV_MAX_LEN" "1024"} :config config} "english"))))
    (testing "nothing configured -> {} (the checkpoint's own values stand)"
      (is (= {} (cfg/limits {:opts {} :env {} :config {}} "english"))))
    (testing "values are integers whatever the source"
      (is (= {:max-len 700} (cfg/limits {:opts {} :env {} :config {:max-len "700"}} "english"))))))

(deftest thinkers-from-config-cli-and-env
  (testing "config.edn :thinkers, names as strings, each entry with its model path"
    (is (= {"minicpm5" {:model "/m/MiniCPM5-2B-Q8_0.gguf" :thinking true :max-think-tokens 512}}
           (cfg/thinkers {:opts {} :env {} :config {:thinkers {:minicpm5 {:model "/m/MiniCPM5-2B-Q8_0.gguf" :thinking true :max-think-tokens 512}}}})))
    (is (= {} (cfg/thinkers {:opts {} :env {} :config {}}))))
  (testing "--thinker PATH / LEV_THINKER add (or replace) the one named `thinker`"
    (is (= {"thinker" {:model "/cli.gguf"}}
           (cfg/thinkers {:opts {"--thinker" "/cli.gguf"} :env {"LEV_THINKER" "/env.gguf"} :config {}})))
    (is (= {"thinker" {:model "/env.gguf"}}
           (cfg/thinkers {:opts {} :env {"LEV_THINKER" "/env.gguf"} :config {}})))
    (is (= {"a" {:model "/a.gguf"} "thinker" {:model "/env.gguf"}}
           (cfg/thinkers {:opts {} :env {"LEV_THINKER" "/env.gguf"} :config {:thinkers {"a" {:model "/a.gguf"}}}}))))
  (testing "a thinker entry must name a model"
    (is (thrown-with-msg? Exception #"minicpm5.*:model" (cfg/thinkers {:opts {} :env {} :config {:thinkers {:minicpm5 {:thinking true}}}})))))

(deftest encoders-from-config
  (testing "config.edn :encoders names prepared data directories per checkpoint; else the :data root's layout"
    (is (= {"english" "/models/en" "multilingual" "/models/ml"}
           (cfg/encoders {:opts {} :env {} :config {:encoders {:english "/models/en" "multilingual" "/models/ml"}}})))
    (is (= {} (cfg/encoders {:opts {} :env {} :config {}})))))
