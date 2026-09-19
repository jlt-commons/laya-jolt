(ns laya.config-test
  "~/.config/laya/config.edn and the CLI > env > config > default precedence.
  The pure functions take explicit values so the suite never reads the
  developer's real config or environment."
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [laya.config :as cfg]))

(def tmp "target/config-test")

(defn- fresh! []
  (jolt.host/delete-tree! tmp)
  (io/make-parents (io/file tmp "x")))

(deftest config-dir-resolution
  (testing "LAYA_CONFIG_DIR wins, then XDG_CONFIG_HOME/laya, then ~/.config/laya"
    (is (= "/tmp/cfg" (cfg/config-dir {"LAYA_CONFIG_DIR" "/tmp/cfg" "XDG_CONFIG_HOME" "/x" "HOME" "/h"})))
    (is (= "/x/laya" (cfg/config-dir {"XDG_CONFIG_HOME" "/x" "HOME" "/h"})))
    (is (= "/h/.config/laya" (cfg/config-dir {"HOME" "/h"})))))

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
        env {"LAYA_DATA" "/from-env"}]
    (testing "CLI flag beats everything"
      (is (= "/from-cli" (cfg/setting {:opts {"--data" "/from-cli"} :env env :config config}
                                      "--data" "LAYA_DATA" :data "data"))))
    (testing "then the environment"
      (is (= "/from-env" (cfg/setting {:opts {} :env env :config config} "--data" "LAYA_DATA" :data "data"))))
    (testing "then config.edn"
      (is (= "/from-config" (cfg/setting {:opts {} :env {} :config config} "--data" "LAYA_DATA" :data "data")))
      (is (= 9000 (cfg/setting {:opts {} :env {} :config config} "--port" "PORT" :port 8080))))
    (testing "then the default"
      (is (= "data" (cfg/setting {:opts {} :env {} :config {}} "--data" "LAYA_DATA" :data "data")))
      (is (nil? (cfg/setting {:opts {} :env {} :config {}} "--api-key" "LAYA_API_KEY" :api-key nil))))
    (testing "a bare --flag (true) is not a value"
      (is (= "/from-env" (cfg/setting {:opts {"--data" true} :env env :config config} "--data" "LAYA_DATA" :data "data"))))))

(deftest workflow-dirs-composition
  (testing "defaults: ./workflows (bundled), config :workflow-dirs, then the user's dir; later wins"
    (is (= ["workflows" "/extra" "/h/.config/laya/workflows"]
           (cfg/workflow-dirs {:opts {} :env {"HOME" "/h"} :config {:workflow-dirs ["/extra"]}})))
    (is (= ["workflows" "/cfg/workflows"]
           (cfg/workflow-dirs {:opts {} :env {"LAYA_CONFIG_DIR" "/cfg"} :config {}}))))
  (testing "LAYA_WORKFLOWS=dir[:dir] scans exactly those directories"
    (is (= ["/a" "/b"]
           (cfg/workflow-dirs {:opts {} :env {"LAYA_WORKFLOWS" "/a:/b" "HOME" "/h"} :config {:workflow-dirs ["/extra"]}})))
    (is (= ["/only"]
           (cfg/workflow-dirs {:opts {} :env {"LAYA_WORKFLOWS" "/only"} :config {}}))))
  (testing "--workflows DIR beats the environment"
    (is (= ["/cli"]
           (cfg/workflow-dirs {:opts {"--workflows" "/cli"} :env {"LAYA_WORKFLOWS" "/a"} :config {}})))))
