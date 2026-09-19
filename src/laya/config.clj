(ns laya.config
  "~/.config/laya: `config.edn` (where the prepared model lives, server
  defaults, extra workflow directories) and the one precedence rule every
  entry point follows: CLI flag > environment variable > config.edn > default.

  config.edn keys:
    :data            prepared model directory (jolt prepare --out)
    :laya-home       checkpoint directory jolt prepare reads
    :workflow-dirs   extra directories of workflow .clj files
    :port :host :api-key   server defaults

  Environment:
    LAYA_CONFIG_DIR  instead of $XDG_CONFIG_HOME/laya or ~/.config/laya
    LAYA_WORKFLOWS   dir[:dir...] to scan for workflows INSTEAD of the
                     defaults (bundled ./workflows, :workflow-dirs, the
                     config dir's workflows/)

  The pure functions take an `env` map so tests never read the real
  environment; nil means the process environment."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(defn parse-args
  "--flag value pairs, bare --flags (true) and positionals under :args:
  (parse-args [\"demo\" \"--port\" \"80\" \"--self-test\"])
  => {:args [\"demo\"] \"--port\" \"80\" \"--self-test\" true}"
  [args]
  (loop [args (seq args) out {:args []}]
    (if (empty? args)
      out
      (let [[k v & more] args]
        (cond
          (not (str/starts-with? k "--")) (recur (rest args) (update out :args conj k))
          (and v (not (str/starts-with? v "--"))) (recur more (assoc out k v))
          :else (recur (rest args) (assoc out k true)))))))

(defn- getenv [env k]
  (if (nil? env) (System/getenv k) (get env k)))

(defn config-dir
  "LAYA_CONFIG_DIR, else $XDG_CONFIG_HOME/laya, else ~/.config/laya."
  ([] (config-dir nil))
  ([env]
   (or (getenv env "LAYA_CONFIG_DIR")
       (some-> (getenv env "XDG_CONFIG_HOME") (str "/laya"))
       (str (or (getenv env "HOME") (System/getProperty "user.home")) "/.config/laya"))))

(defn load-config
  "The map in <dir>/config.edn, or {} when there is no such file."
  ([] (load-config (config-dir)))
  ([dir]
   (let [f (io/file dir "config.edn")]
     (if-not (.exists f)
       {}
       (let [m (try (edn/read-string (slurp f))
                    (catch Exception e
                      (throw (ex-info (str "cannot read " f ": " (ex-message e)) {:file (str f)} e))))]
         (when-not (map? m)
           (throw (ex-info (str f " must hold a map, not " (pr-str m)) {:file (str f)})))
         m)))))

(defn context
  "What the resolvers below need: parsed CLI opts ({\"--data\" \"dir\"}), the
  process environment, and config.edn."
  [opts]
  {:opts opts :env nil :config (load-config (config-dir))})

(defn setting
  "CLI flag > env var > config.edn key > default. A bare --flag (true) is
  not a value."
  [{:keys [opts env config]} flag env-var key default]
  (let [v (get opts flag)
        e (getenv env env-var)]
    (cond
      (string? v) v
      (some? e) e
      (contains? config key) (get config key)
      :else default)))

(defn workflow-dirs
  "Directories to load workflows from, in load order (later wins on a name
  clash): the bundled ./workflows, config.edn :workflow-dirs, then the
  config dir's own workflows/. --workflows or LAYA_WORKFLOWS (dir[:dir...])
  replaces that list, which is what a test wants."
  [{:keys [opts env config]}]
  (let [cli (get opts "--workflows")
        from-env (getenv env "LAYA_WORKFLOWS")
        split #(vec (remove str/blank? (str/split % #":")))]
    (cond
      (string? cli) (split cli)
      from-env (split from-env)
      :else (vec (concat ["workflows"]
                         (:workflow-dirs config)
                         [(str (config-dir env) "/workflows")])))))
