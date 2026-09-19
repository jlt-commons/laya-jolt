(ns laya.workflows
  "User-defined workflows: a use case packaged as a namespace.

  A workflow is a file <dir>/<name>.clj defining `workflows.<name>`
  (underscores in the file name become dashes) with

    (defn questions ([] ...) ([opts] ...))   ; the typed question map, required;
                                             ; the 1-arity is optional and takes
                                             ; the caller's options map
    (defn state [input] ...)                 ; optional: raw input -> model state
                                             ; (without it the input is the state)

  The `questions` docstring is the workflow's description. The bundled ones
  live in ./workflows (email, demo); users add theirs under
  ~/.config/laya/workflows or wherever config.edn :workflow-dirs / the
  LAYA_WORKFLOWS variable point (see laya.config/workflow-dirs)."
  (:require [clojure.java.io :as io]
            [clojure.string :as str]))

(defn workflow-name
  "File stem -> workflow name: hello_world.clj -> \"hello-world\"."
  [file]
  (-> (.getName (io/file file)) (str/replace #"\.clj$" "") (str/replace "_" "-")))

(defn- clj-files [dir]
  (let [d (io/file dir)]
    (when (.isDirectory d)
      (->> (.listFiles d)
           (filter #(and (.isFile %) (str/ends-with? (.getName %) ".clj")))
           (sort-by #(.getName %))))))

(defn- fail [file msg data]
  (throw (ex-info (str "workflow " file ": " msg) (assoc data :file file))))

(defn load-workflow
  "load-file one workflow and describe it: {:name :ns :file :questions :state
  :options? :doc}. Throws with the file name in the message when the file
  does not load, defines the wrong namespace, or has no `questions`."
  [file]
  (let [path (.getPath (io/file file))
        name (workflow-name path)
        ns-sym (symbol (str "workflows." name))]
    ;; jolt's load-file leaves *ns* on the file's namespace; keep the caller's
    (try (binding [*ns* *ns*] (load-file path))
         (catch Exception e
           (fail path (str "failed to load: " (ex-message e)) {:cause e})))
    (let [n (or (find-ns ns-sym)
                (fail path (str "must define the namespace " ns-sym " (it is named after the file)") {:ns ns-sym}))
          qv (or (ns-resolve n 'questions)
                 (fail path (str "defines no `questions` fn in " ns-sym) {:ns ns-sym}))
          sv (ns-resolve n 'state)
          arglists (:arglists (meta qv))]
      {:name name
       :ns ns-sym
       :file path
       :questions @qv
       :state (when sv @sv)
       :options? (boolean (some #(= 1 (count %)) arglists))
       :doc (:doc (meta qv))})))

(defn load-workflows
  "Load every *.clj under each directory, in order; a later directory's
  workflow replaces an earlier one of the same name. Missing directories
  are skipped. Returns {name workflow}."
  [dirs]
  (reduce (fn [acc dir]
            (reduce (fn [acc f] (let [w (load-workflow f)] (assoc acc (:name w) w)))
                    acc (clj-files dir)))
          {} dirs))

(defn questions
  "The workflow's question map, with the caller's options when it takes them."
  ([wf] ((:questions wf)))
  ([wf opts]
   (cond
     (empty? opts) ((:questions wf))
     (:options? wf) ((:questions wf) opts)
     :else (throw (ex-info (str "workflow " (:name wf) " takes no options; got " (pr-str (keys opts)))
                           {:type :invalid-request :workflow (:name wf) :options opts})))))

(defn state
  "Build the model state from the raw input, or pass it through when the
  workflow has no `state` fn."
  [wf input]
  (if-let [f (:state wf)] (f input) input))
