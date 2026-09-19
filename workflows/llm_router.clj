(ns workflows.llm-router
  "Route a request to a small or a frontier language model (laya 0.3.0
  presets.router_questions): difficulty, domain, tool needs, sensitivity.
  Not to be confused with lev.router, which picks a Laya checkpoint.
  State: {\"request\" text}; a bare string is taken as the request.")

(defn questions
  "LLM request routing: difficulty, domain, needs tools, is sensitive."
  []
  (array-map
   "difficulty" (array-map
                 "type" "score"
                 "instructions" "How hard is `request` for a language model?"
                 "criteria" ["trivial: a lookup or one-liner"
                             "easy: short answer, no reasoning"
                             "moderate: several steps"
                             "hard: long multi-step reasoning or specialist knowledge"])
   "domain" (array-map
             "type" "choice"
             "instructions" "What domain does `request` belong to?"
             "criteria" (array-map
                         "code" "software engineering, programming, refactoring, architecture, debugging"
                         "math_or_logic" "mathematics, logic puzzles, proofs, complex calculation"
                         "writing" "creative writing, essays, emails, blog posts, copywriting"
                         "factual_lookup" "facts, definitions, trivia, history"
                         "data_analysis" "statistics, SQL, data manipulation, metrics"
                         "chitchat" "casual conversation, greetings, small talk"))
   "needs_tools" (array-map
                  "type" "noul"
                  "instructions" "Does answering `request` require external tools, search or private data?")
   "is_sensitive" (array-map
                   "type" "noul"
                   "instructions" "Does `request` involve money, legal, medical or safety consequences?")))

(defn state [input]
  (if (map? input) input {"request" (str input)}))
