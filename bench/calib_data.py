"""Build bench/data/calib.jsonl: labeled cases for lev.calibrate covering
every temperature bucket the encoders carry, from public datasets on the
Hub (datasets-server, no auth), balanced per class. The texts are not
checked in; the same seed gives the same rows.

  choice:2      SST-2 (validation)            200
  choice:3-5    AG News (test)                200
  choice:6-10   dair-ai/emotion (test)        240
  choice:11+    SetFit/20_newsgroups (test)   200
  noul:2        BoolQ (validation)            200
  score:3-5     SST-5 (test)                  200
"""
import json, random, time, urllib.error, urllib.request

def fetch(u, tries=10):
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(u))
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(5 * (i + 1))   # the datasets-server rate limit

def rows(ds, cfg, split, n):
    out = []
    while len(out) < n:
        u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={len(out)}&length=100"
        r = [x["row"] for x in fetch(u)["rows"]]
        if not r:
            break
        out += r
        time.sleep(1.5)
    return out[:n]

def balanced(rs, key, per, classes):
    picked = []
    for c in classes:
        xs = [r for r in rs if r[key] == c]
        picked += random.sample(xs, min(per, len(xs)))
    random.shuffle(picked)
    return picked

random.seed(20260919)
cases = []

sst2 = balanced(rows("stanfordnlp/sst2", "default", "validation", 800), "label", 100, [0, 1])
for r in sst2:
    cases.append({"task": "sst2", "state": r["sentence"],
                  "questions": {"q": {"type": "choice", "instructions": "What is the sentiment of this movie review?",
                                      "criteria": {"negative": "a negative review", "positive": "a positive review"}}},
                  "labels": {"q": ["negative", "positive"][r["label"]]}})

ag = balanced(rows("fancyzhx/ag_news", "default", "test", 800), "label", 50, [0, 1, 2, 3])
for r in ag:
    cases.append({"task": "ag_news", "state": r["text"],
                  "questions": {"q": {"type": "choice", "instructions": "Which topic does this news article belong to?",
                                      "criteria": {"World": "world news and politics", "Sports": "sports",
                                                   "Business": "business and finance", "Sci/Tech": "science and technology"}}},
                  "labels": {"q": ["World", "Sports", "Business", "Sci/Tech"][r["label"]]}})

emo = balanced(rows("dair-ai/emotion", "split", "test", 2000), "label", 40, list(range(6)))
names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
for r in emo:
    cases.append({"task": "emotion", "state": r["text"],
                  "questions": {"q": {"type": "choice", "instructions": "Which emotion does the writer express?",
                                      "criteria": {n: None for n in names}}},
                  "labels": {"q": names[r["label"]]}})

ng = rows("SetFit/20_newsgroups", "default", "test", 1500)
ng_names = sorted(set(r["label_text"] for r in ng))
ngb = balanced(ng, "label_text", 10, ng_names)
for r in ngb:
    cases.append({"task": "newsgroups", "state": r["text"][:1500],
                  "questions": {"q": {"type": "choice", "instructions": "Which newsgroup was this post made to?",
                                      "criteria": {n: None for n in ng_names}}},
                  "labels": {"q": r["label_text"]}})

bq = balanced(rows("google/boolq", "default", "validation", 800), "answer", 100, [True, False])
for r in bq:
    cases.append({"task": "boolq", "state": {"passage": r["passage"], "question": r["question"]},
                  "questions": {"q": {"type": "noul", "instructions": "Based on the passage, is the answer to the question yes?"}},
                  "labels": {"q": r["answer"]}})

sst5 = balanced(rows("SetFit/sst5", "default", "test", 1200), "label", 40, list(range(5)))
for r in sst5:
    cases.append({"task": "sst5", "state": r["text"],
                  "questions": {"q": {"type": "score", "instructions": "How positive is the sentiment of this movie review?",
                                      "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}},
                  "labels": {"q": r["label"]}})

random.shuffle(cases)
with open("bench/data/calib.jsonl", "w") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")
print(len(cases), "cases -> bench/data/calib.jsonl")
