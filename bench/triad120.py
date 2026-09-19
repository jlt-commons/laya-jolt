"""Build bench/data/triad120.jsonl: 40 balanced cases each from AG News (test),
BoolQ (validation) and SST-5 (test), fetched from the Hub's datasets-server.
The same seed gives the same rows. The texts are not checked in."""
import json, random, urllib.request

def rows(ds, cfg, split, n=400):
    out = []
    while len(out) < n:
        u = f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={len(out)}&length=100"
        r = [x["row"] for x in json.load(urllib.request.urlopen(u))["rows"]]
        if not r:
            break
        out += r
    return out[:n]

random.seed(20260919)
cases = []
ag = rows("fancyzhx/ag_news", "default", "test")
labels = ["World", "Sports", "Business", "Sci/Tech"]
for i in range(4):
    for r in random.sample([r for r in ag if r["label"] == i], 10):
        cases.append({"task": "ag_news", "type": "choice", "state": r["text"],
                      "instructions": "Which topic does this news article belong to?",
                      "criteria": {"World": "world news and politics", "Sports": "sports",
                                   "Business": "business and finance", "Sci/Tech": "science and technology"},
                      "expected": labels[i]})
bq = rows("google/boolq", "default", "validation")
for b in (True, False):
    for r in random.sample([r for r in bq if r["answer"] == b], 20):
        cases.append({"task": "boolq", "type": "noul", "state": {"passage": r["passage"], "question": r["question"]},
                      "instructions": "Based on the passage, is the answer to the question yes?", "expected": b})
sst = rows("SetFit/sst5", "default", "test")
for i in range(5):
    for r in random.sample([r for r in sst if r["label"] == i], 8):
        cases.append({"task": "sst5", "type": "score", "state": r["text"],
                      "instructions": "How positive is the sentiment of this movie review?",
                      "criteria": ["very negative", "negative", "neutral", "positive", "very positive"], "expected": i})
with open("bench/data/triad120.jsonl", "w") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")
print(len(cases), "cases -> bench/data/triad120.jsonl")
