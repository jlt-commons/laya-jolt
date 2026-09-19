#!/usr/bin/env python3
"""Dump golden traces from the torch CPU float32 oracle into golden/.

Everything runs in eval mode, CPU, float32, eager attention (the eager path
materializes additive masks we can capture). The RLAgent quickstart answers
are ALSO captured through the real sdpa path as shipped, so final-answer
parity is checked against shipped behavior, not just our eager variant.

Outputs EDN files under golden/:
  cases.edn       - the fixed inputs (texts, questions, states) so the jolt
                    side rebuilds identical sequences from source data
  tok.edn         - input_ids per text (tokenizer parity)
  masks.edn       - full + sliding allowed-matrices for a padded batch
  layers.edn      - small case: embeddings, per-layer outputs (full vectors
                    for layers 0,1,2,27 + max-abs stats for all), final norm,
                    rope freqs/cos/sin for both thetas, marker gather, head
                    intermediates, logits, act; plus the ReLU-vs-GELU gap
  readme.edn      - README quickstart: input_ids per question, per-layer
                    max-abs stats, logits, act probs, RLAgent.system_one JSON
  email.edn       - email_utils.py reference: clean_email_body in/out pairs,
                    email_state and email_questions as JSON strings
  sequences.edn   - build_sequence ids/markers for the branches the README
                    case never takes (truncation, option shrink, [MASK] in
                    the state, non-string instructions, noul criteria, ...)
  email_answers.edn - system_one on email_state + email_questions (+ a
                    14-option choice) for two emails: the end-to-end check
                    on question shapes beyond the quickstart
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "laya")))
from rl_common import (QTYPES, DecisionModel, build_model, build_sequence,  # noqa: E402
                       collate_items, render_options)
from rl_agent_api import RLAgent  # noqa: E402
import email_utils  # noqa: E402

EMAIL_BODIES = [
    # sign-off near the end
    "Hi team,\n\nWe were billed twice for March. Please refund the duplicate.\n\nBest regards,\nJohn Smith\nAcme Corp",
    # quoted reply header + quoted lines
    "Thanks, that fixed it.\n\nOn Mon, Jan 8, 2024 at 10:02 AM Bob <bob@example.com> wrote:\n> Can you try clearing the cache?\n> It usually helps.\n\nBob",
    # outlook-style original message + From: header
    "Please see below.\n\n-----Original Message-----\nFrom: Alice <alice@example.com>\nSent: Tuesday\nTo: Bob\nSubject: RE: invoice\n\nold thread body",
    # From: line alone breaks the thread
    "Forwarding for visibility.\nFrom: someone@example.com\nthis is the old mail",
    # CRLF, CR and literal backslash-n
    "Line one\r\nLine two\rLine three\\nLine four\r\n\r\nCheers,\nMaria",
    # disclaimer paragraphs are dropped
    "Order #4411 arrived damaged.\n\nThis message is confidential and intended solely for the named addressee.\n\nIf you have received this e-mail in error please notify the sender.\n\nPlease advise.",
    # long body is truncated to 3000 chars
    ("word " * 800).strip(),
    # whitespace collapse + multiple blank lines
    "Hello   there\t\tfriend.\n\n\n\n\nSecond    paragraph.\n \n\nThird.",
    # sent-from marker
    "Can we reschedule to 3pm?\n\nSent from my iPhone",
    # -- delimiter
    "See attached invoice.\n\n--\nJane Doe\nCFO",
    # short mail: line 0 is never a sign-off; 'Thanks' on line 1 is
    "Thanks\nfor the update\nThanks",
    # sign-off longer than 40 chars does not cut
    "Regards and thank you for all the help you gave us during this very long and difficult migration project\nBob",
    # sign-off with punctuation allowed by the class, and one rejected by it
    "Fixed.\nThanks a lot, see you!\nnot cut here\nThanks - Bob\nstill here",
    # empty and whitespace only
    "",
    "   \n\t\n",
    # non-ascii + NBSP whitespace + accented word chars
    "Bonjour,\n\nLa facture\u00a0#22 est en double.\n\nMerci beaucoup,\nJean-Pierre \u00c9tienne",
    # quote header as the very first line is kept (nothing before it)
    "On Friday, someone wrote:\nthe actual content\n> quoted",
    # case-insensitive markers, indented
    "  ON 2 FEB 2024, X WROTE:\nshould not be reached",
    "Update attached.\n\n   best REGARDS,\n   Sam",
    # underscore rule and '>' lines interleaved
    "Top.\n> a quote\nMiddle.\n________\nBelow the rule",
]

TOK_CASES = [
    "Which department should handle this email?",
    " choice question: invoices, payments, refunds",
    "  leading double space and\ttab",
    "Don't stop — it's 2024, isn't it? naïve café résumé ﬁne",
    "unicodé combining acute (NFD source; NFC must fold)",
    "MixedCASE WithNumbers 123 45.67 and #hashtags @at",
    "level 0: not urgent",
    "true: yes, the statement holds",
    "Duplicate billing on March invoice #4411",
    "https://example.com/path?q=1&r=2&x=%20",
    "emoji 😀 and cjk 中文测试",
    "$1,234.56 —  -99.2e-3  +0.5",
    # whitespace semantics: only U+0020 is the optional token prefix; \s is
    # Unicode White_Space (U+001C..1F are NOT whitespace; NBSP/U+2028 are)
    "and\tThe end",
    "line1\nline2",
    "a\u001cb\u001fc",
    "x\u00a0y",
    "tab\t\tdouble",
    "nl\n\n\nrun",
    "a \tb",
    "DON'T Shout 'S",
    "end   ",
    "\u00a0\u00a0lead",
    "\u2028sep",
    "'re're 're",
    "z\n\n\n\n\nq",
    "\r\nThe",
]


def f32(x):
    a = np.asarray(x, dtype=np.float32)
    return a


def fmt(v):
    v = float(v)
    if v != v or math.isinf(v):
        return "##NaN" if v != v else ("##Inf" if v > 0 else "##-Inf")
    return "%.9g" % np.float32(v)


def edn_vec(a):
    return "[" + " ".join(fmt(v) for v in np.asarray(a).ravel()) + "]"


def edn_mat(a):
    a = np.asarray(a)
    return "[" + " ".join(edn_vec(r) for r in a.reshape(a.shape[0], -1)) + "]"


def write_f32(path, a):
    a = np.asarray(a, dtype=np.float32)
    a.tofile(path)
    return a.shape


def edn_ref(k, shape, fn):
    dims = " ".join(str(int(d)) for d in shape)
    return " :%s {:shape [%s] :file %s}\n" % (k, dims, '"' + fn + '"')


def write_edn(path, body):
    with open(path, "w") as f:
        f.write(body)
    print("wrote", path, os.path.getsize(path), "bytes")


def hook_capture(model):
    caps = {"emb": None, "layers": {}, "final": None, "head_layers": {},
            "attn_masks": {}}

    def emb_h(_m, _i, out):
        caps["emb"] = out.detach().clone()

    def final_h(_m, _i, out):
        caps["final"] = out.detach().clone()

    def layer_h(i):
        def h(_m, _i, out):
            caps["layers"][i] = out.detach().clone()
        return h

    def head_layer_h(i):
        def h(_m, _i, out):
            caps["head_layers"][i] = out.detach().clone()
        return h

    def attn_h(i):
        def h(mod, args, kwargs):
            m = kwargs.get("attention_mask", None)
            if m is not None and i not in caps["attn_masks"]:
                caps["attn_masks"][i] = m.detach().clone()
        return h

    hs = [model.encoder.embeddings.register_forward_hook(emb_h),
          model.encoder.final_norm.register_forward_hook(final_h)]
    for i, layer in enumerate(model.encoder.layers):
        hs.append(layer.register_forward_hook(layer_h(i)))
        hs.append(layer.attn.register_forward_pre_hook(attn_h(i), with_kwargs=True))
    for i, layer in enumerate(model.head.layers):
        hs.append(layer.register_forward_hook(head_layer_h(i)))
    return caps, hs


def stats_of(t):
    a = t.detach().to(torch.float32).numpy()
    return float(np.abs(a).max()), float(np.sqrt((a.astype(np.float64) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", default="../laya")
    ap.add_argument("--out", default="golden")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    laya = os.path.abspath(args.laya)
    torch.manual_seed(0)

    agent = RLAgent(laya, device="cpu")
    tok = agent.tok
    cfg = agent.cfg
    hf_tok = tok._tokenizer

    # byte-level alphabet inverse (vocab chars -> raw bytes), so pieces can be
    # decoded back to the text they were cut from
    _n = 0
    INV = {}
    for b in range(256):
        if (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255):
            INV[chr(b)] = b
        else:
            INV[chr(256 + _n)] = b
            _n += 1

    def unbyte(piece):
        if not all(c in INV for c in piece):
            return piece  # an added token (e.g. a run of spaces) is emitted verbatim
        return bytes(INV[c] for c in piece).decode("utf-8", errors="replace")

    # ---- cases.edn: the inputs the jolt side must rebuild identically -----
    readme_state = {
        "from": "customer@acme.com",
        "subject": "Duplicate billing on March invoice #4411",
        "body": "Hi team, we were billed twice for March. Please refund the duplicate before Friday or we will cancel our plan.",
    }
    readme_qs = {
        "department": {"type": "choice",
                       "instructions": "Which department should handle this email?",
                       "criteria": {"billing": "invoices, payments, refunds",
                                    "technical": "bugs, outages, integrations",
                                    "sales": "pricing, contracts, demos",
                                    "other": "everything else"}},
        "urgency": {"type": "score",
                    "instructions": "How urgent is this request?",
                    "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
        "churn_risk": {"type": "noul",
                       "instructions": "Does the user threaten to cancel or switch to a competitor?"},
        "is_phishing": {"type": "noul",
                        "instructions": "Is this email a phishing or scam attempt?"},
    }
    def edn_escape(s):
        out = []
        for ch in s:
            if ch == "\\":
                out.append("\\\\")
            elif ch == '"':
                out.append('\\"')
            elif ch == "\n":
                out.append("\\n")
            elif ch == "\t":
                out.append("\\t")
            elif ord(ch) < 32:
                out.append("\\u%04x" % ord(ch))
            else:
                out.append(ch)
        return "".join(out)

    def edn_str_of(x):
        return '"' + edn_escape(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)) + '"'

    def edn_of(x):
        # dicts become #laya/omap [[k v] ...]: an EDN map literal past 8 keys
        # reads back as a hash-map and the tests need Python's insertion order
        if x is None:
            return "nil"
        if x is True or x is False:
            return "true" if x else "false"
        if isinstance(x, str):
            return edn_str_of(x)
        if isinstance(x, dict):
            return "#laya/omap [" + " ".join("[%s %s]" % (edn_str_of(k), edn_of(v)) for k, v in x.items()) + "]"
        if isinstance(x, list):
            return "[" + " ".join(edn_of(v) for v in x) + "]"
        return repr(x)

    with open(os.path.join(args.out, "cases.edn"), "w") as f:
        f.write("{:tok-cases [\n")
        for t in TOK_CASES:
            f.write("  %s\n" % edn_str_of(t))
        f.write("]\n")
        # pre-token boundaries straight from the HF tokenizer, so the jolt
        # scanner is validated against the real thing, not a regex re-derivation
        f.write(" :pre-tok {\n")
        for t in TOK_CASES:
            enc = hf_tok.encode(t, add_special_tokens=False)
            toks = [unbyte(tok) for tok in enc.tokens]
            f.write("  %s %s\n" % (edn_str_of(t), edn_of(toks)))
        f.write("}\n")
        # raw ByteLevel pre-tokenizer boundaries (pre_tokenize_str: no added
        # tokens, no normalizer) for the scanner unit test
        f.write(" :pre-tokens {\n")
        for t in TOK_CASES:
            pieces = [p for p, _ in hf_tok.pre_tokenizer.pre_tokenize_str(t)]
            toks = [unbyte(p) for p in pieces]
            f.write("  %s %s\n" % (edn_str_of(t), edn_of(toks)))
        f.write("}\n")
        f.write(" :readme-state %s\n" % edn_of(readme_state))
        f.write(" :readme-questions %s}\n" % edn_of(readme_qs))
    print("wrote", os.path.join(args.out, "cases.edn"))

    # ---- sequences.edn: build_sequence branches the README case never takes --
    long_opts = {"option number %d" % i: "a fairly long description of this option that runs on for a while to eat the head budget" for i in range(12)}
    SEQ_CASES = [
        ("mask-injection", {"note": "[MASK] and [CLS] inside the state", "text": "please [MASK] this [CLS] token [SEP]"},
         {"type": "noul", "instructions": "Is there a [MASK] token?"}),
        ("long-state-truncates", {"body": ("word " * 600).strip()}, {"type": "noul", "instructions": "Is it long?"}),
        ("many-long-options-shrink", "state", {"type": "choice", "instructions": "pick", "criteria": long_opts}),
        ("long-instructions-truncate", "state", {"type": "score", "instructions": ("x " * 300).strip(), "criteria": ["a", "b", "c"]}),
        ("dict-instructions-ascii", {"a": 1}, {"type": "noul", "instructions": {"rule": "caf\u00e9 \u2265 3", "n": 2.5, "ok": True, "none": None}}),
        ("choice-null-desc", "state", {"type": "choice", "instructions": "which", "criteria": {"a": None, "b": "", "c": "desc", "d": 0}}),
        ("choice-list", "state", {"type": "choice", "instructions": "which", "criteria": ["x", "y", "z"]}),
        ("noul-criteria", "state", {"type": "noul", "instructions": "q", "criteria": {"true": "yes it is", "false": "no"}}),
        ("score-many-levels", "s", {"type": "score", "instructions": "rate", "criteria": ["level %d desc" % i for i in range(11)]}),
        ("unicode-state", {"from": "jos\u00e9@example.com", "body": "Merci \u2014 r\u00e9sum\u00e9 \u4e2d\u6587 \U0001F600 \u201cquotes\u201d\ttab\nnl"},
         {"type": "noul", "instructions": "Is it non-ascii?"}),
        ("numeric-state", {"amount": 0.0001, "big": 1e16, "n": 3, "ok": True, "none": None, "list": [1.5, 2, "x"]},
         {"type": "noul", "instructions": "numbers?"}),
        ("array-state", [{"role": "user", "text": "hi"}, {"role": "agent", "text": "hello"}],
         {"type": "score", "instructions": "tone", "criteria": ["bad", "ok", "good"]}),
    ]
    body = "{"
    for name, state, qdef in SEQ_CASES:
        q = RLAgent._to_internal(qdef)
        seq, markers = build_sequence(tok, state, q, cfg["max_len"], cfg["head_max_len"])
        body += " %s {:state %s :question %s :ids %s :markers %s}\n" % (
            edn_str_of(name), edn_of(state), edn_of(qdef), edn_vec(seq), edn_vec(markers))
    body += "}\n"
    write_edn(os.path.join(args.out, "sequences.edn"), body)

    # ---- tok.edn ----------------------------------------------------------
    body = "{:cases {\n"
    for i, t in enumerate(TOK_CASES):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        body += " \"%d\" %s\n" % (i, edn_vec(ids))
    body += "}}\n"
    write_edn(os.path.join(args.out, "tok.edn"), body)

    # ---- eager float32 model for layer capture ----------------------------
    from transformers import AutoConfig, AutoModel
    ecfg = AutoConfig.from_pretrained(os.path.join(laya, "encoder"))
    enc = AutoModel.from_config(ecfg, attn_implementation="eager")
    model = DecisionModel(enc, cfg["head_layers"], len(cfg["act_costs"]) + 1)
    from safetensors.torch import load_file
    model.load_state_dict(load_file(os.path.join(laya, "model.safetensors")), strict=True)
    model.to("cpu").eval().float()
    ecfg.reference_compile = False

    assert model.head.layers[0].activation is F.relu, \
        "torch TransformerEncoderLayer default activation is not relu - trace assumption broken"
    assert len(model.head.layers) == 2

    # ---- masks.edn: exact allowed matrices for a padded batch -------------
    B, L = 2, 64
    lens = [64, 45]
    pad = torch.zeros(B, L, dtype=torch.long)
    for b, n in enumerate(lens):
        pad[b, :n] = 1
    ids = torch.randint(0, 50000, (B, L))
    for b, n in enumerate(lens):  # keep pad ids = pad token
        ids[b, n:] = tok.pad_token_id
    with torch.no_grad():
        h0 = model.encoder.embeddings(ids)
        from transformers.masking_utils import (create_bidirectional_mask,
                                                create_bidirectional_sliding_window_mask)
        mfull = create_bidirectional_mask(config=ecfg, inputs_embeds=h0, attention_mask=pad)
        mslid = create_bidirectional_sliding_window_mask(config=ecfg, inputs_embeds=h0, attention_mask=pad)
    # eager interface: additive float masks, 0 = allowed, min = masked
    def allowed(m):
        a = m.detach().float().numpy()
        return (a == 0).astype(np.int8)
    body = "{:lens [%d %d]\n :full %s\n :sliding %s}\n" % (
        lens[0], lens[1], edn_mat(allowed(mfull[1])), edn_mat(allowed(mslid[1])))
    write_edn(os.path.join(args.out, "masks.edn"), body)

    # ---- layers.edn: small forward, hooks on every layer ------------------
    small_state = {"subject": "Duplicate billing on March invoice #4411",
                   "body": "We were billed twice for March. Please refund the duplicate."}
    q1 = {"t": "choice", "ins": "Which department should handle this email?",
          "crit": {"billing": "invoices, payments, refunds", "technical": "bugs and outages",
                   "sales": "pricing and demos", "other": "everything else"}}
    q2 = {"t": "noul", "ins": "Does the user threaten to cancel?", "crit": None}
    items = []
    for q in (q1, q2):
        seq, markers = build_sequence(tok, small_state, q, cfg["max_len"], cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                      "target": [0.0] * len(markers), "label": -1, "episode": 0,
                      "ep_step": 0, "ep_len": 1, "src": "api"})
    b = collate_items([items], tok.pad_token_id)
    caps, hs = hook_capture(model)
    with torch.no_grad():
        logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                            b["marker_mask"], b["qtype"])
    for h in hs:
        h.remove()
    assert torch.isfinite(logits).all() and torch.isfinite(act).all()

    os.makedirs(os.path.join(args.out, "layers"), exist_ok=True)
    refs = []
    def sidecar(k, a):
        a = np.asarray(a, dtype=np.float32)
        fn = "layers/%s.f32" % k
        a.tofile(os.path.join(args.out, fn))
        refs.append(edn_ref(k, a.shape, fn))

    body = "{:ids %s\n :att %s\n :marker-pos %s\n :marker-mask %s\n :qtype %s\n" % (
        edn_mat(b["input_ids"].numpy()), edn_mat(b["attention_mask"].numpy()),
        edn_mat(b["marker_pos"].numpy()), edn_mat(b["marker_mask"].long().numpy()),
        edn_vec(b["qtype"].numpy()))
    sidecar("embeddings", caps["emb"].numpy())
    for i in range(len(model.encoder.layers)):
        t = caps["layers"][i]
        if i in (0, 1, 2, 27):
            sidecar("layer-%d" % i, t.numpy())
        mx, l2 = stats_of(t)
        body += " :layer-%d-stats [%.9g %.9g]\n" % (i, mx, l2)
    sidecar("final-norm", caps["final"].numpy())

    # rope tables for both thetas at L positions
    Lp = caps["final"].shape[1]
    pos = torch.arange(Lp).unsqueeze(0)
    for name, lt in (("full", "full_attention"), ("sliding", "sliding_attention")):
        cos, sin = model.encoder.rotary_emb(caps["emb"], pos, lt)
        sidecar("rope-%s-cos" % name, cos[0].numpy())
        sidecar("rope-%s-sin" % name, sin[0].numpy())
    # captured additive masks -> allowed
    sidecar("mask-full", allowed(caps["attn_masks"][0]))
    sidecar("mask-sliding", allowed(caps["attn_masks"][1]))
    with torch.no_grad():
        sidecar("head-in", (caps["final"] + model.type_emb(b["qtype"])[:, None, :]).numpy())
    for i in range(len(model.head.layers)):
        sidecar("head-layer-%d" % i, caps["head_layers"][i].numpy())
    # marker gather (what the scorer sees)
    m = torch.gather(caps["head_layers"][1], 1,
                     b["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, caps["head_layers"][1].size(-1)))
    sidecar("markers", m.numpy())
    sidecar("logits", logits.numpy())
    sidecar("act-softmax", torch.softmax(act.float(), -1).numpy())
    # act-head input features, verbatim from DecisionModel.forward
    with torch.no_grad():
        pf = torch.softmax(logits.detach(), -1)
        kf = b["marker_mask"].sum(-1).clamp(min=2).float()
        entf = -(pf * torch.log(pf.clamp_min(1e-9))).sum(-1) / torch.log(kf)
        top2f = pf.topk(2, -1).values
        feats = torch.stack([top2f[:, 0], top2f[:, 0] - top2f[:, 1], entf, kf / 255.0], -1)
    body += " :act-feats %s\n" % edn_mat(feats.numpy())
    # matmul oracle: first 8 rows of head-layer-0 out @ scorer.1.weight^T
    with torch.no_grad():
        X8 = caps["head_layers"][0].reshape(-1, caps["head_layers"][0].size(-1))[:8]
        sidecar("matmul-x", X8.numpy())
        sidecar("matmul", (X8 @ model.scorer[1].weight.T).numpy())

    # ReLU-vs-GELU discrimination. The layer's self.activation monkeypatch is
    # ignored by torch's fast path (_transformer_encoder_layer_fwd), so force
    # the FF activation at the linear2 input instead, where it cannot be skipped.
    def force_act(fn):
        def hook(_m, args):
            return (fn(args[0]),)
        return hook
    hs = [l.linear2.register_forward_pre_hook(force_act(F.gelu)) for l in model.head.layers]
    with torch.no_grad():
        glogits, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                           b["marker_mask"], b["qtype"])
    for h in hs:
        h.remove()
    sidecar("gelu-logits", glogits.numpy())
    body += " :relu-vs-gelu-gap %.9g\n" % float((logits - glogits).abs().max())
    body += "".join(refs)
    body += "}\n"
    write_edn(os.path.join(args.out, "layers.edn"), body)

    # ---- readme.edn: the shipped quickstart through RLAgent (sdpa) --------
    result = agent.system_one(readme_state, readme_qs)
    body = "{:input-ids {\n"
    for qid, qd in readme_qs.items():
        q = RLAgent._to_internal(qd)
        seq, markers = build_sequence(tok, readme_state, q, cfg["max_len"], cfg["head_max_len"])
        body += " %s {:ids %s :markers %s :qtype %d}\n" % (
            json.dumps(qid), edn_vec(seq), edn_vec(markers), QTYPES[q["t"]])
    body += "}\n"
    # per-layer stats through the sdpa encoder as shipped
    caps2, hs2 = hook_capture(agent.model)
    with torch.no_grad():
        logits2, act2 = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"],
                                    b["marker_mask"], b["qtype"])
    for h in hs2:
        h.remove()
    for i in range(len(agent.model.encoder.layers)):
        mx, l2 = stats_of(caps2["layers"][i])
        body += " :sdpa-layer-%d-stats [%.9g %.9g]\n" % (i, mx, l2)
    body += " :system-one %s}\n" % edn_str_of(json.dumps(result))
    write_edn(os.path.join(args.out, "readme.edn"), body)

    # ---- email_answers.edn: system_one on email_state + email_questions -------
    wide = {"type": "choice", "instructions": "Pick the closest topic.",
            "criteria": {k: None for k in ["billing", "refund", "bug", "outage", "login", "pricing", "demo",
                                           "hiring", "payroll", "legal", "shipping", "returns", "feedback", "other"]}}
    eqs = email_utils.email_questions()
    eqs["wide"] = wide
    body = "{:questions %s\n :cases [\n" % edn_of(eqs)
    for idx in (0, 5):
        st = email_utils.email_state("Support request", EMAIL_BODIES[idx], sender="someone@example.com")
        res = agent.system_one(st, eqs)
        body += "  {:body-index %d :state %s :result %s}\n" % (idx, edn_of(st), edn_str_of(json.dumps(res)))
    body += " ]}\n"
    write_edn(os.path.join(args.out, "email_answers.edn"), body)

    # ---- email.edn: email_utils.py reference outputs -----------------------
    body = "{:clean [\n"
    for raw in EMAIL_BODIES:
        body += "  [%s %s]\n" % (edn_str_of(raw), edn_str_of(email_utils.clean_email_body(raw)))
    body += " [%s %s]\n" % (edn_str_of(EMAIL_BODIES[6]), edn_str_of(email_utils.clean_email_body(EMAIL_BODIES[6], max_chars=100)))
    body += "]\n :max-chars-case %s\n" % edn_str_of(EMAIL_BODIES[6])
    states = [
        email_utils.email_state("  Duplicate billing  ", EMAIL_BODIES[0], sender="customer@acme.com"),
        email_utils.email_state(None, None),
        email_utils.email_state("raw", EMAIL_BODIES[1], clean=False),
        email_utils.email_state("extras", "body text", sender="a@b.c", priority=2, tag=None, score=0.5, flag=True),
    ]
    body += " :states %s\n" % edn_of([json.dumps(s, ensure_ascii=False) for s in states])
    body += " :questions %s\n" % edn_str_of(json.dumps(email_utils.email_questions(), ensure_ascii=False))
    body += " :questions-custom %s}\n" % edn_str_of(json.dumps(
        email_utils.email_questions({"refund": "money back", "bug": "it is broken"}), ensure_ascii=False))
    write_edn(os.path.join(args.out, "email.edn"), body)


if __name__ == "__main__":
    main()
