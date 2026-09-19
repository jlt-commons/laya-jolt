#!/usr/bin/env python3
"""Reference converter: the Laya checkpoint -> jolt-friendly artifacts.

This is the oracle for `jolt prepare` (src/laya/prepare.clj), which is the
conversion users run. With --golden it also writes the size and zlib CRC-32
of every file it produced, and test/laya/prepare_test.clj checks that the
jolt conversion reproduces them byte for byte.

- model.safetensors -> <out>/model/<tensor>.f32 raw little-endian float32
- <out>/manifest.edn  — tensor name -> {shape file}
- tokenizer.json      -> <out>/tokenizer.edn {vocab merges specials added}
- encoder config.json + rl_agent_config.json -> <out>/config.edn

python3 + numpy only (no torch needed here). F16->F32 is an exact upcast, so
the blobs are numerically identical to torch's upcast weights.
"""
import argparse
import json
import os
import struct
import zlib

import numpy as np

DTYPES = {"F16": "<f2", "F32": "<f4", "BF16": None, "I64": "<i8", "U8": "u1"}


def esc(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 32:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def edn_str(s):
    return '"' + esc(s) + '"'


def edn_map(kvs, indent=""):
    parts = []
    for k, v in kvs:
        parts.append("%s%s %s" % (indent, k, v))
    return "{" + " ".join(parts) + "}"


def read_safetensors_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n)), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", default="../laya")
    ap.add_argument("--out", default="data")
    ap.add_argument("--golden", default=None,
                    help="also write {file -> size, crc32} of everything produced (golden/prepare.edn)")
    args = ap.parse_args()
    laya, out = args.laya, args.out
    os.makedirs(os.path.join(out, "model"), exist_ok=True)

    # ---- weights ---------------------------------------------------------
    entries = {}
    with open(os.path.join(laya, "model.safetensors"), "rb") as f:
        hdr, base = read_safetensors_header(f)
        for name, meta in sorted(hdr.items()):
            if name == "__metadata__":
                continue
            dt = DTYPES.get(meta["dtype"])
            if dt is None:
                raise SystemExit("unexpected dtype %s for %s" % (meta["dtype"], name))
            n = 1
            for d in meta["shape"]:
                n *= d
            f.seek(base + meta["data_offsets"][0])
            raw = f.read(meta["data_offsets"][1] - meta["data_offsets"][0])
            arr = np.frombuffer(raw, dtype=dt).astype(np.float32)
            if meta["dtype"] == "F16":
                assert arr.dtype == np.float32
            fn = "model/%s.f32" % name
            arr.tofile(os.path.join(out, fn))
            entries[name] = (meta["shape"], fn)
            print("  %-48s %-16s %s" % (name, meta["shape"], meta["dtype"]))

    with open(os.path.join(out, "manifest.edn"), "w") as f:
        f.write("{:format 1\n :tensors {\n")
        for name, (shape, fn) in sorted(entries.items()):
            f.write("  %s {:shape [%s] :file %s}\n"
                    % (edn_str(name), " ".join(str(s) for s in shape), edn_str(fn)))
        f.write("}}\n")

    # ---- tokenizer -------------------------------------------------------
    tok = json.load(open(os.path.join(laya, "tokenizer", "tokenizer.json")))
    vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"]
    added = {t["content"]: t["id"] for t in tok["added_tokens"]}
    specials = {k: added["[%s]" % k.upper()] for k in ("cls", "sep", "pad", "mask", "unk")}

    with open(os.path.join(out, "tokenizer.edn"), "w") as f:
        f.write("{:format 1\n")
        f.write(" :vocab {\n")
        for t, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            f.write("  %s %d\n" % (edn_str(t), i))
        f.write(" }\n :merges [\n")
        for m in merges:
            a, b = m if isinstance(m, list) else m.split(" ")
            f.write("  %s\n" % edn_str(a + " " + b))
        f.write(" ]\n :specials {\n")
        for k, i in sorted(specials.items()):
            f.write("  :%s %d\n" % (k, i))
        f.write(" }\n :added [\n")
        for t in sorted(tok["added_tokens"], key=lambda x: x["id"]):
            f.write("  [%s %d]\n" % (edn_str(t["content"]), t["id"]))
        f.write(" ]}\n")
    print("tokenizer: %d vocab, %d merges, %d added, specials %s"
          % (len(vocab), len(merges), len(tok["added_tokens"]), specials))

    # ---- config ----------------------------------------------------------
    enc = json.load(open(os.path.join(laya, "encoder", "config.json")))
    rl = json.load(open(os.path.join(laya, "rl_agent_config.json")))
    hidden = enc["hidden_size"]
    cfg = {
        ":hidden-size": hidden,
        ":num-layers": enc["num_hidden_layers"],
        ":num-heads": enc["num_attention_heads"],
        ":head-dim": hidden // enc["num_attention_heads"],
        ":intermediate": enc["intermediate_size"],
        # sliding-attention radius = local_attention // 2 (torch: config.sliding_window = local_attention // 2)
        ":window": enc["local_attention"] // 2,
        ":layer-types": enc["layer_types"],
        ":rope-full": enc["rope_parameters"]["full_attention"]["rope_theta"],
        ":rope-local": enc["rope_parameters"]["sliding_attention"]["rope_theta"],
        ":norm-eps": enc["norm_eps"],
        ":vocab-size": enc["vocab_size"],
        ":max-len": rl["max_len"],
        ":head-max-len": rl["head_max_len"],
        ":head-layers": rl["head_layers"],
        ":head-ffn": 4 * hidden,
        ":temperature": rl["temperature"],
        ":temperature-by-options": rl.get("temperature_by_options", {}),
    }
    with open(os.path.join(out, "config.edn"), "w") as f:
        f.write("{\n")
        for k, v in cfg.items():
            if isinstance(v, str):
                v = edn_str(v)
            elif isinstance(v, list):
                v = "[" + " ".join(edn_str(x) if isinstance(x, str) else str(x) for x in v) + "]"
            elif isinstance(v, dict):
                v = "{" + " ".join("%s %s" % (edn_str(a), b) for a, b in v.items()) + "}"
            f.write(" %s %s\n" % (k, v))
        f.write("}\n")
    print("config.edn written; %d tensors, total %.2f GB"
          % (len(entries), sum(s[0][0] * (s[0][1] if len(s[0]) > 1 else 1) for s in entries.values()) * 4 / 1e9))

    # ---- golden checksums --------------------------------------------------
    if args.golden:
        files = ["manifest.edn", "tokenizer.edn", "config.edn"] + sorted(fn for _, fn in entries.values())
        with open(args.golden, "w") as g:
            g.write("{:files {\n")
            for fn in files:
                with open(os.path.join(out, fn), "rb") as f:
                    data = f.read()
                g.write("  %s {:size %d :crc32 %d}\n" % (edn_str(fn), len(data), zlib.crc32(data) & 0xFFFFFFFF))
            g.write("}}\n")
        print("golden checksums:", args.golden, len(files), "files")


if __name__ == "__main__":
    main()
