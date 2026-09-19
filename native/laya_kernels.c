/* laya_kernels.c — elementwise/reduction kernels for the laya inference
 * engine. Matmuls go to cblas_sgemm (Accelerate / OpenBLAS); everything
 * torch does outside a gemm lives here. Layout convention: row-major,
 * tensors are [rows x cols] contiguous, masks are uint8 (1 = allowed).
 *
 * Tolerances: every kernel is tested against the torch CPU-float32 oracle
 * (golden/ sidecars dumped from transformers/torch itself). f32 throughout.
 */
/* fseeko/off_t are POSIX, not C11: glibc hides them under -std=c11 unless
 * asked (macOS exposes them regardless). Must precede every include. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* forward decls: lla_attention calls lla_scores/lla_masked_softmax, which
 * are defined later in the file */
void lla_scores(const float *q, const float *k, int64_t n, int64_t m,
                int64_t d, double scale, float *out);
void lla_masked_softmax(const float *scores, const uint8_t *allowed,
                        int64_t L, float *out);

/* ---------------- gather + layernorm ---------------- */

/* token embedding gather: rows of emb selected by ids -> out [n x d].
 * torch: nn.Embedding(input_ids) — exactly a row gather. */
void lla_gather_rows(const float *emb, const int64_t *ids, int64_t n, int64_t d,
                     float *out) {
    for (int64_t i = 0; i < n; i++) {
        memcpy(out + i * d, emb + ids[i] * d, sizeof(float) * d);
    }
}

/* LayerNorm over the last dim. weight-only when bias==NULL (ModernBERT's
 * norm_bias=false); with bias for the torch head (norm1/norm2 have bias).
 * torch computes mean/var in f32 with eps inside the sqrt. */
void lla_layernorm(const float *x, const float *w, const float *b, int64_t n,
                   int64_t d, float eps, float *out) {
    for (int64_t i = 0; i < n; i++) {
        const float *xi = x + i * d;
        float *oi = out + i * d;
        double sum = 0.0;
        for (int64_t j = 0; j < d; j++) sum += xi[j];
        double mean = sum / d;
        double var = 0.0;
        for (int64_t j = 0; j < d; j++) {
            double t = xi[j] - mean;
            var += t * t;
        }
        var /= d;
        double inv = 1.0 / sqrt(var + eps);
        if (b) {
            for (int64_t j = 0; j < d; j++)
                oi[j] = (float)((xi[j] - mean) * inv * w[j] + b[j]);
        } else {
            for (int64_t j = 0; j < d; j++)
                oi[j] = (float)((xi[j] - mean) * inv * w[j]);
        }
    }
}

/* ---------------- activations ---------------- */

/* erf GELU, exactly torch's nn.functional.gelu default:
 * 0.5 * x * (1 + erf(x / sqrt(2))). */
void lla_gelu(const float *x, int64_t n, float *out) {
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    }
}

void lla_relu(const float *x, int64_t n, float *out) {
    for (int64_t i = 0; i < n; i++) out[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

void lla_silu(const float *x, int64_t n, float *out) {
    for (int64_t i = 0; i < n; i++) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

/* SwiGLU with erf-gelu on the gate input, torch ModernBertMLP:
 * out = act(Wi(x)[0:mid]) * Wi(x)[mid:2*mid]. in = [n x 2*mid] -> out [n x mid]. */
void lla_swiglu(const float *x, int64_t n, int64_t mid, float *out) {
    for (int64_t i = 0; i < n; i++) {
        const float *xi = x + i * 2 * mid;
        float *oi = out + i * mid;
        for (int64_t j = 0; j < mid; j++) {
            float a = xi[j];
            float g = xi[mid + j];
            float act = 0.5f * a * (1.0f + erff(a * 0.70710678118654752440f));
            oi[j] = act * g;
        }
    }
}

/* ---------------- rope ---------------- */

/* cos/sin tables for one theta. torch computes inv_freq and freqs in f32:
 * inv_freq = 1/(base**(arange(0,d,2)/d)) as f32, freqs = inv_freq*pos f32,
 * emb = cat(freqs,freqs). We mirror the precision points so the tables
 * match to ~1 ulp (golden rope sidecars pin this). */
void lla_rope_tables(double theta, int64_t d, int64_t len, float *cos_t,
                     float *sin_t) {
    int64_t half = d / 2;
    float *inv = malloc(sizeof(float) * half);
    for (int64_t i = 0; i < half; i++)
        inv[i] = (float)(1.0 / pow(theta, (double)(2 * i) / (double)d));
    for (int64_t p = 0; p < len; p++) {
        float pf = (float)p;
        for (int64_t i = 0; i < half; i++) {
            float f = inv[i] * pf;
            float c = cosf(f), s = sinf(f);
            cos_t[p * d + i] = c;
            cos_t[p * d + half + i] = c;
            sin_t[p * d + i] = s;
            sin_t[p * d + half + i] = s;
        }
    }
    free(inv);
}

/* apply rope in HEAD-MAJOR layout: rows ordered (h, t) -> row (h*L+t)*d.
 * x' = x*cos + rotate_half(x)*sin; x1 = x[0:half], x2 = x[half:d]. */
void lla_rope_apply(float *qk, const float *cos_t, const float *sin_t,
                    int64_t n_heads, int64_t L, int64_t d) {
    int64_t half = d / 2;
    for (int64_t h = 0; h < n_heads; h++) {
        for (int64_t t = 0; t < L; t++) {
            float *x = qk + (h * L + t) * d;
            const float *c = cos_t + t * d;
            const float *s = sin_t + t * d;
            for (int64_t i = 0; i < half; i++) {
                float x1 = x[i], x2 = x[half + i];
                x[i] = x1 * c[i] - x2 * s[i];
                x[half + i] = x2 * c[half + i] + x1 * s[i];
            }
        }
    }
}

/* token-major [L x (H*hd)] -> head-major [H x L x hd] */
void lla_split_heads(const float *x, int64_t L, int64_t H, int64_t hd,
                     float *dst) {
    for (int64_t h = 0; h < H; h++)
        for (int64_t t = 0; t < L; t++)
            memcpy(dst + (h * L + t) * hd, x + (t * H + h) * hd,
                   sizeof(float) * hd);
}

/* qkv [L x 3d] (Wqkv output, per-token [q|k|v]) -> head-major q,k,v
 * [H*L x hd each]. torch: view(L,3,H,hd).unbind(-3) then transpose(1,2). */
void lla_split_qkv(const float *qkv, int64_t L, int64_t H, int64_t hd,
                   float *qh, float *kh, float *vh) {
    int64_t d = H * hd;
    for (int64_t t = 0; t < L; t++) {
        for (int64_t h = 0; h < H; h++) {
            memcpy(qh + (h * L + t) * hd, qkv + t * 3 * d + h * hd,
                   sizeof(float) * hd);
            memcpy(kh + (h * L + t) * hd, qkv + t * 3 * d + d + h * hd,
                   sizeof(float) * hd);
            memcpy(vh + (h * L + t) * hd, qkv + t * 3 * d + 2 * d + h * hd,
                   sizeof(float) * hd);
        }
    }
}

/* full per-layer attention: head-major q,k,v [H*L x hd] + allowed [L x L]
 * -> token-major out [L x (H*hd)]. scores/softmax/weighted-sum per head,
 * writing straight to the token-major slot. */
void lla_attention(const float *qh, const float *kh, const float *vh,
                   const uint8_t *allowed, int64_t H, int64_t L, int64_t hd,
                   double scale, float *out) {
    float *S = malloc(sizeof(float) * L * L);
    float *P = malloc(sizeof(float) * L * L);
    for (int64_t h = 0; h < H; h++) {
        const float *q = qh + h * L * hd;
        const float *k = kh + h * L * hd;
        const float *v = vh + h * L * hd;
        lla_scores(q, k, L, L, hd, scale, S);
        lla_masked_softmax(S, allowed, L, P);
        for (int64_t i = 0; i < L; i++) {
            float *oi = out + (i * H + h) * hd;
            memset(oi, 0, sizeof(float) * hd);
            for (int64_t j = 0; j < L; j++) {
                float w = P[i * L + j];
                if (w == 0.0f) continue;
                const float *vj = v + j * hd;
                for (int64_t x = 0; x < hd; x++) oi[x] += w * vj[x];
            }
        }
    }
    free(S);
    free(P);
}

/* head-major [H x L x hd] -> token-major [L x (H*hd)] */
void lla_merge_heads(const float *src, int64_t H, int64_t L, int64_t hd,
                     float *x) {
    for (int64_t h = 0; h < H; h++) {
        for (int64_t t = 0; t < L; t++)
            memcpy(x + (t * H + h) * hd, src + (h * L + t) * hd,
                   sizeof(float) * hd);
    }
}

/* ---------------- masks ---------------- */

/* Allowed-attention matrix, uint8: 1 = attend. full mode (window < 0):
 * allowed(i,j) = att[b] row j present (padding mask only).
 * sliding: additionally |i - j| <= window.
 * att: [B x L] 1=present. out: [L x L] for batch row b. */
void lla_allowed_mask(const uint8_t *att, int64_t b, int64_t L, int64_t window,
                      uint8_t *out) {
    for (int64_t i = 0; i < L; i++) {
        for (int64_t j = 0; j < L; j++) {
            int ok = att[b * L + j];
            if (ok && window >= 0) ok = llabs(i - j) <= window;
            out[i * L + j] = (uint8_t)ok;
        }
    }
}

/* ---------------- attention scores ---------------- */

/* softmax over allowed keys per query row, torch-style with max subtraction.
 * scores: [L x L] f32 (already scaled). Rows with zero allowed keys are
 * left at zero (they are all-padding rows the caller discards). */
void lla_masked_softmax(const float *scores, const uint8_t *allowed, int64_t L,
                        float *out) {
    for (int64_t i = 0; i < L; i++) {
        const float *si = scores + i * L;
        const uint8_t *ai = allowed + i * L;
        float *oi = out + i * L;
        float m = -INFINITY;
        for (int64_t j = 0; j < L; j++)
            if (ai[j] && si[j] > m) m = si[j];
        if (m == -INFINITY) {
            for (int64_t j = 0; j < L; j++) oi[j] = 0.0f;
            continue;
        }
        float z = 0.0f;
        for (int64_t j = 0; j < L; j++) {
            if (ai[j]) {
                oi[j] = expf(si[j] - m);
                z += oi[j];
            } else {
                oi[j] = 0.0f;
            }
        }
        float inv = 1.0f / z;
        for (int64_t j = 0; j < L; j++) oi[j] *= inv;
    }
}

/* qk^T for one head block: A [n x d], B [m x d] -> S [n x m], scaled.
 * Kept here because a deinterleaved sgemm per head would need d==k packing;
 * dot over d is a single pass and stays cache-hot. */
void lla_scores(const float *q, const float *k, int64_t n, int64_t m,
                int64_t d, double scale, float *out) {
    for (int64_t i = 0; i < n; i++) {
        const float *qi = q + i * d;
        float *oi = out + i * m;
        for (int64_t j = 0; j < m; j++) {
            const float *kj = k + j * d;
            double s = 0.0;
            for (int64_t t = 0; t < d; t++) s += (double)qi[t] * kj[t];
            oi[j] = (float)(s * scale);
        }
    }
}

/* weighted value sum P [n x m] @ V [m x d] -> O [n x d] */
void lla_weighted_sum(const float *p, const float *v, int64_t n, int64_t m,
                      int64_t d, float *out) {
    for (int64_t i = 0; i < n; i++) {
        const float *pi = p + i * m;
        float *oi = out + i * d;
        memset(oi, 0, sizeof(float) * d);
        for (int64_t j = 0; j < m; j++) {
            float w = pi[j];
            if (w == 0.0f) continue;
            const float *vj = v + j * d;
            for (int64_t t = 0; t < d; t++) oi[t] += w * vj[t];
        }
    }
}

/* ---------------- misc ---------------- */

/* out = a + b*alpha rowwise: [n x d]. Used for residual adds and type_emb. */
void lla_add_scaled(const float *a, const float *b, int64_t n, int64_t d,
                    float alpha, float *out) {
    int64_t total = n * d;
    for (int64_t i = 0; i < total; i++) out[i] = a[i] + alpha * b[i];
}

/* add a bias vector to every row: x[i,j] += bias[j], [n x k]. torch Linear. */
void lla_add_bias(float *x, const float *bias, int64_t n, int64_t k) {
    for (int64_t i = 0; i < n; i++) {
        float *xi = x + i * k;
        for (int64_t j = 0; j < k; j++) xi[j] += bias[j];
    }
}

/* per-batch broadcast add of a bias row: x[r,:] += bias[q[r],:] where q is
 * [n] qtype ids and bias is [3 x d] (type_emb add). */
void lla_add_qtype_bias(const float *x, const float *bias, const int64_t *q,
                        int64_t n, int64_t d, float *out) {
    for (int64_t i = 0; i < n; i++) {
        const float *bi = bias + q[i] * d;
        const float *xi = x + i * d;
        float *oi = out + i * d;
        for (int64_t j = 0; j < d; j++) oi[j] = xi[j] + bi[j];
    }
}

/* softmax over k logits per row (float32, max-subtracted) */
void lla_softmax(const float *x, int64_t n, int64_t k, float *out) {
    for (int64_t i = 0; i < n; i++) {
        const float *xi = x + i * k;
        float *oi = out + i * k;
        float m = xi[0];
        for (int64_t j = 1; j < k; j++)
            if (xi[j] > m) m = xi[j];
        float z = 0.0f;
        for (int64_t j = 0; j < k; j++) {
            oi[j] = expf(xi[j] - m);
            z += oi[j];
        }
        float inv = 1.0f / z;
        for (int64_t j = 0; j < k; j++) oi[j] *= inv;
    }
}

/* numeric stability check helper for tests: max |a-b| */
float lla_max_abs_diff(const float *a, const float *b, int64_t n) {
    float m = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

/* ---------------- prepare: checkpoint conversion ---------------- */
/* Used by laya.prepare (jolt prepare): model.safetensors is F16 (one F32
 * buffer), stored little-endian. Widening F16->F32 is exact; the byte
 * order is handled explicitly so the output is little-endian on any host. */
#include <stdio.h>

/* f16 bits -> f32 bits. Same as numpy's npy_halfbits_to_floatbits: exact
 * for every finite value, subnormals renormalized, inf kept, NaN payloads
 * shifted up by 13 bits so the blobs match a numpy astype(float32). */
static uint32_t lla_h2f_bits(uint32_t h) {
    uint32_t sgn = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t sig = h & 0x3ffu;
    if (exp == 0) {
        if (sig == 0) return sgn;
        int e = 0;
        do { sig <<= 1; e++; } while ((sig & 0x400u) == 0);
        return sgn | ((uint32_t)(127 - 15 - e + 1) << 23) | ((sig & 0x3ffu) << 13);
    }
    if (exp == 0x1fu) return sgn | 0x7f800000u | (sig << 13);
    return sgn | ((exp + 112u) << 23) | (sig << 13);
}

/* Widen `count` f16 values found at byte `offset` of `in` into raw f32 at
 * `out`. Returns the number of values written, or -1 on any I/O error. */
int64_t lla_f16_file_to_f32(const char *in, int64_t offset, int64_t count,
                            const char *out) {
    FILE *fi = fopen(in, "rb");
    if (!fi) return -1;
    FILE *fo = fopen(out, "wb");
    if (!fo) { fclose(fi); return -1; }
    if (fseeko(fi, (off_t)offset, SEEK_SET) != 0) { fclose(fi); fclose(fo); return -1; }
    enum { CH = 1 << 16 };
    uint8_t *ib = malloc(CH * 2);
    uint8_t *ob = malloc(CH * 4);
    int64_t done = 0;
    while (done < count) {
        size_t want = (size_t)((count - done) < CH ? (count - done) : CH);
        if (fread(ib, 2, want, fi) != want) { done = -1; break; }
        for (size_t i = 0; i < want; i++) {
            uint32_t h = (uint32_t)ib[2 * i] | ((uint32_t)ib[2 * i + 1] << 8);
            uint32_t f = lla_h2f_bits(h);
            ob[4 * i] = (uint8_t)f; ob[4 * i + 1] = (uint8_t)(f >> 8);
            ob[4 * i + 2] = (uint8_t)(f >> 16); ob[4 * i + 3] = (uint8_t)(f >> 24);
        }
        if (fwrite(ob, 4, want, fo) != want) { done = -1; break; }
        done += (int64_t)want;
    }
    free(ib); free(ob);
    fclose(fi);
    if (fclose(fo) != 0) return -1;
    return done;
}

/* Copy `nbytes` at `offset` of `in` to `out` verbatim (F32 tensors).
 * Returns bytes copied, or -1. */
int64_t lla_copy_file_range(const char *in, int64_t offset, int64_t nbytes,
                            const char *out) {
    FILE *fi = fopen(in, "rb");
    if (!fi) return -1;
    FILE *fo = fopen(out, "wb");
    if (!fo) { fclose(fi); return -1; }
    if (fseeko(fi, (off_t)offset, SEEK_SET) != 0) { fclose(fi); fclose(fo); return -1; }
    enum { CH = 1 << 18 };
    uint8_t *buf = malloc(CH);
    int64_t done = 0;
    while (done < nbytes) {
        size_t want = (size_t)((nbytes - done) < CH ? (nbytes - done) : CH);
        if (fread(buf, 1, want, fi) != want || fwrite(buf, 1, want, fo) != want) { done = -1; break; }
        done += (int64_t)want;
    }
    free(buf);
    fclose(fi);
    if (fclose(fo) != 0) return -1;
    return done;
}

/* Read `n` bytes at `offset` of `path` into dst. Returns bytes read or -1.
 * (the safetensors header: 8-byte LE length + JSON) */
int64_t lla_read_file_range(const char *path, int64_t offset, int64_t n, uint8_t *dst) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseeko(f, (off_t)offset, SEEK_SET) != 0) { fclose(f); return -1; }
    size_t got = fread(dst, 1, (size_t)n, f);
    fclose(f);
    return (int64_t)got;
}

/* zlib-compatible CRC-32 of a whole file (poly 0xEDB88320); *size_out gets
 * the byte length. Returns the CRC, or -1 if the file cannot be read. The
 * prepare parity test compares against zlib.crc32 values pinned in
 * golden/prepare.edn. */
int64_t lla_file_crc32(const char *path, int64_t *size_out) {
    static uint32_t table[256];
    static int init = 0;
    if (!init) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = 1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    enum { CH = 1 << 18 };
    uint8_t *buf = malloc(CH);
    uint32_t crc = 0xFFFFFFFFu;
    int64_t size = 0;
    size_t got;
    while ((got = fread(buf, 1, CH, f)) > 0) {
        for (size_t i = 0; i < got; i++) crc = table[(crc ^ buf[i]) & 0xffu] ^ (crc >> 8);
        size += (int64_t)got;
    }
    free(buf);
    fclose(f);
    if (size_out) *size_out = size;
    return (int64_t)(crc ^ 0xFFFFFFFFu);
}
