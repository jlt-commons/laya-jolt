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

/* ---------------- attention softmax (the two gemms are cblas, see laya.tensors/attention) ---------------- */

/* e^x for x <= 0 (a max-subtracted score), written so the loop vectorizes:
 * no libm call, no branch, rounding by the 1.5*2^23 trick rather than rintf
 * (which is a libcall on baseline x86-64). Cody-Waite reduction
 * x = n ln2 + r with |r| <= ln2/2, Cephes' degree-6 polynomial for e^r, and
 * 2^n through the exponent field: about 1 ulp against libm expf, checked
 * by tensors_test. Below -87.3 the f32 result is denormal or zero either
 * way, so the argument is clamped there; above 0 it is clamped to 0, which
 * only masked keys reach (they are multiplied by 0 afterwards). */
static inline float exp_neg(float x) {
    x = x < -87.3f ? -87.3f : x;
    x = x > 0.0f ? 0.0f : x;
    float n = x * 1.44269504088896341f + 12582912.0f;
    n -= 12582912.0f;                              /* nearest integer, as a float */
    float r = x - n * 0.693145751953125f;          /* ln2 split in two: hi is exact in f32 */
    r -= n * 1.428606765330187e-06f;
    float p = 1.9875691500E-4f;
    p = p * r + 1.3981999507E-3f;
    p = p * r + 8.3334519073E-3f;
    p = p * r + 4.1665795894E-2f;
    p = p * r + 1.6666665459E-1f;
    p = p * r + 5.0000001201E-1f;
    p = p * r * r + r + 1.0f;
    union { int32_t i; float f; } two_n;
    two_n.i = ((int32_t)n + 127) << 23;
    return p * two_n.f;
}

#ifdef __clang__
#define LLA_INTERLEAVE4 _Pragma("clang loop interleave_count(4)")
#else
#define LLA_INTERLEAVE4
#endif

/* softmax over the allowed keys of each query row, torch-style with max
 * subtraction, on a [rows x cols] block. S and P are the block itself
 * (leading dimension cols); `allowed` points at the block's top-left entry
 * of the [L x L] mask with leading dimension lda. Rows with no allowed key
 * are left at zero (padding rows the caller discards).
 *
 * Four passes so each is a straight loop the compiler vectorizes. The max
 * and the sum are f32 reductions, which -O2 will not reassociate on its
 * own, so both are spelled out in eight lanes. Measured at 16 x 512 x 512
 * on an M-series core: 5.5 ms, of which the exp pass is half; interleaving
 * it 4x is worth a third of that pass. */
void lla_masked_softmax(const float *S, const uint8_t *allowed, int64_t lda,
                        int64_t rows, int64_t cols, float *P) {
    for (int64_t i = 0; i < rows; i++) {
        const float *si = S + i * cols;
        const uint8_t *ai = allowed + i * lda;
        float *oi = P + i * cols;
        float mx[8] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY,
                       -INFINITY, -INFINITY, -INFINITY, -INFINITY};
        int64_t j = 0;
        for (; j + 8 <= cols; j += 8)
            for (int k = 0; k < 8; k++) {
                float v = ai[j + k] ? si[j + k] : -INFINITY;
                mx[k] = v > mx[k] ? v : mx[k];
            }
        for (; j < cols; j++) {
            float v = ai[j] ? si[j] : -INFINITY;
            mx[0] = v > mx[0] ? v : mx[0];
        }
        float m = mx[0];
        for (int k = 1; k < 8; k++) m = mx[k] > m ? mx[k] : m;
        if (m == -INFINITY) {
            for (int64_t j2 = 0; j2 < cols; j2++) oi[j2] = 0.0f;
            continue;
        }
        LLA_INTERLEAVE4
        for (int64_t j2 = 0; j2 < cols; j2++)
            oi[j2] = (float)ai[j2] * exp_neg(si[j2] - m);
        float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        j = 0;
        for (; j + 8 <= cols; j += 8)
            for (int k = 0; k < 8; k++) lane[k] += oi[j + k];
        for (; j < cols; j++) lane[0] += oi[j];
        float z = ((lane[0] + lane[1]) + (lane[2] + lane[3])) +
                  ((lane[4] + lane[5]) + (lane[6] + lane[7]));
        float inv = 1.0f / z;
        for (int64_t j2 = 0; j2 < cols; j2++) oi[j2] *= inv;
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
