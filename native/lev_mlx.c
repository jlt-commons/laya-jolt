/* lev_mlx.c — the encoder forward on MLX (Apple silicon GPU), a flat C
 * face on mlx-c for jolt.ffi.
 *
 * The same graph lev.model runs on the C kernels, built lazily with
 * mlx-c ops and evaluated once per batch: ModernBERT embeddings -> N
 * layers (RoPE'd attention with the padding / sliding-window mask, gated
 * GELU MLP) -> final norm -> type embedding -> the two-layer decision
 * head -> the scorer at the [MASK] markers and the act head on the CLS
 * row. What crosses the boundary is what lev.model/forward-batch
 * answers: one logit per marker and the two act logits per row, f32.
 *
 * Weights come from lev's own prepared data/ (raw little-endian f32
 * files, one per tensor, named by the manifest); the caller feeds them
 * one at a time with lev_mlx_load_tensor, which parses the name into its
 * slot. They are held on the device in the handle's dtype: f32 (what the
 * golden traces are compared against) or f16 (half the memory, faster,
 * argmax-exact but not value-exact).
 *
 * One handle is one model on one stream; calls on it are not thread-safe
 * (the server serializes inference on a lock).
 *
 * Built by `jolt mlx` (native/build_mlx.sh) with mlx-c and mlx static
 * into liblev_mlx.dylib for jolt run/test and liblev_mlx.a for jolt
 * build; MLX loads its Metal kernels (mlx.metallib) from next to the
 * binary holding it, so the build puts a copy beside the dylib.
 */
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlx/c/mlx.h"

#define LEV_MAX_LAYERS 64
#define LEV_MAX_HEAD_LAYERS 4
#define LEV_MAX_TMP 8192

struct enc_layer {
    mlx_array attn_norm, wqkv, wo, mlp_norm, wi, wo_mlp;
};

struct head_layer {
    mlx_array in_w, in_b, out_w, out_b, n1w, n1b, n2w, n2b, l1w, l1b, l2w, l2b;
};

struct lev_mlx {
    mlx_stream s;
    mlx_dtype dtype;
    /* config */
    int hidden, layers, heads, head_dim, inter, window, head_layers, vocab;
    float rope_full, rope_local, norm_eps;
    int sliding[LEV_MAX_LAYERS];
    /* weights */
    mlx_array tok_emb, emb_norm, final_norm, type_emb;
    struct enc_layer enc[LEV_MAX_LAYERS];
    struct head_layer head[LEV_MAX_HEAD_LAYERS];
    mlx_array sc0w, sc0b, sc1w, sc1b, sc3w, sc3b;
    mlx_array act0w, act0b, act2w, act2b;
    int loaded;
    char error[1024];
};

/* --- errors ------------------------------------------------------------------ */

/* The first mlx error since an entry point cleared it: a failure inside
 * MLX (the Metal library missing, say) empties the array the op was to
 * answer, and every op after that fails with "expected a non-empty
 * mlx_array", which is not the message wanted. */
static char last_error[1024];

static void on_error(const char *msg, void *data) {
    (void)data;
    if (!last_error[0]) snprintf(last_error, sizeof last_error, "%s", msg);
}

static void clear_error(void) { last_error[0] = 0; }

static int handler_installed = 0;

static void install_handler(void) {
    if (!handler_installed) {
        mlx_set_error_handler(on_error, NULL, NULL);
        handler_installed = 1;
    }
}

static void set_error(struct lev_mlx *h, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(h->error, sizeof h->error, fmt, ap);
    va_end(ap);
}

/* --- the per-forward arena ----------------------------------------------------- */

/* Every intermediate array is registered here and freed at the end of the
 * forward; a failed op marks the arena and every later op is a no-op
 * answering an empty array, so the forward unwinds through one check. */
struct fwd {
    struct lev_mlx *h;
    mlx_array tmp[LEV_MAX_TMP];
    int n;
    int failed;
};

static mlx_array keep(struct fwd *f, mlx_array a) {
    if (f->n < LEV_MAX_TMP) f->tmp[f->n++] = a;
    else { mlx_array_free(a); f->failed = 1; set_error(f->h, "forward: too many intermediates"); }
    return a;
}

static void fail(struct fwd *f, const char *what) {
    if (!f->failed) {
        f->failed = 1;
        set_error(f->h, "%s: %s", what, last_error[0] ? last_error : "mlx error");
    }
}

static void release(struct fwd *f) {
    for (int i = 0; i < f->n; i++) mlx_array_free(f->tmp[i]);
    f->n = 0;
}

#define S (f->h->s)

/* op wrappers: new + op + keep, failing softly */
#define OP1(name, call)                                                \
    static mlx_array name {                                            \
        mlx_array r = mlx_array_new();                                 \
        if (f->failed) return keep(f, r);                              \
        if (call) fail(f, #name);                                      \
        return keep(f, r);                                             \
    }

OP1(add(struct fwd *f, mlx_array a, mlx_array b), mlx_add(&r, a, b, S))
OP1(sub(struct fwd *f, mlx_array a, mlx_array b), mlx_subtract(&r, a, b, S))
OP1(mul(struct fwd *f, mlx_array a, mlx_array b), mlx_multiply(&r, a, b, S))
OP1(divide(struct fwd *f, mlx_array a, mlx_array b), mlx_divide(&r, a, b, S))
OP1(maximum(struct fwd *f, mlx_array a, mlx_array b), mlx_maximum(&r, a, b, S))
OP1(erf_(struct fwd *f, mlx_array a), mlx_erf(&r, a, S))
OP1(logf_(struct fwd *f, mlx_array a), mlx_log(&r, a, S))
OP1(transpose(struct fwd *f, mlx_array a), mlx_transpose(&r, a, S))
OP1(matmul(struct fwd *f, mlx_array a, mlx_array b), mlx_matmul(&r, a, b, S))
OP1(addmm(struct fwd *f, mlx_array c, mlx_array a, mlx_array b), mlx_addmm(&r, c, a, b, 1.0f, 1.0f, S))
OP1(astype(struct fwd *f, mlx_array a, mlx_dtype t), mlx_astype(&r, a, t, S))
OP1(take(struct fwd *f, mlx_array a, mlx_array idx, int axis), mlx_take_axis(&r, a, idx, axis, S))
OP1(reshape(struct fwd *f, mlx_array a, const int *shape, int n), mlx_reshape(&r, a, shape, n, S))
OP1(transpose_axes(struct fwd *f, mlx_array a, const int *axes, int n), mlx_transpose_axes(&r, a, axes, n, S))
OP1(expand_dims(struct fwd *f, mlx_array a, int axis), mlx_expand_dims(&r, a, axis, S))
OP1(squeeze_axis(struct fwd *f, mlx_array a, int axis), mlx_squeeze_axis(&r, a, axis, S))
OP1(layer_norm(struct fwd *f, mlx_array x, mlx_array w, mlx_array b, float eps), mlx_fast_layer_norm(&r, x, w, b, eps, S))
OP1(where(struct fwd *f, mlx_array c, mlx_array a, mlx_array b), mlx_where(&r, c, a, b, S))
OP1(softmax(struct fwd *f, mlx_array a, int axis), mlx_softmax_axis(&r, a, axis, true, S))
OP1(sum_axis(struct fwd *f, mlx_array a, int axis), mlx_sum_axis(&r, a, axis, false, S))
OP1(sort_axis(struct fwd *f, mlx_array a, int axis), mlx_sort_axis(&r, a, axis, S))
OP1(slice(struct fwd *f, mlx_array a, const int *start, const int *stop, const int *strides, int n), mlx_slice(&r, a, start, n, stop, n, strides, n, S))
OP1(abs_(struct fwd *f, mlx_array a), mlx_abs(&r, a, S))
OP1(less_equal(struct fwd *f, mlx_array a, mlx_array b), mlx_less_equal(&r, a, b, S))
OP1(logical_and(struct fwd *f, mlx_array a, mlx_array b), mlx_logical_and(&r, a, b, S))
OP1(logical_or(struct fwd *f, mlx_array a, mlx_array b), mlx_logical_or(&r, a, b, S))
OP1(logical_not(struct fwd *f, mlx_array a), mlx_logical_not(&r, a, S))
OP1(arange(struct fwd *f, int n), mlx_arange(&r, 0, n, 1, MLX_INT32, S))
OP1(rope(struct fwd *f, mlx_array x, int dims, float base),
    mlx_fast_rope(&r, x, dims, false, (mlx_optional_float){base, true}, 1.0f, 0, (mlx_array){0}, S))
OP1(sdpa(struct fwd *f, mlx_array q, mlx_array k, mlx_array v, float scale, mlx_array mask),
    mlx_fast_scaled_dot_product_attention(&r, q, k, v, scale, "", mask, (mlx_array){0}, S))
OP1(concat(struct fwd *f, mlx_vector_array v, int axis), mlx_concatenate_axis(&r, v, axis, S))
OP1(stack(struct fwd *f, mlx_vector_array v, int axis), mlx_stack_axis(&r, v, axis, S))

static mlx_array scalar(struct fwd *f, float v, mlx_dtype t) {
    /* a typed scalar: mlx promotes f16 * f32-scalar to f32, so the
     * constants take the activations' dtype */
    mlx_array s = mlx_array_new_float(v);
    keep(f, s);
    return t == MLX_FLOAT32 ? s : astype(f, s, t);
}

static mlx_array data(struct fwd *f, const void *p, const int *shape, int n, mlx_dtype t) {
    return keep(f, mlx_array_new_data(p, shape, n, t));
}

/* x [.., 2m] -> gelu(value) * gate over the halves; gelu exact (erf), as
 * lev_kernels' swiglu and nn.gelu */
static mlx_array gelu(struct fwd *f, mlx_array x, mlx_dtype t) {
    mlx_array half = scalar(f, 0.5f, t);
    mlx_array one = scalar(f, 1.0f, t);
    mlx_array inv_sqrt2 = scalar(f, 0.70710678118654752440f, t);
    return mul(f, mul(f, x, half), add(f, one, erf_(f, mul(f, x, inv_sqrt2))));
}

/* x @ W^T (+ b): torch Linear layout, W [out x in] */
static mlx_array linear(struct fwd *f, mlx_array x, mlx_array w, mlx_array b) {
    mlx_array wt = transpose(f, w);
    return b.ctx ? addmm(f, b, x, wt) : matmul(f, x, wt);
}

/* [B, L, 3d] -> q, k, v each [B, H, L, hd] */
static void split_heads(struct fwd *f, mlx_array qkv, int B, int L, int H, int hd,
                        mlx_array *q, mlx_array *k, mlx_array *v) {
    int shape[5] = {B, L, 3, H, hd};
    mlx_array r = reshape(f, qkv, shape, 5);
    int perm[4] = {0, 2, 1, 3};
    mlx_array *out[3] = {q, k, v};
    for (int i = 0; i < 3; i++) {
        int start[5] = {0, 0, i, 0, 0}, stop[5] = {B, L, i + 1, H, hd}, strides[5] = {1, 1, 1, 1, 1};
        mlx_array part = squeeze_axis(f, slice(f, r, start, stop, strides, 5), 2); /* [B, L, H, hd] */
        *out[i] = transpose_axes(f, part, perm, 4);
    }
}

/* --- the graph ---------------------------------------------------------------- */

static mlx_array encoder_layer(struct fwd *f, int i, mlx_array x, mlx_array mask, int B, int L) {
    struct lev_mlx *h = f->h;
    struct enc_layer *w = &h->enc[i];
    mlx_array none = {0};
    mlx_array a = i == 0 ? x : layer_norm(f, x, w->attn_norm, none, h->norm_eps);
    mlx_array q, k, v;
    split_heads(f, linear(f, a, w->wqkv, none), B, L, h->heads, h->head_dim, &q, &k, &v);
    float base = h->sliding[i] ? h->rope_local : h->rope_full;
    q = rope(f, q, h->head_dim, base);
    k = rope(f, k, h->head_dim, base);
    mlx_array o = sdpa(f, q, k, v, 1.0f / sqrtf((float)h->head_dim), mask);
    int perm[4] = {0, 2, 1, 3}, flat[3] = {B, L, h->hidden};
    o = reshape(f, transpose_axes(f, o, perm, 4), flat, 3);
    x = add(f, x, linear(f, o, w->wo, none));
    mlx_array m = layer_norm(f, x, w->mlp_norm, none, h->norm_eps);
    mlx_array wi = linear(f, m, w->wi, none); /* [B, L, 2*inter] */
    int s0[3] = {0, 0, 0}, s1[3] = {B, L, h->inter}, s2[3] = {0, 0, h->inter}, s3[3] = {B, L, 2 * h->inter}, st[3] = {1, 1, 1};
    mlx_array value = slice(f, wi, s0, s1, st, 3);
    mlx_array gate = slice(f, wi, s2, s3, st, 3);
    mlx_array act = mul(f, gelu(f, value, h->dtype), gate);
    return add(f, x, linear(f, act, w->wo_mlp, none));
}

static mlx_array head_layer(struct fwd *f, int li, mlx_array x, mlx_array mask, int B, int L) {
    struct lev_mlx *h = f->h;
    struct head_layer *w = &h->head[li];
    int H = h->hidden / 64 > 0 ? h->hidden / 64 : 1, hd = h->hidden / H;
    mlx_array n1 = layer_norm(f, x, w->n1w, w->n1b, 1e-5f);
    mlx_array q, k, v;
    split_heads(f, linear(f, n1, w->in_w, w->in_b), B, L, H, hd, &q, &k, &v);
    mlx_array o = sdpa(f, q, k, v, 1.0f / sqrtf((float)hd), mask);
    int perm[4] = {0, 2, 1, 3}, flat[3] = {B, L, h->hidden};
    o = reshape(f, transpose_axes(f, o, perm, 4), flat, 3);
    x = add(f, x, linear(f, o, w->out_w, w->out_b));
    mlx_array n2 = layer_norm(f, x, w->n2w, w->n2b, 1e-5f);
    mlx_array l1 = maximum(f, linear(f, n2, w->l1w, w->l1b), scalar(f, 0.0f, h->dtype)); /* ReLU */
    return add(f, x, linear(f, l1, w->l2w, w->l2b));
}

/* --- the face ------------------------------------------------------------------ */

struct lev_mlx *lev_mlx_new(int hidden, int layers, int heads, int head_dim, int inter, int window,
                            int head_layers, int vocab, float rope_full, float rope_local,
                            float norm_eps, const int32_t *sliding, int half) {
    install_handler();
    clear_error();
    struct lev_mlx *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    if (layers > LEV_MAX_LAYERS || head_layers > LEV_MAX_HEAD_LAYERS) {
        set_error(h, "unsupported depth: %d encoder layers, %d head layers", layers, head_layers);
        return h;
    }
    h->hidden = hidden; h->layers = layers; h->heads = heads; h->head_dim = head_dim;
    h->inter = inter; h->window = window; h->head_layers = head_layers; h->vocab = vocab;
    h->rope_full = rope_full; h->rope_local = rope_local; h->norm_eps = norm_eps;
    for (int i = 0; i < layers; i++) h->sliding[i] = sliding[i];
    h->dtype = half ? MLX_FLOAT16 : MLX_FLOAT32;
    bool gpu = false;
    mlx_metal_is_available(&gpu);
    h->s = gpu ? mlx_default_gpu_stream_new() : mlx_default_cpu_stream_new();
    /* touching the device is what loads mlx.metallib: a probe here turns a
     * missing library into this handle's error instead of a later one */
    mlx_array probe = mlx_array_new_float(1.0f), probe2 = mlx_array_new();
    if (!h->s.ctx || mlx_add(&probe2, probe, probe, h->s) || mlx_array_eval(probe2))
        set_error(h, "cannot open the mlx device: %s", last_error[0] ? last_error : "no stream");
    mlx_array_free(probe);
    mlx_array_free(probe2);
    return h;
}

int lev_mlx_ok(struct lev_mlx *h) { return h && h->error[0] == 0; }
const char *lev_mlx_error(struct lev_mlx *h) { return h ? h->error : "out of memory"; }
int lev_mlx_gpu(struct lev_mlx *h) {
    mlx_device d = mlx_device_new();
    mlx_device_type t = MLX_CPU;
    if (h && mlx_stream_get_device(&d, h->s) == 0) mlx_device_get_type(&t, d);
    mlx_device_free(d);
    return t == MLX_GPU;
}
int lev_mlx_half(struct lev_mlx *h) { return h && h->dtype == MLX_FLOAT16; }

const char *lev_mlx_version(void) {
#ifdef LEV_MLX_BUILD
    return LEV_MLX_BUILD;
#else
    return "unknown";
#endif
}

/* The slot a tensor name fills, or NULL for a name this model has no use
 * for (the checkpoint's temperature buffer). */
static mlx_array *slot_for(struct lev_mlx *h, const char *name) {
    int i;
    char rest[128];
    if (sscanf(name, "encoder.layers.%d.%127s", &i, rest) == 2 && i >= 0 && i < h->layers) {
        struct enc_layer *w = &h->enc[i];
        if (!strcmp(rest, "attn_norm.weight")) return &w->attn_norm;
        if (!strcmp(rest, "attn.Wqkv.weight")) return &w->wqkv;
        if (!strcmp(rest, "attn.Wo.weight")) return &w->wo;
        if (!strcmp(rest, "mlp_norm.weight")) return &w->mlp_norm;
        if (!strcmp(rest, "mlp.Wi.weight")) return &w->wi;
        if (!strcmp(rest, "mlp.Wo.weight")) return &w->wo_mlp;
        return NULL;
    }
    if (sscanf(name, "head.layers.%d.%127s", &i, rest) == 2 && i >= 0 && i < h->head_layers) {
        struct head_layer *w = &h->head[i];
        if (!strcmp(rest, "self_attn.in_proj_weight")) return &w->in_w;
        if (!strcmp(rest, "self_attn.in_proj_bias")) return &w->in_b;
        if (!strcmp(rest, "self_attn.out_proj.weight")) return &w->out_w;
        if (!strcmp(rest, "self_attn.out_proj.bias")) return &w->out_b;
        if (!strcmp(rest, "norm1.weight")) return &w->n1w;
        if (!strcmp(rest, "norm1.bias")) return &w->n1b;
        if (!strcmp(rest, "norm2.weight")) return &w->n2w;
        if (!strcmp(rest, "norm2.bias")) return &w->n2b;
        if (!strcmp(rest, "linear1.weight")) return &w->l1w;
        if (!strcmp(rest, "linear1.bias")) return &w->l1b;
        if (!strcmp(rest, "linear2.weight")) return &w->l2w;
        if (!strcmp(rest, "linear2.bias")) return &w->l2b;
        return NULL;
    }
    if (!strcmp(name, "encoder.embeddings.tok_embeddings.weight")) return &h->tok_emb;
    if (!strcmp(name, "encoder.embeddings.norm.weight")) return &h->emb_norm;
    if (!strcmp(name, "encoder.final_norm.weight")) return &h->final_norm;
    if (!strcmp(name, "type_emb.weight")) return &h->type_emb;
    if (!strcmp(name, "scorer.0.weight")) return &h->sc0w;
    if (!strcmp(name, "scorer.0.bias")) return &h->sc0b;
    if (!strcmp(name, "scorer.1.weight")) return &h->sc1w;
    if (!strcmp(name, "scorer.1.bias")) return &h->sc1b;
    if (!strcmp(name, "scorer.3.weight")) return &h->sc3w;
    if (!strcmp(name, "scorer.3.bias")) return &h->sc3b;
    if (!strcmp(name, "act_head.0.weight")) return &h->act0w;
    if (!strcmp(name, "act_head.0.bias")) return &h->act0b;
    if (!strcmp(name, "act_head.2.weight")) return &h->act2w;
    if (!strcmp(name, "act_head.2.bias")) return &h->act2b;
    return NULL;
}

/* Load one tensor: `path` holds rows*cols (cols 0: a vector of rows)
 * little-endian f32. Answers 1 when the name filled a slot, 0 when the
 * model has no slot for it (not an error), -1 on failure (see error). */
int lev_mlx_load_tensor(struct lev_mlx *h, const char *name, const char *path, int64_t rows, int64_t cols) {
    if (!h || h->error[0]) return -1;
    clear_error();
    mlx_array *slot = slot_for(h, name);
    if (!slot) return 0;
    int64_t n = cols > 0 ? rows * cols : rows;
    float *buf = malloc((size_t)n * sizeof *buf);
    if (!buf) { set_error(h, "%s: out of memory for %lld floats", name, (long long)n); return -1; }
    FILE *fp = fopen(path, "rb");
    if (!fp || fread(buf, sizeof *buf, (size_t)n, fp) != (size_t)n) {
        set_error(h, "%s: cannot read %lld floats from %s", name, (long long)n, path);
        if (fp) fclose(fp);
        free(buf);
        return -1;
    }
    fclose(fp);
    int shape[2] = {(int)rows, (int)cols};
    mlx_array a = mlx_array_new_data(buf, shape, cols > 0 ? 2 : 1, MLX_FLOAT32);
    free(buf);
    if (h->dtype != MLX_FLOAT32) {
        mlx_array c = mlx_array_new();
        if (mlx_astype(&c, a, h->dtype, h->s)) { set_error(h, "%s: %s", name, last_error); mlx_array_free(a); return -1; }
        mlx_array_free(a);
        a = c;
    }
    if (mlx_array_eval(a)) { set_error(h, "%s: %s", name, last_error); mlx_array_free(a); return -1; }
    if (slot->ctx) mlx_array_free(*slot);
    *slot = a;
    return 1;
}

/* Every slot filled? Answers the first missing name into `missing`. */
int lev_mlx_complete(struct lev_mlx *h, char *missing, int cap) {
    if (!h) return 0;
#define NEED(a, nm) if (!(a).ctx) { snprintf(missing, cap, "%s", nm); return 0; }
    NEED(h->tok_emb, "encoder.embeddings.tok_embeddings.weight");
    NEED(h->emb_norm, "encoder.embeddings.norm.weight");
    NEED(h->final_norm, "encoder.final_norm.weight");
    NEED(h->type_emb, "type_emb.weight");
    for (int i = 0; i < h->layers; i++) {
        struct enc_layer *w = &h->enc[i];
        if (i > 0) NEED(w->attn_norm, "encoder.layers.N.attn_norm.weight");
        NEED(w->wqkv, "encoder.layers.N.attn.Wqkv.weight");
        NEED(w->wo, "encoder.layers.N.attn.Wo.weight");
        NEED(w->mlp_norm, "encoder.layers.N.mlp_norm.weight");
        NEED(w->wi, "encoder.layers.N.mlp.Wi.weight");
        NEED(w->wo_mlp, "encoder.layers.N.mlp.Wo.weight");
    }
    for (int i = 0; i < h->head_layers; i++) {
        struct head_layer *w = &h->head[i];
        NEED(w->in_w, "head.layers.N.self_attn.in_proj_weight"); NEED(w->in_b, "head.layers.N.self_attn.in_proj_bias");
        NEED(w->out_w, "head.layers.N.self_attn.out_proj.weight"); NEED(w->out_b, "head.layers.N.self_attn.out_proj.bias");
        NEED(w->n1w, "head.layers.N.norm1.weight"); NEED(w->n1b, "head.layers.N.norm1.bias");
        NEED(w->n2w, "head.layers.N.norm2.weight"); NEED(w->n2b, "head.layers.N.norm2.bias");
        NEED(w->l1w, "head.layers.N.linear1.weight"); NEED(w->l1b, "head.layers.N.linear1.bias");
        NEED(w->l2w, "head.layers.N.linear2.weight"); NEED(w->l2b, "head.layers.N.linear2.bias");
    }
    NEED(h->sc0w, "scorer.0.weight"); NEED(h->sc0b, "scorer.0.bias");
    NEED(h->sc1w, "scorer.1.weight"); NEED(h->sc1b, "scorer.1.bias");
    NEED(h->sc3w, "scorer.3.weight"); NEED(h->sc3b, "scorer.3.bias");
    NEED(h->act0w, "act_head.0.weight"); NEED(h->act0b, "act_head.0.bias");
    NEED(h->act2w, "act_head.2.weight"); NEED(h->act2b, "act_head.2.bias");
#undef NEED
    h->loaded = 1;
    return 1;
}

/* B rows of L tokens through the model. ids [B*L] int32 (padded), att
 * [B*L] bytes (1 = token present), marker_pos [B*kmax] int32 (0 where
 * unused), marker_mask [B*kmax] bytes, qtype [B] int32. Writes logits
 * [B*kmax] (-1e4 at unused markers) and act [B*2], f32. Answers 0, or -1
 * with the error set. */
int lev_mlx_forward(struct lev_mlx *h, const int32_t *ids, const uint8_t *att, int B, int L,
                    const int32_t *marker_pos, const uint8_t *marker_mask, int kmax,
                    const int32_t *qtype, float *out_logits, float *out_act) {
    if (!h || !h->loaded) { if (h) set_error(h, "forward: weights not loaded"); return -1; }
    if (B < 1 || L < 1 || kmax < 1) { set_error(h, "forward: empty batch"); return -1; }
    struct fwd fs = {h, {{0}}, 0, 0}, *f = &fs;
    clear_error();
    mlx_array none = {0};
    int d = h->hidden;

    /* inputs */
    int bl[2] = {B, L}, bk[2] = {B, kmax}, b1[1] = {B};
    mlx_array ids_a = data(f, ids, bl, 2, MLX_INT32);
    mlx_array valid = data(f, att, bl, 2, MLX_BOOL); /* uint8 0/1 as bool */
    mlx_array mmask = data(f, marker_mask, bk, 2, MLX_BOOL);
    mlx_array qtype_a = data(f, qtype, b1, 1, MLX_INT32);

    /* masks: full [B,1,1,L] (key present); local: also |i-j| <= window,
     * with padded queries seeing every present key so no row is all
     * masked (they are never keys, never read) */
    mlx_array full = expand_dims(f, expand_dims(f, valid, 1), 1);
    mlx_array pos = arange(f, L);
    mlx_array diff = abs_(f, sub(f, expand_dims(f, pos, 1), expand_dims(f, pos, 0))); /* [L, L] */
    mlx_array win = mlx_array_new_int(h->window); keep(f, win);
    mlx_array local = less_equal(f, diff, win);                                   /* [L, L] */
    local = expand_dims(f, expand_dims(f, local, 0), 0);                          /* [1,1,L,L] */
    mlx_array padded_q = logical_not(f, expand_dims(f, expand_dims(f, valid, 1), 3)); /* [B,1,L,1] */
    local = logical_and(f, logical_or(f, local, padded_q), full);                 /* [B,1,L,L] */

    /* encoder */
    mlx_array x = layer_norm(f, take(f, h->tok_emb, ids_a, 0), h->emb_norm, none, h->norm_eps);
    for (int i = 0; i < h->layers && !f->failed; i++)
        x = encoder_layer(f, i, x, h->sliding[i] ? local : full, B, L);
    x = layer_norm(f, x, h->final_norm, none, h->norm_eps);

    /* head */
    x = add(f, x, expand_dims(f, take(f, h->type_emb, qtype_a, 0), 1));
    for (int i = 0; i < h->head_layers && !f->failed; i++)
        x = head_layer(f, i, x, full, B, L);

    /* the scorer at the markers: rows b*L + pos of the flat [B*L, d] */
    int32_t *flat_idx = malloc((size_t)B * kmax * sizeof *flat_idx);
    if (!flat_idx) { release(f); set_error(h, "forward: out of memory"); return -1; }
    for (int b = 0; b < B; b++)
        for (int k = 0; k < kmax; k++) {
            int32_t p = marker_pos[b * kmax + k];
            flat_idx[b * kmax + k] = b * L + (p > 0 ? p : 0);
        }
    int bkn[1] = {B * kmax};
    mlx_array idx = data(f, flat_idx, bkn, 1, MLX_INT32);
    free(flat_idx);
    int flat_shape[2] = {B * L, d};
    mlx_array hflat = reshape(f, x, flat_shape, 2);
    mlx_array markers = take(f, hflat, idx, 0); /* [B*kmax, d] */
    mlx_array sc = layer_norm(f, markers, h->sc0w, h->sc0b, 1e-5f);
    sc = gelu(f, linear(f, sc, h->sc1w, h->sc1b), h->dtype);
    sc = linear(f, sc, h->sc3w, h->sc3b); /* [B*kmax, 1] */
    mlx_array logits = astype(f, reshape(f, sc, bk, 2), MLX_FLOAT32);
    logits = where(f, mmask, logits, scalar(f, -1e4f, MLX_FLOAT32));

    /* act features: top1, top1 - top2, normalized entropy, k / 255 */
    mlx_array p = softmax(f, logits, -1);
    mlx_array two = mlx_array_new_int(2); keep(f, two);
    mlx_array kcount = astype(f, maximum(f, sum_axis(f, astype(f, mmask, MLX_INT32), -1), two), MLX_FLOAT32); /* [B] */
    mlx_array floor_ = scalar(f, 1e-9f, MLX_FLOAT32);
    mlx_array ent = divide(f, sub(f, scalar(f, 0.0f, MLX_FLOAT32), sum_axis(f, mul(f, p, logf_(f, maximum(f, p, floor_))), -1)),
                           logf_(f, kcount));
    mlx_array sorted = sort_axis(f, p, -1);
    int st0[2] = {0, kmax - 1}, st1[2] = {B, kmax}, st2[2] = {0, kmax - 2}, st3[2] = {B, kmax - 1}, one2[2] = {1, 1};
    mlx_array top1 = squeeze_axis(f, slice(f, sorted, st0, st1, one2, 2), 1);
    mlx_array top2 = squeeze_axis(f, slice(f, sorted, st2, st3, one2, 2), 1);
    mlx_array feats_v[4] = {top1, sub(f, top1, top2), ent, divide(f, kcount, scalar(f, 255.0f, MLX_FLOAT32))};
    mlx_vector_array fv = mlx_vector_array_new_data(feats_v, 4);
    mlx_array feats = stack(f, fv, 1); /* [B, 4] */
    mlx_vector_array_free(fv);
    int c0[3] = {0, 0, 0}, c1[3] = {B, 1, d}, c2[3] = {1, 1, 1}, bd[2] = {B, d};
    mlx_array cls = astype(f, reshape(f, slice(f, x, c0, c1, c2, 3), bd, 2), MLX_FLOAT32);
    mlx_array pooled_v[2] = {cls, feats};
    mlx_vector_array pv = mlx_vector_array_new_data(pooled_v, 2);
    mlx_array pooled = astype(f, concat(f, pv, 1), h->dtype); /* [B, d+4] */
    mlx_vector_array_free(pv);
    mlx_array act = astype(f, linear(f, gelu(f, linear(f, pooled, h->act0w, h->act0b), h->dtype), h->act2w, h->act2b), MLX_FLOAT32);

    if (f->failed) { release(f); return -1; }

    /* one evaluation for both outputs */
    mlx_array outs[2] = {logits, act};
    mlx_vector_array ov = mlx_vector_array_new_data(outs, 2);
    int rc = mlx_eval(ov);
    mlx_vector_array_free(ov);
    if (rc) { fail(f, "eval"); release(f); return -1; }
    const float *lp = mlx_array_data_float32(logits);
    const float *ap = mlx_array_data_float32(act);
    if (!lp || !ap) { set_error(h, "forward: outputs are not f32"); release(f); return -1; }
    memcpy(out_logits, lp, (size_t)B * kmax * sizeof *out_logits);
    memcpy(out_act, ap, (size_t)B * 2 * sizeof *out_act);
    release(f);
    return 0;
}

void lev_mlx_free(struct lev_mlx *h) {
    if (!h) return;
#define DROP(a) if ((a).ctx) { mlx_array_free(a); (a).ctx = NULL; }
    DROP(h->tok_emb); DROP(h->emb_norm); DROP(h->final_norm); DROP(h->type_emb);
    for (int i = 0; i < LEV_MAX_LAYERS; i++) {
        struct enc_layer *w = &h->enc[i];
        DROP(w->attn_norm); DROP(w->wqkv); DROP(w->wo); DROP(w->mlp_norm); DROP(w->wi); DROP(w->wo_mlp);
    }
    for (int i = 0; i < LEV_MAX_HEAD_LAYERS; i++) {
        struct head_layer *w = &h->head[i];
        DROP(w->in_w); DROP(w->in_b); DROP(w->out_w); DROP(w->out_b); DROP(w->n1w); DROP(w->n1b);
        DROP(w->n2w); DROP(w->n2b); DROP(w->l1w); DROP(w->l1b); DROP(w->l2w); DROP(w->l2b);
    }
    DROP(h->sc0w); DROP(h->sc0b); DROP(h->sc1w); DROP(h->sc1b); DROP(h->sc3w); DROP(h->sc3b);
    DROP(h->act0w); DROP(h->act0b); DROP(h->act2w); DROP(h->act2b);
#undef DROP
    if (h->s.ctx) mlx_stream_free(h->s);
    free(h);
}

/* MLX's allocator keeps freed buffers for reuse; a server that has
 * unloaded a model gives them back with this. Answers the active bytes. */
int64_t lev_mlx_memory(int clear) {
    size_t active = 0;
    if (clear) mlx_clear_cache();
    mlx_get_active_memory(&active);
    return (int64_t)active;
}
