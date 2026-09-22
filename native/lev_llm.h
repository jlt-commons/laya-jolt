/* lev_llm.h — the handle lev_llm.c and lev_decision.cpp share. */
#ifndef LEV_LLM_H
#define LEV_LLM_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct lev_llm {
    struct llama_model *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    int n_ctx;
    int n_batch;
    int n_seq_max;
    int n_vocab;
    /* the decision engine (lev_decision.cpp), created by the first
     * lev_llm_jev; it keeps the cached prefix in the context's memory */
    void *jev;
    /* the texts of the control / unknown tokens, for lev_llm_escape
     * (collected on its first call) */
    const char **specials;
    int n_specials;
    char error[512];
};

/* drop the decision engine: its cached prefix is gone once another call
 * clears the context's memory */
void lev_jev_forget(struct lev_llm *h);

#ifdef __cplusplus
}
#endif

#endif
