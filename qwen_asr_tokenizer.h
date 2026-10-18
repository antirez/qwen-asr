/*
 * qwen_asr_tokenizer.h - Qwen2/Qwen3 BPE tokenizer (GPT-2 byte-level)
 *
 * Supports:
 *  - decode token ID -> UTF-8 text
 *  - encode UTF-8 text -> token IDs (vocab.json + merges.txt, or tokenizer.json)
 */

#ifndef QWEN_ASR_TOKENIZER_H
#define QWEN_ASR_TOKENIZER_H

typedef struct {
    char **id_to_text;   /* [vocab_size] decoded text strings */
    char **id_to_bpe;    /* [vocab_size] raw BPE token strings from vocab.json */
    int vocab_size;

    /* Internal hash maps (opaque to callers) */
    void *vocab_map;
    int vocab_map_cap;
    void *merge_map;
    int merge_map_cap;
} qwen_tokenizer_t;

/* Load from model dir: vocab.json + merges.txt, else tokenizer.json (-hf repos) */
qwen_tokenizer_t *qwen_tokenizer_load(const char *model_dir);

/* Decode a single token ID to text. Returns pointer to internal string. */
const char *qwen_tokenizer_decode(const qwen_tokenizer_t *tok, int token_id);

/* Encode UTF-8 text into token IDs using BPE.
 * Returns malloc'd array of token IDs and sets *out_n_tokens.
 * Returns NULL on error (and sets *out_n_tokens to 0). */
int *qwen_tokenizer_encode(const qwen_tokenizer_t *tok, const char *text, int *out_n_tokens);

/* Free tokenizer */
void qwen_tokenizer_free(qwen_tokenizer_t *tok);

#endif /* QWEN_ASR_TOKENIZER_H */
