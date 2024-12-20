/*reference from llama2.c*/
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

typedef struct {
    int dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int hidden_dim;
    int head_dim;
    int vocab_size;
    int max_seq_len;
    float eps;
} Config;

typedef struct {
    float *embeddings;
    float *attn_norm;
    float *wqkv;
    float *wo;
    float *post_attn_norm;
    float *w1;
    float *w2;
    float *w3;
    float *layer_norm;
    float *wcls;
}Weights;

typedef struct {
    float *x;
    float *x2;
    float *xt;
    float *h;
    float *h2;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *logits;
    float *key_cache;
    float *value_cache;
}RunState;

typedef struct {
    Config config;
    Weights weights;
    RunState runstate;
    int fd;
    float *data;
    ssize_t file_size;
} Transformer;

void alloc_runstate(Runstate *state, Config *params)
{
    int kv_size = (params->dim * params->n_kv_heads) / p->n_heads;
    state->x =  calloc(params->dim, sizeof(float));
    state->x2 = calloc(params->dim, sizeof(float));
    state->xt = calloc(params->dim, sizeof(float));
    state->h =  calloc(params->hidden_dim, sizeof(float));
    state->h2 =  calloc(params->hidden_dim, sizeof(float));
    state->q =  calloc(params->hidden_dim, sizeof(float));
    state->key_cache = calloc(p->n_layers * p->max_seq_len * kv_size, sizeof(float));
    state->key_cache = calloc(p->n_layers * p->max_seq_len * kv_size, sizeof(float));
    state->attn = calloc(p->n_heads * p->max_seq_len, sizeof(float));
    state->logits = calloc(p->vocab_size, sizeof(float));
    if (!state->x || !state->x2 || !state->xt || !state->h || !state->h2 || !state->q
     || !state->key_cache || !state->value_cache || !state->attn || !state->logits) {
        fprintf(stderr, "memmory allocation failed for runstate!\n");
        exit(EXIT_FAILURE);
}

void free_runstate(Runstate *state)
{
    free(state->x);
    free(state->x2);
    free(state->xt);
    free(state->h);
    free(state->h2);
    free(state->q);
    free(state->attn);
    free(state->logits);
    free(state->key_cache);
    free(state->value_cache);
}

void matmul(float *out,float *x, float *y, int n, int dim)
{
    int i,j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < dim; i++)
    {
        float val = 0.0f;
        for (j = 0; j < n; j++)
        {
            val += w[i *n + j] * x[j];
        }
        out[i] = val;
    }
}

void rms_norm(float *o, float *x, float *w, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    float inv_sqrt = 1.0f / sqrtf(sum_sq + eps);

    for (int i = 0; i < dim; i++) {
        o[i] = x[i] * inv_sqrt * (w + 1.0f);
    }
}

void softmax(float *x, int size)
{
    float max = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max)
        {
            max = x[i];
        }
    }
    float sum = 0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}


































/*code for tokenization encode and decode*/

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void build_tokenizer(Tokenizer *tokenizer, char *tokenizer_path)
{
    for (int i =0; i < 256 ; i++) {
        tokenizer->byte_pieces[i *2] = (unsigned char)i;
        tokenizer->byte_pieces[i *2 + 1] = '\0';
        }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Could not open file %s\n", tokenizer_path);
        exit(1);
    }
    if(fread(&tokenizer->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Could not read max token length %s\n", tokenizer_path);
        exit(1);
    }
    if(fread(&tokenizer->vocab_size, sizeof(int), 1, file)!= 1)
    {
        fprintf(stderr, "Could not read vocab size %s\n", tokenizer_path);
        exit(1);
    }
    int vocab_size = tokenizer->vocab_size;
    tokenizer->vocab = (char**)malloc(vocab_size * sizeof(char*));
    tokenizer->vocab_scores = (float*)malloc(vocab_size * sizeof(float));

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(tokenizer->vocab_scores + i, sizeof(float), 1, file) != 1)
        { 
            fprintf(stderr, "failed read\n"); 
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) 
        { 
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE); 
        }
        tokenizer->vocab[i] = (char *)malloc(len + 1);
        if (fread(tokenizer->vocab[i], len, 1, file) != 1) 
        {
             fprintf(stderr, "failed read\n"); 
             exit(EXIT_FAILURE); 
        }
        tokenizer->vocab[i][len] = '\0';
    }
    if ((tokenizer->sorted_vocab = malloc(vocab_size * sizeof(TokenIndex))))
    {
        for (int i =0; i < tokenizer->vocab_size; i++)
        {
            tokenizer->sorted_vocab[i].id = i;
            tokenizer->sorted_vocab[i].str = tokenizer->vocab[i];
        }
        qsort(tokenizer->sorted_vocab, tokenizer->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    fclose(file);
    return;
}

void free_tokenizer(Tokenizer *tokenizer)
{
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab[i]);
    }
    free(tokenizer->vocab);
    free(tokenizer->vocab_scores);
    free(tokenizer->sorted_vocab);
}

void encode(Tokenizer* t, char *text, bool bos, bool eos, int *tokens, int *n_tokens) {

    if (text == NULL) { 
        fprintf(stderr, "cannot encode NULL text\n"); 
        exit(EXIT_FAILURE); 
        }

    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    *n_tokens = 0;

    if (bos == true) {
        tokens[(*n_tokens)++] = 2;
    }

    for (char *c = text; *c != '\0'; c++) {

        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos == true) {
        tokens[(*n_tokens)++] = 1;
    }
    free(str_buffer);
}


char *decode(Tokenizer *tokenizer, int prev_token, int token)
{
    char *string = tokenizer->vocab[token];

    if (prev_token == 1 && string[0] == ' ')
    {
        string++;
    }
    unsigned char byte_val;
    if (sscanf(string, "<0x%02hhX>", &byte_val) == 1)
    {
        string = (char *)tokenizer->byte_pieces + byte_val * 2;
    }
    return string;
}
