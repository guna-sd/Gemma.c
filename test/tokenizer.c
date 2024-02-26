#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    int id;
    char *token_str;
} Token;


typedef struct {
    char** vocab;
    float* token_scores;
    Token *sorted_tokens;
    unsigned int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;


void build_tokenizer(Tokenizer *tokenizer, const char *tokenizer_path)
{
    FILE *file = fopen(tokenizer_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Could not open tokenizer file %s\n", tokenizer_path);
        exit(1);
    }
    fread(&tokenizer->max_token_length, sizeof(unsigned int), 1, file);
    fread(&tokenizer->vocab_size, sizeof(unsigned int), 1, file);
    tokenizer->vocab = (char**) malloc(sizeof(char*) * tokenizer->vocab_size);
    tokenizer->token_scores = (float*) malloc(sizeof(float) * tokenizer->vocab_size);
    tokenizer->sorted_tokens = NULL;
    for (int i = 0; i < 256; i++) {
        tokenizer->byte_pieces[i * 2] = (unsigned char)i;
        tokenizer->byte_pieces[i * 2 + 1] = '\0';
    }

    int length;
    for (int i = 0; i < tokenizer->vocab_size; i++) 
    {
        fread(tokenizer->token_scores + i, sizeof(float), 1, file);
        fread(&length, sizeof(int), 1, file);
        tokenizer->vocab[i] = (char*) malloc(sizeof(char) * length + 1);
        fread(tokenizer->vocab[i], length, 1, file);
        tokenizer->vocab[i][length] = '\0';
    }
    fclose(file);
}


void free_tokenizer(Tokenizer *tokenizer)
{
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab[i]);
    }
    free(tokenizer->vocab);
    free(tokenizer->token_scores);
    free(tokenizer->sorted_tokens);
}


int compare_tokens(const void *a, const void *b)
{
    return strcmp(((Token*)a)->str, ((Token*)b)->str);
}

int str_lookup(char *str, Token *sorted_tokens, int vocab_size)
{
    Token tok = {.str = str};
    Token *next = bsearch(&tok, sorted_tokens, vocab_size, sizeof(Token), compare_tokens);
    return next != NULL ? next->id : -1;
}

void encode(Tokenizer *tokenizer, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
{
    if (text == NULL)
    {
        fprintf(stderr, "Error no text to encode\n");
        exit(1);
    }
    if (tokenizer->sorted_tokens == NULL)
    {
        tokenizer->sorted_tokens = (Token*) malloc(sizeof(Token) * tokenizer->vocab_size);
        for (int i = 0; i < tokenizer->vocab_size; i++)
        {
            tokenizer->sorted_tokens[i].str = tokenizer->vocab[i];
            tokenizer->sorted_tokens[i].id = i;
        }
        qsort(tokenizer->sorted_tokens, tokenizer->vocab_size, sizeof(Token), compare_tokens);
    }

    char *string_buffer = malloc((tokenizer->msx_token_length * 2 +1 +2) * sizeof(char));
    size_t string_len = 0;

    *n_tokens = 0;

    if (bos)
    {
        tokens[(*n_tokens)++] = 1;
    }

    if (text[0] == '\0')
    {
        int prefix = str_lookup(" ", tokenizer->sorted_tokens, tokenizer->vocab_size);
        tokens[(*n_tokens)++] = prefix;
    }

    for (char *c = text; *c != '\0'; c++)
    {
        if ((*c & 0xC0) != 0x80)
        {
            string_len=0;
        }
        string_buffer[string_len++] = *c;
        string_buffer[string_len] = '\0';

        if ((*(c+1) & 0xC0) != 0x80 && string_len < 4)
        {
            continue;
        }
        int id = str_lookup(string_buffer, tokenizer->sorted_tokens, tokenizer->vocab_size);
        
        if (id != -1)
        {
            tokens[(*n_tokens)++] = id;
        }
        else {
            for (int i = 0; i < string_len; i++)
            {
                tokens[(*n_tokens)++] = (unsigned char)string_buffer[i] + 3;
            }
        }
        string_len = 0;
    }

    while (1)
    {
        float best_score = 
    }
}