#ifndef GLINER_CANDLE_H
#define GLINER_CANDLE_H
#ifdef __cplusplus
extern "C" {
#endif
/* Model handles are owned by the caller and must be used serially. Open always
   requests Metal device 0 and F32; it never falls back to CPU. Returns NULL on
   failure and writes an owned UTF-8 error string when error is non-NULL. */
typedef struct gliner_candle_model gliner_candle_model;
gliner_candle_model *gliner_candle_open(const char *checkpoint_directory, char **error);
/* request is a NUL-terminated UTF-8 JSON request. The model must be a valid,
   live handle created by open, with no concurrent operations or close. Returns
   owned UTF-8 JSON or NULL on error. */
char *gliner_candle_extract(gliner_candle_model *model, const char *request, char **error);
/* Close exactly once, after all operations finish; NULL is accepted. */
void gliner_candle_close(gliner_candle_model *model);
/* Release each returned result/error string exactly once; NULL is accepted. */
void gliner_candle_string_free(char *string);
#ifdef __cplusplus
}
#endif
#endif
