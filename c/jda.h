#pragma once

#ifdef _MSC_VER
#ifdef JDA_EXPORTS
#define JDA_API __declspec(dllexport)
#else
#define JDA_API __declspec(dllimport)
#endif
#else
#define JDA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*! \breif result */
typedef struct {
  int n; // number of faces
  int landmark_n; // number of landmarks
  int *bboxes; // bboxes of faces, (x, y, size)
  float *shapes; // shapes of faces, (x1, y1, x2, y2, ...)
  float *scores; // score of faces
  int total_patches;
} jdaResult;

/*!
 * \breif create jda cascador
 * \param model   model file
 * \return        cascador, NULL if failed
 */
JDA_API void *jdaCascadorCreate(const char *model);

/*!
 * \breif release jda cascador
 * \param cascador  jda cascador
 */
JDA_API void jdaCascadorRelease(void *cascador);

/*!
 * \breif detect face
 * \param cascador  jda cascador
 * \param img       image data
 * \param width     image width
 * \param height    image height
 * \return          detect result
 */
JDA_API jdaResult jdaDetect(void *cascador, unsigned char *img, int width, int height);

/*!
 * \breif release detection result memory
 */
JDA_API void jdaResultRelease(jdaResult result);

#ifdef __cplusplus
}
#endif
