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

/*! \brief result */
typedef struct {
  int n; // number of faces
  int landmark_n; // number of landmarks
  int *bboxes; // bboxes of faces, (x, y, size)
  float *shapes; // shapes of faces, (x1, y1, x2, y2, ...)
  float *scores; // score of faces
  int total_patches;
} jdaResult;

/*!
 * \brief create jda cascador
 * \param model   model file
 * \return        cascador, NULL if failed
 */
JDA_API void *jdaCascadorCreateDouble(const char *model);
JDA_API void *jdaCascadorCreateFloat(const char *model);

/*!
 * \brief serialize model to a binary file
 * \note  this function serialze float data type, can reduce model size
 *
 * \param cascador  jda cascador
 * \param model     model file
 */
JDA_API void jdaCascadorSerializeTo(void *cascador, const char *model);

/*!
 * \brief release jda cascador
 * \param cascador  jda cascador
 */
JDA_API void jdaCascadorRelease(void *cascador);

/*!
 * \brief detect face
 * \param cascador  jda cascador
 * \param img       image data
 * \param width     image width
 * \param height    image height
 * \return          detect result
 */
JDA_API jdaResult jdaDetect(void *cascador, unsigned char *img, int width, int height);

/*!
 * \brief release detection result memory
 */
JDA_API void jdaResultRelease(jdaResult result);

#ifdef __cplusplus
}
#endif
