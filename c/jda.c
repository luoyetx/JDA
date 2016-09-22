#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h> // C99 needed
#include "jda.h"

#ifdef _MSC_VER
#define inline __inline
#endif

/*!
 * \brief jda global constance
 * \param JDA_T               number of stages
 * \param JDA_K               number of carts every stage
 * \param JDA_LANDMARK_N      number of landmarks
 * \param JDA_TREE_DEPTH      depth of a cart
 * \param JDA_TREE_LEAF_N     leaf number of a cart
 * \param JDA_CART_N          number of total carts in the model
 * \param JDA_LANDMARK_DIM    dimension of landmarks
 * \param JDA_LBF_N           dimension of local binary feature
 */
#define JDA_T             5
#define JDA_K             540
#define JDA_LANDMARK_N    27
#define JDA_TREE_DEPTH    4
#define JDA_TREE_LEAF_N   (1 << (JDA_TREE_DEPTH - 1))
#define JDA_TREE_NODE_N   (JDA_TREE_LEAF_N - 1)
#define JDA_CART_N        (JDA_T*JDA_K)
#define JDA_LANDMARK_DIM  (2 * JDA_LANDMARK_N)
#define JDA_LBF_N         (JDA_K*JDA_TREE_LEAF_N)

/*!
 * \brief A marco based generic vector type for C
 * \note  the vector is only support for operation `insert`
 *
 * \usage
 *  1. define the type
 *      JDA_VECTOR(int);
 *  2. define the variable
 *      JDA_VECTOR_DEC(int) vector_of_int;
 *  3. malloc initial memory, no element
 *      JDA_VECTOR_NEW(vector_of_int, size);
 *  4. insert an element, resize vector if needed
 *      JDA_VECTOR_INSERT(vector_of_int, value);
 *  5. insert some elements, resize vector if needed
 *      JDA_VECTOR_INSERT_MORE(vector_of_int, values, size)
 */

#define JDA_VECTOR_DEF(Type) \
  struct jdaVector##Type { \
    int size; \
    int capacity; \
    Type *data; \
  }

#define JDA_VECTOR(Type) \
  struct jdaVector##Type

#define JDA_VECTOR_NEW(v, size_) \
  do { \
    (v).size = 0; \
    (v).capacity = 2 * size_; \
    (v).data = malloc((v).capacity * sizeof((v).data[0])); \
  } while (0)

#define JDA_VECTOR_INSERT(v, value) \
  do { \
    (v).size++; \
    if ((v).size > (v).capacity) { \
      int capacity_new = 2 * (v).capacity; \
      (v).data = realloc((v).data, capacity_new * sizeof(value)); \
      (v).capacity = capacity_new; \
    } \
    (v).data[(v).size - 1] = (value); \
  } while (0)

#define JDA_VECTOR_INSERT_MORE(v, values, size_) \
  do { \
    int size_new; \
    size_new = (v).size + size_; \
    if (size_new > (v).capacity) { \
      int capacity_new = 2 * (((v).capacity < size_new) ? size_new : (v).capacity); \
      (v).data = realloc((v).data, capacity_new * sizeof((values)[0])); \
      (v).capacity = capacity_new; \
    } \
    memcpy(&(v).data[(v).size], values, size_ * sizeof((values)[0])); \
    (v).size = size_new; \
  } while (0)

#define JDA_VECTOR_RELEASE(v) \
  do { \
    if ((v).data) free((v).data) \
  } while (0)

#define JDA_VECTOR_DEFAULT_LEN  200

JDA_VECTOR_DEF(int);
JDA_VECTOR_DEF(float);

// data structures

/*! \brief jda bbox */
typedef struct {
  /*! breif x, y, w, h */
  int x, y, size;
} jdaBBox;

/*! \brief jda shape */
typedef float jdaShape[JDA_LANDMARK_DIM];

/*!\brief jda cart node */
typedef struct {
  /*! breif scale */
  int scale;
  /*! breif landmark id */
  int landmark1_x;
  int landmark2_x;
  /*! breif landmark offset to generate feature value */
  float landmark1_offset_x;
  float landmark1_offset_y;
  float landmark2_offset_x;
  float landmark2_offset_y;
  /*! \brief feature threshold */
  int th;
} jdaNode;

/*! \brief jda cart */
typedef struct {
  /*! \brief nodes in this cart */
  jdaNode nodes[JDA_TREE_NODE_N];
  /*! \brief scores stored in the leaf nodes */
  float score[JDA_TREE_LEAF_N];
  /*! \brief score thrshold */
  float th;
  /*! \brief mean and std apply to the score */
  float mean, std;
} jdaCart;

/*! \brief jda cascador */
typedef struct {
  /*! \brief all carts in the model */
  jdaCart carts[JDA_CART_N];
  /*! \brief regression weights of every stage */
  float ws[JDA_T][JDA_LBF_N][JDA_LANDMARK_DIM];
  /*! \brief mean shape of the face */
  float mean_shape[JDA_LANDMARK_DIM];
  /*! \brief final score threshold */
  float th;
} jdaCascador;

/*! \brief jda image */
typedef struct {
  /*! \brief width and height */
  int w, h;
  /*! \brief step of a row in the image, usally equals to width */
  int step;
  /*! \brief gray image data */
  unsigned char *data;
} jdaImage;

// Internal Helpers

#define JDA_IMAGE_AT(img, x, y) ((img)->data[(y)*(img)->step+(x)])

#define JDA_MAX(x, y) (((x)<(y))?(y):(x))
#define JDA_MIN(x, y) (((x)<(y))?(x):(y))

/*!
 * \brief create image
 * \param w   w
 * \param h   h
 * \return    image
 */
static inline
jdaImage jdaImageCreate(int w, int h) {
  jdaImage img;
  img.w = img.step = w;
  img.h = h;
  img.data = (unsigned char*)malloc(w*h*sizeof(unsigned char));
  return img;
}

/*!
 * \brief release internal data buffer
 * \note  don't release the image which borrow from others
 *
 * \param img   image to free
 */
static inline
void jdaImageRelease(jdaImage *img) {
  if (img->data) free(img->data);
}

/*!
 * \brief resize image, bilinear interpolation
 * \param img   image
 * \param w     w
 * \param h     h
 * \return      new image with size = (w, h)
 */
static jdaImage jdaImageResize(jdaImage img, int w, int h) {
  jdaImage img_ = jdaImageCreate(w, h);
  float x_ratio = (float)(img.w - 1) / w;
  float y_ratio = (float)(img.h - 1) / h;
  int x, y, index;
  int a, b, c, d;
  float x_diff, y_diff;
  int offset = 0;
  int i, j;
  for (i = 0; i < h; i++) {
    for (j = 0; j < w; j++) {
      x = (int)(x_ratio*j);
      y = (int)(y_ratio*i);
      x_diff = (x_ratio*j) - x;
      y_diff = (y_ratio*i) - y;
      index = y*img.w + x;
      a = img.data[index];
      b = img.data[index + 1];
      c = img.data[index + img.w];
      d = img.data[index + img.w + 1];
      img_.data[offset++] = (unsigned char)(a*(1.f - x_diff)*(1.f - y_diff) + \
                                            b*(x_diff)*(1.f - y_diff) + \
                                            c*(1.f - x_diff)*(y_diff) + \
                                            d*(x_diff)*(y_diff));
    }
  }
  return img_;
}

/*!
 * \brief nms
 * \param result  jda detection result
 * \return        merged result
 */
static jdaResult jdaNms(jdaResult result) {
  float overlap = 0.3f;

  int n = result.n;
  float *scores = result.scores;
  jdaBBox *bboxes = (jdaBBox*)result.bboxes;
  jdaShape *shapes = (jdaShape*)result.shapes;

  int *idx = (int*)malloc(n*sizeof(int));
  bool *flag = (bool*)malloc(n*sizeof(bool));
  int *area = (int*)malloc(n*sizeof(int));
  int i, j;
  for (i = 0; i < n; i++) {
    idx[i] = i;
    flag[i] = true;
    area[i] = bboxes[i].size*bboxes[i].size;
  }

  // sort by score
  for (i = 0; i < n - 1; i++) {
    for (j = i + 1; j < n; j++) {
      if (scores[idx[i]] < scores[idx[j]]) {
        int temp = idx[i];
        idx[i] = idx[j];
        idx[j] = temp;
      }
    }
  }

  // merge
  for (i = 0; i < n - 1; i++) {
    int k1 = idx[i];
    if (flag[k1] == false) continue;
    for (j = i + 1; j < n; j++) {
      int k2 = idx[j];
      if (flag[k2] == false) continue;
      int x1 = JDA_MAX(bboxes[k1].x, bboxes[k2].x);
      int y1 = JDA_MAX(bboxes[k1].y, bboxes[k2].y);
      int x2 = JDA_MIN(bboxes[k1].x + bboxes[k1].size, bboxes[k2].x + bboxes[k2].size);
      int y2 = JDA_MIN(bboxes[k1].y + bboxes[k1].size, bboxes[k2].y + bboxes[k2].size);
      int w = JDA_MAX(0, x2 - x1);
      int h = JDA_MAX(0, y2 - y1);
      float ov = (float)(w*h) / (float)(area[k1] + area[k2] - w*h);
      if (ov > overlap) {
        flag[k2] = false;
      }
    }
  }

  // move
  jdaResult merged;
  JDA_VECTOR(int) merged_bboxes;
  JDA_VECTOR(float) merged_shapes;
  JDA_VECTOR(float) merged_scores;
  JDA_VECTOR_NEW(merged_scores, n);
  JDA_VECTOR_NEW(merged_bboxes, n * 3);
  JDA_VECTOR_NEW(merged_shapes, n * JDA_LANDMARK_DIM);

  for (i = 0; i < n; i++) {
    if (flag[i] == true) {
      JDA_VECTOR_INSERT(merged_scores, scores[i]);
      JDA_VECTOR_INSERT_MORE(merged_bboxes, (int*)&bboxes[i], 3);
      JDA_VECTOR_INSERT_MORE(merged_shapes, (float*)&shapes[i], JDA_LANDMARK_DIM);
    }
  }
  merged.n = merged_scores.size;
  merged.landmark_n = JDA_LANDMARK_N;
  merged.bboxes = merged_bboxes.data; // transfer memory
  merged.shapes = merged_shapes.data; // transfer memory
  merged.scores = merged_scores.data; // transfer memory

  free(flag);
  free(area);
  free(idx);
  // free previous result
  free(result.scores);
  free(result.shapes);
  free(result.bboxes);
  return merged;
}

static jdaResult jdaInternalDetect(jdaCascador *cascador, jdaImage o, jdaImage h, jdaImage q, \
                                   float scale, float step, int min_size, int max_size, float th) {
  int win_size = 24; // fixed initial window size
  max_size = JDA_MIN(max_size, o.w);
  max_size = JDA_MIN(max_size, o.h);

  JDA_VECTOR(int) bboxes;
  JDA_VECTOR(float) shapes;
  JDA_VECTOR(float) scores;
  JDA_VECTOR_NEW(scores, JDA_VECTOR_DEFAULT_LEN);
  JDA_VECTOR_NEW(bboxes, JDA_VECTOR_DEFAULT_LEN * 3);
  JDA_VECTOR_NEW(shapes, JDA_VECTOR_DEFAULT_LEN * JDA_LANDMARK_DIM);

  while (win_size < min_size) win_size *= scale;
  for (; win_size <= max_size; win_size *= scale) {
    const int step = (int)(win_size*0.1f);
    const int x_max = o.w - win_size;
    const int y_max = o.h - win_size;

    int x, y;
    for (y = 0; y <= y_max; y += step) {
      for (x = 0; x <= x_max; x += step) {
        jdaImage ps[3];
        const float r = 1.f / sqrtf(2.f);
        ps[0].w = ps[0].h = win_size;
        ps[0].step = o.step;
        ps[0].data = &o.data[y*o.step + x]; // borrow memory
        int h_x = (int)(x*r);
        int h_y = (int)(y*r);
        ps[1].w = ps[1].h = win_size;
        ps[1].step = h.step;
        ps[1].data = &h.data[h_y*h.step + h_x]; // borrow memory
        int q_x = x / 2;
        int q_y = y / 2;
        ps[2].w = ps[2].h = win_size;
        ps[2].step = q.step;
        ps[2].data = &q.data[q_y*q.step + q_x]; // borrow memory

        // validate
        jdaShape shape;
        float score = 0.f;
        int lbf[JDA_K];
        jdaCart *cart = cascador->carts;
        memcpy(shape, cascador->mean_shape, JDA_LANDMARK_DIM*sizeof(float));
        int t, k, i;
        // stages
        for (t = 0; t < JDA_T; t++) {
          // carts
          for (k = 0; k < JDA_K; k++) {
            // nodes
            int node_idx = 0;
            for (i = 0; i < JDA_TREE_DEPTH - 1; i++) {
              jdaNode *node = &cart->nodes[node_idx];
              int landmark1 = node->landmark1_x;
              int landmark2 = node->landmark2_x;
              float x1 = shape[landmark1] + node->landmark1_offset_x;
              float y1 = shape[landmark1 + 1] + node->landmark1_offset_y;
              float x2 = shape[landmark2] + node->landmark2_offset_x;
              float y2 = shape[landmark2 + 1] + node->landmark2_offset_y;
              jdaImage *p = ps + node->scale;
              int x1_ = (int)(x1*p->w);
              int y1_ = (int)(y1*p->w);
              int x2_ = (int)(x2*p->w);
              int y2_ = (int)(y2*p->w);
              if (x1_ < 0) x1_ = 0;
              else if (x1_ >= p->w) x1_ = p->w - 1;
              if (x2_ < 0) x2_ = 0;
              else if (x2_ >= p->w) x2_ = p->w - 1;
              if (y1_ < 0) y1_ = 0;
              else if (y1_ >= p->w) y1_ = p->w - 1;
              if (y2_ < 0) y2_ = 0;
              else if (y2_ >= p->w) y2_ = p->w - 1;

              int feature = (int)JDA_IMAGE_AT(p, x1_, y1_) - (int)JDA_IMAGE_AT(p, x2_, y2_);
              if (feature <= node->th) node_idx = 2 * node_idx + 1;
              else node_idx = 2 * node_idx + 2;
            }
            int leaf_idx = node_idx - JDA_TREE_NODE_N;
            score += cart->score[leaf_idx];
            score = (score - cart->mean) / cart->std;
            // not a face
            if (score < cart->th) goto next;
            lbf[k] = k*JDA_TREE_LEAF_N + leaf_idx;
            cart++;
          }
          // regression
          jdaShape *ws = cascador->ws[t];
          for (k = 0; k < JDA_K; k++) {
            float *w = ws[lbf[k]];
            for (i = 0; i < JDA_LANDMARK_DIM; i += 2) {
              shape[i] += w[i];
              shape[i + 1] += w[i + 1];
            }
          }
        }
        // final threshold
        if (score < th) goto next;

        jdaBBox bbox;
        bbox.x = x; bbox.y = y;
        bbox.size = win_size;

        // may use openmp
        #pragma omp critical
        {
          JDA_VECTOR_INSERT(scores, score);
          JDA_VECTOR_INSERT_MORE(bboxes, (int*)&bbox, 3);
          JDA_VECTOR_INSERT_MORE(shapes, shape, JDA_LANDMARK_DIM);
        }
        next:;
      }
    }
  }

  jdaResult result;
  result.n = scores.size;
  result.landmark_n = JDA_LANDMARK_N;
  result.bboxes = bboxes.data; // transfer memory
  result.shapes = shapes.data; // transfer memory
  result.scores = scores.data; // transfer memory
  return result;
}

// APIs

jdaResult jdaDetect(void *cascador, unsigned char *data, int width, int height, \
                    float scale, float step, int min_size, int max_size, float th) {
  jdaImage o, h, q;
  o.w = o.step = width;
  o.h = height;
  o.data = data; // borrow memory

  float r = 1.f / sqrtf(2.f);
  h.w = (int)(width*r);
  h.h = (int)(height*r);
  h = jdaImageResize(o, h.w, h.h);

  q.w = width / 2;
  q.h = height / 2;
  q = jdaImageResize(o, q.w, q.h);

  min_size = JDA_MAX(min_size, 24);
  if (max_size <= 0) max_size = JDA_MIN(o.w, o.h);
  jdaResult result = jdaInternalDetect((jdaCascador*)cascador, o, h, q, scale, \
                                       step, min_size, max_size, th);
  jdaResult merged = jdaNms(result);
  int i, j;
  for (i = 0; i < merged.n; i++) {
    int x = merged.bboxes[3 * i];
    int y = merged.bboxes[3 * i + 1];
    int size = merged.bboxes[3 * i + 2];
    float *shape = &merged.shapes[i*JDA_LANDMARK_DIM];
    for (j = 0; j < JDA_LANDMARK_N; j++) {
      shape[2 * j] = shape[2 * j] * size + x;
      shape[2 * j + 1] = shape[2 * j + 1] * size + y;
    }
  }

  jdaImageRelease(&h);
  jdaImageRelease(&q);

  return merged;
}

/*!
 * \brief serialize model from JDA
 * \note  JDA dump data type is double
 */
void *jdaCascadorCreateDouble(const char *model) {
  FILE *fin = fopen(model, "rb");
  if (!fin) return NULL;
  jdaCascador *cascador = (jdaCascador*)malloc(sizeof(jdaCascador));
  if (!cascador) {
    fclose(fin);
    return NULL;
  }

  int i4;
  double f8;
  int t, k, i, j;
  // meta
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  // mean shape
  for (i = 0; i < JDA_LANDMARK_DIM; i++) {
    fread(&f8, sizeof(double), 1, fin);
    cascador->mean_shape[i] = (float)f8;
  }
  // carts
  for (t = 0; t < JDA_T; t++) {
    for (k = 0; k < JDA_K; k++) {
      jdaCart *cart = &cascador->carts[t*JDA_K + k];
      // feature
      for (i = 0; i < JDA_TREE_NODE_N; i++) {
        jdaNode *node = &cart->nodes[i];
        fread(&i4, sizeof(int), 1, fin);
        node->scale = i4;
        fread(&i4, sizeof(int), 1, fin);
        node->landmark1_x = i4 << 1;
        fread(&i4, sizeof(int), 1, fin);
        node->landmark2_x = i4 << 1;
        fread(&f8, sizeof(double), 1, fin);
        node->landmark1_offset_x = (float)f8;
        fread(&f8, sizeof(double), 1, fin);
        node->landmark1_offset_y = (float)f8;
        fread(&f8, sizeof(double), 1, fin);
        node->landmark2_offset_x = (float)f8;
        fread(&f8, sizeof(double), 1, fin);
        node->landmark2_offset_y = (float)f8;
        fread(&i4, sizeof(int), 1, fin);
        node->th = i4;
      }
      // scores
      for (i = 0; i < JDA_TREE_LEAF_N; i++) {
        fread(&f8, sizeof(double), 1, fin);
        cart->score[i] = (float)f8;
      }
      // classificatio threshold
      fread(&f8, sizeof(double), 1, fin);
      cart->th = (float)f8;
      fread(&f8, sizeof(double), 1, fin);
      cart->mean = (float)f8;
      fread(&f8, sizeof(double), 1, fin);
      cart->std = (float)f8;
    }
    // global regression weight
    for (i = 0; i < JDA_LBF_N; i++) {
      for (j = 0; j < JDA_LANDMARK_DIM; j++) {
        fread(&f8, sizeof(double), 1, fin);
        cascador->ws[t][i][j] = (float)f8;
      }
    }
  }
  fread(&i4, sizeof(int), 1, fin);
  fclose(fin);
  // set final score threshold, this can be changed
  cascador->th = 0;
  return (void*)cascador;
}

void *jdaCascadorCreateFloat(const char *model) {
  FILE *fin = fopen(model, "rb");
  if (!fin) return NULL;
  jdaCascador *cascador = (jdaCascador*)malloc(sizeof(jdaCascador));
  if (!cascador) {
    fclose(fin);
    return NULL;
  }

  int i4;
  float f4;
  int t, k, i, j;
  // meta
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  fread(&i4, sizeof(int), 1, fin);
  // mean shape
  for (i = 0; i < JDA_LANDMARK_DIM; i++) {
    fread(&f4, sizeof(float), 1, fin);
    cascador->mean_shape[i] = f4;
  }
  // carts
  for (t = 0; t < JDA_T; t++) {
    for (k = 0; k < JDA_K; k++) {
      jdaCart *cart = &cascador->carts[t*JDA_K + k];
      // feature
      for (i = 0; i < JDA_TREE_NODE_N; i++) {
        jdaNode *node = &cart->nodes[i];
        fread(&i4, sizeof(int), 1, fin);
        node->scale = i4;
        fread(&i4, sizeof(int), 1, fin);
        node->landmark1_x = i4 << 1;
        fread(&i4, sizeof(int), 1, fin);
        node->landmark2_x = i4 << 1;
        fread(&f4, sizeof(float), 1, fin);
        node->landmark1_offset_x = f4;
        fread(&f4, sizeof(float), 1, fin);
        node->landmark1_offset_y = f4;
        fread(&f4, sizeof(float), 1, fin);
        node->landmark2_offset_x = f4;
        fread(&f4, sizeof(float), 1, fin);
        node->landmark2_offset_y = f4;
        fread(&i4, sizeof(int), 1, fin);
        node->th = i4;
      }
      // scores
      for (i = 0; i < JDA_TREE_LEAF_N; i++) {
        fread(&f4, sizeof(float), 1, fin);
        cart->score[i] = f4;
      }
      // classificatio threshold
      fread(&f4, sizeof(float), 1, fin);
      cart->th = f4;
      fread(&f4, sizeof(float), 1, fin);
      cart->mean = f4;
      fread(&f4, sizeof(float), 1, fin);
      cart->std = f4;
    }
    // global regression weight
    for (i = 0; i < JDA_LBF_N; i++) {
      for (j = 0; j < JDA_LANDMARK_DIM; j++) {
        fread(&f4, sizeof(float), 1, fin);
        cascador->ws[t][i][j] = f4;
      }
    }
  }
  fread(&i4, sizeof(int), 1, fin);
  fclose(fin);
  // set final score threshold, this can be changed
  cascador->th = 0;
  return (void*)cascador;
}

/*!
 * \brief serialize model to a binary file
 * \note  this function serialze float data type, can reduce model size
 */
void jdaCascadorSerializeTo(void *cascador_, const char *model) {
  FILE *fout = fopen(model, "wb");
  if (!fout) return;
  jdaCascador *cascador = (jdaCascador*)cascador_;
  int i4;
  float f4;
  int t, k, i, j;
  // meta
  i4 = 0; // mask
  fwrite(&i4, sizeof(int), 1, fout);
  i4 = JDA_T;
  fwrite(&i4, sizeof(int), 1, fout);
  i4 = JDA_K;
  fwrite(&i4, sizeof(int), 1, fout);
  i4 = JDA_LANDMARK_N;
  fwrite(&i4, sizeof(int), 1, fout);
  i4 = JDA_TREE_DEPTH;
  fwrite(&i4, sizeof(int), 1, fout);
  i4 = JDA_T + 1;
  fwrite(&i4, sizeof(int), 1, fout);
  i4 = -1;
  fwrite(&i4, sizeof(int), 1, fout);
  // mean shape
  fwrite(cascador->mean_shape, sizeof(float), JDA_LANDMARK_DIM, fout);
  // carts
  for (t = 0; t < JDA_T; t++) {
    for (k = 0; k < JDA_K; k++) {
      jdaCart *cart = &cascador->carts[t*JDA_K + k];
      // feature
      for (i = 0; i < JDA_TREE_NODE_N; i++) {
        jdaNode *node = &cart->nodes[i];
        i4 = node->scale;
        fwrite(&i4, sizeof(int), 1, fout);
        i4 = node->landmark1_x >> 1;
        fwrite(&i4, sizeof(int), 1, fout);
        i4 = node->landmark2_x >> 1;
        fwrite(&i4, sizeof(int), 1, fout);
        f4 = node->landmark1_offset_x;
        fwrite(&f4, sizeof(float), 1, fout);
        f4 = node->landmark1_offset_y;
        fwrite(&f4, sizeof(float), 1, fout);
        f4 = node->landmark2_offset_x;
        fwrite(&f4, sizeof(float), 1, fout);
        f4 = node->landmark2_offset_y;
        fwrite(&f4, sizeof(float), 1, fout);
        i4 = node->th;
        fwrite(&i4, sizeof(int), 1, fout);
      }
      // scores
      for (i = 0; i < JDA_TREE_LEAF_N; i++) {
        f4 = cart->score[i];
        fwrite(&f4, sizeof(float), 1, fout);
      }
      // classificatio threshold
      f4 = cart->th;
      fwrite(&f4, sizeof(float), 1, fout);
      f4 = cart->mean;
      fwrite(&f4, sizeof(float), 1, fout);
      f4 = cart->std;
      fwrite(&f4, sizeof(float), 1, fout);
    }
    // global regression weight
    for (i = 0; i < JDA_LBF_N; i++) {
      for (j = 0; j < JDA_LANDMARK_DIM; j++) {
        f4 = cascador->ws[t][i][j];
        fwrite(&f4, sizeof(float), 1, fout);
      }
    }
  }
  i4 = 0; // mask
  fwrite(&i4, sizeof(int), 1, fout);
  fclose(fout);
}

void jdaCascadorRelease(void *cascador) {
  if (cascador) free((jdaCascador*)cascador);
}

void jdaResultRelease(jdaResult result) {
  // free vector's internal buff
  if (result.bboxes) free(result.bboxes);
  if (result.shapes) free(result.shapes);
  if (result.scores) free(result.scores);
}
