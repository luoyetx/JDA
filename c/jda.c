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

// jda global constance

#define JDA_T             5
#define JDA_K             290
#define JDA_LANDMARK_N    29
#define JDA_TREE_DEPTH    4
#define JDA_TREE_LEAF_N   (1 << (JDA_TREE_DEPTH - 1))
#define JDA_TREE_NODE_N   (JDA_TREE_LEAF_N - 1)
#define JDA_CART_N        (JDA_T*JDA_K)
#define JDA_LANDMARK_DIM  (2 * JDA_LANDMARK_N)
#define JDA_LBF_N         (JDA_K*JDA_TREE_LEAF_N)

/*!
 * \breif A marco based generic vector type for C
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

#define JDA_VECTOR(Type) \
  struct jdaVector##Type { \
    int size; \
    int capacity; \
    Type *data; \
  }

#define JDA_VECTOR_DEC(Type) \
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

JDA_VECTOR(int);
JDA_VECTOR(float);

// data structures

/*! \breif jda bbox */
typedef struct {
  int x, y, size;
} jdaBBox;

/*! \breif jda shape */
typedef float jdaShape[JDA_LANDMARK_DIM];

/*!\breif jda cart node */
typedef struct {
  int scale;
  int landmark1_x;
  int landmark2_x;
  float landmark1_offset_x;
  float landmark1_offset_y;
  float landmark2_offset_x;
  float landmark2_offset_y;
  int th;
} jdaNode;

/*! \breif jda cart */
typedef struct {
  jdaNode nodes[JDA_TREE_NODE_N];
  float score[JDA_TREE_LEAF_N];
  float th;
} jdaCart;

/*! \breif jda cascador */
typedef struct {
  jdaCart carts[JDA_CART_N];
  float ws[JDA_T][JDA_LBF_N][JDA_LANDMARK_DIM];
  float mean_shape[JDA_LANDMARK_DIM];
} jdaCascador;

/*! \breif jda image */
typedef struct {
  int w;
  int h;
  int step;
  unsigned char *data;
} jdaImage;

// Internal Helpers

#define JDA_IMAGE_AT(img, x, y) ((img)->data[(y)*(img)->step+(x)])

#define JDA_MAX(x, y) (((x)<(y))?(y):(x))
#define JDA_MIN(x, y) (((x)<(y))?(x):(y))

/*!
 * \breif create image
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
 * \breif release internal data buffer
 * \note  don't release the image which borrow from others
 *
 * \param img   image to free
 */
static inline
void jdaImageRelease(jdaImage *img) {
  if (img->data) free(img->data);
}

/*!
 * \breif resize image, bilinear interpolation
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
 * \breif nms
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
  JDA_VECTOR_DEC(int) merged_bboxes;
  JDA_VECTOR_DEC(float) merged_shapes;
  JDA_VECTOR_DEC(float) merged_scores;
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
  merged.total_patches = result.total_patches;

  free(flag);
  free(area);
  free(idx);
  // free previous result
  free(result.scores);
  free(result.shapes);
  free(result.bboxes);
  return merged;
}

static jdaResult jdaInternalDetect(jdaCascador *cascador, jdaImage o, jdaImage h, jdaImage q) {
  int mini_size = 20;
  float factor = 1.2f;
  float r = 1.f / sqrtf(2.f);
  int counter = 0;

  JDA_VECTOR_DEC(int) bboxes;
  JDA_VECTOR_DEC(float) shapes;
  JDA_VECTOR_DEC(float) scores;
  JDA_VECTOR_NEW(scores, JDA_VECTOR_DEFAULT_LEN);
  JDA_VECTOR_NEW(bboxes, JDA_VECTOR_DEFAULT_LEN * 3);
  JDA_VECTOR_NEW(shapes, JDA_VECTOR_DEFAULT_LEN * JDA_LANDMARK_DIM);

  int win_max_size = ((o.h < o.w) ? o.h : o.w);
  int win_size;
  for (win_size = mini_size; win_size <= win_max_size; win_size = (int)(win_size*factor)) {
    int step = (int)(win_size*0.1f);
    int x_max = o.w - win_size;
    int y_max = o.h - win_size;
    int win_h_size = (int)(win_size*r);
    int win_q_size = win_size / 2;

    int x, y;
    for (y = 0; y <= y_max; y += step) {
      for (x = 0; x <= x_max; x += step) {
        counter++;

        jdaImage ps[3];
        ps[0].w = ps[0].h = win_size;
        ps[0].step = o.step;
        ps[0].data = &o.data[y*o.step + x]; // borrow memory
        int h_x = (int)(x*r);
        int h_y = (int)(y*r);
        ps[1].w = ps[1].h = win_h_size;
        ps[1].step = h.step;
        ps[1].data = &h.data[h_y*h.step + h_x]; // borrow memory
        int q_x = x / 2;
        int q_y = y / 2;
        ps[2].w = ps[2].h = win_q_size;
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
            // not a face
            if (score <= cart->th) goto next;
            lbf[k] = k*JDA_TREE_LEAF_N + leaf_idx;
            cart++;
          }
          // regression
          jdaShape *ws = cascador->ws[t];
          for (k = 0; k < JDA_K; k++) {
            float *w = ws[lbf[k]];
            for (i = 0; i < JDA_LANDMARK_DIM; i++) {
              shape[i] += w[i];
            }
          }
        }
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
  result.total_patches = counter;
  return result;
}

// APIs

jdaResult jdaDetect(void *cascador, unsigned char *data, int width, int height) {
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

  jdaResult result = jdaInternalDetect((jdaCascador*)cascador, o, h, q);
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
 * \breif serialize model from JDA
 * \note  JDA dump data type is double
 */
void *jdaCascadorCreate(const char *model) {
  FILE *fin = fopen(model, "rb");
  if (!fin) return NULL;
  jdaCascador *cascador = (jdaCascador*)malloc(sizeof(jdaCascador));
  if (!cascador) {
    fclose(fin);
    return NULL;
  }

  int t4;
  double t8;
  int t, k, i, j;
  // meta
  fread(&t4, sizeof(int), 1, fin);
  fread(&t4, sizeof(int), 1, fin);
  fread(&t4, sizeof(int), 1, fin);
  fread(&t4, sizeof(int), 1, fin);
  fread(&t4, sizeof(int), 1, fin);
  fread(&t4, sizeof(int), 1, fin);
  fread(&t4, sizeof(int), 1, fin);
  // mean shape
  for (i = 0; i < JDA_LANDMARK_DIM; i++) {
    fread(&t8, sizeof(double), 1, fin);
    cascador->mean_shape[i] = (float)t8;
  }
  // carts
  for (t = 0; t < JDA_T; t++) {
    for (k = 0; k < JDA_K; k++) {
      jdaCart *cart = &cascador->carts[t*JDA_K + k];
      // feature
      for (i = 0; i < JDA_TREE_NODE_N; i++) {
        jdaNode *node = &cart->nodes[i];
        fread(&t4, sizeof(int), 1, fin);
        node->scale = t4;
        fread(&t4, sizeof(int), 1, fin);
        node->landmark1_x = t4 << 1;
        fread(&t4, sizeof(int), 1, fin);
        node->landmark2_x = t4 << 1;
        fread(&t8, sizeof(double), 1, fin);
        node->landmark1_offset_x = (float)t8;
        fread(&t8, sizeof(double), 1, fin);
        node->landmark1_offset_y = (float)t8;
        fread(&t8, sizeof(double), 1, fin);
        node->landmark2_offset_x = (float)t8;
        fread(&t8, sizeof(double), 1, fin);
        node->landmark2_offset_y = (float)t8;
        fread(&t4, sizeof(int), 1, fin);
        node->th = t4;
      }
      // scores
      for (i = 0; i < JDA_TREE_LEAF_N; i++) {
        fread(&t8, sizeof(double), 1, fin);
        cart->score[i] = (float)t8;
      }
      // classificatio threshold
      fread(&t8, sizeof(double), 1, fin);
      cart->th = (float)t8;
    }
    // global regression weight
    for (i = 0; i < JDA_LBF_N; i++) {
      for (j = 0; j < JDA_LANDMARK_DIM; j++) {
        fread(&t8, sizeof(double), 1, fin);
        cascador->ws[t][i][j] = (float)t8;
      }
    }
  }
  fread(&t4, sizeof(int), 1, fin);
  fclose(fin);
  return (void*)cascador;
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
