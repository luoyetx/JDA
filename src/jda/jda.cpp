#include <cstdio>
#include "jda/jda.hpp"

using namespace cv;
using namespace std;

namespace jda {

// Meta parameters of JDA model, you may change it under your own condition
/*! \breif number of stages */
static const int T = 5;
/*! \breif number of carts per stage */
static const int K = 1080;
/*! \breif depth of a cart */
static const int tree_depth = 4;
/*! \breif number of landmarks */
static const int landmark_n = 5;
/*! \breif original image width */
static const int o_w = 80;
/*! \breif original image height */
static const int o_h = 80;
/*! \breif half image width */
static const int h_w = 56;
/*! \breif half image height */
static const int h_h = 56;
/*! \breif quarter image width */
static const int q_w = 40;
/*! \breif quarter image height */
static const int q_h = 40;
// parameters below are determined by parameters above
/*! \breif total nodes in a cart */
static const int node_n = 1 << tree_depth;
/*! \breif leaf nodes in a cart */
static const int leaf_n = 1 << (tree_depth - 1);
/*! \breif non-leaf nodes in a cart */
static const int non_leaf_n = leaf_n;

struct jdaCascador::jdaCart {
  int scales[non_leaf_n];
  int landmark_id1[non_leaf_n];
  int landmark_id2[non_leaf_n];
  // all offset will be absolute after the model is loaded
  double offset1_x[non_leaf_n];
  double offset1_y[non_leaf_n];
  double offset2_x[non_leaf_n];
  double offset2_y[non_leaf_n];
  double thresholds[non_leaf_n];
  double scores[leaf_n];
  double th;
};

bool jdaCascador::SerializeFrom(FILE* fd) {
  int YO;
  int tmp;
  fread(&YO, sizeof(YO), 1, fd);
  fread(&tmp, sizeof(int), 1, fd);
  if (tmp != T) return false;
  fread(&tmp, sizeof(int), 1, fd);
  if (tmp != K) return false;
  fread(&tmp, sizeof(int), 1, fd);
  if (tmp != landmark_n) return false;
  fread(&tmp, sizeof(int), 1, fd);
  if (tmp != tree_depth) return false;

  mean_shape.create(1, 2 * landmark_n);
  fread(mean_shape.ptr<double>(0), sizeof(double), mean_shape.cols, fd);

  carts.resize(T*K);
  ws.resize(T);
  const int w_rows = 2 * landmark_n;
  const int w_cols = leaf_n*K;
  for (int i = 0; i < T; i++) {
    for (int j = 0; j < K; j++) {
      jdaCart& cart = carts[i*K + j];
      // non-leaf nodes hold the feature, index start from 1, 0 will never be used
      for (int q = 1; q < non_leaf_n; q++) {
        fread(&cart.scales[q], sizeof(int), 1, fd);
        fread(&cart.landmark_id1[q], sizeof(int), 1, fd);
        fread(&cart.landmark_id2[q], sizeof(int), 1, fd);
        fread(&cart.offset1_x[q], sizeof(double), 1, fd);
        fread(&cart.offset1_y[q], sizeof(double), 1, fd);
        fread(&cart.offset2_x[q], sizeof(double), 1, fd);
        fread(&cart.offset2_y[q], sizeof(double), 1, fd);
        fread(&cart.thresholds[q], sizeof(int), 1, fd);
        // absolute offset
        cart.offset1_x[q] *= o_w; cart.offset1_y[q] *= o_h;
        cart.offset2_x[q] *= o_w; cart.offset2_y[q] *= o_h;
      }
      // leaf nodes hold the scores
      for (int q = 0; q < leaf_n; q++) {
        fread(&cart.scores[q], sizeof(double), 1, fd);
      }
      // threshold
      fread(&cart.th, sizeof(double), 1, fd);
    }
    // regression weight
    ws[i].create(w_rows, w_cols);
    for (int j = 0; j < w_rows; j++) {
      fread(ws[i].ptr<double>(j), sizeof(double), w_cols, fd);
    }
  }

  fread(&YO, sizeof(YO), 1, fd);
}

int jdaCascador::Detect(Mat& img, vector<Rect>& rects, vector<double>& scores, \
                        vector<Mat_<double> >& shapes) const {
  // **TODO** Detection
  return 0;
}

} // namespace jda
