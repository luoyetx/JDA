#include <omp.h>
#include <cmath>
#include <cstdio>
#include <climits>
#include <limits>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/data.hpp"
#include "jda/cart.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;

namespace jda {

/*! \breif is zeros */
static inline bool isZero(double num) {
  if (std::abs(num) < 1e-9) return true;
  else return false;
}

Cart::Cart(int stage, int landmark_id) {
  const Config& c = Config::GetInstance();
  this->stage = stage;
  this->landmark_id = landmark_id;
  depth = c.tree_depth;
  leafNum = 1 << (depth - 1);
  nodes_n = 1 << depth;
  featNum = c.feats[stage];
  radius = c.radius[stage];
  features.resize(nodes_n / 2); // all 0
  thresholds.resize(nodes_n / 2); // all 0
  scores.resize(nodes_n / 2); // all 0
}
Cart::~Cart() {
}

void Cart::Train(const DataSet& pos, const DataSet& neg) {
  vector<int> pos_idx, neg_idx;
  int pos_n = pos.size;
  int neg_n = neg.size;
  pos_idx.resize(pos_n);
  neg_idx.resize(neg_n);

  #pragma omp parallel for
  for (int i = 0; i < pos_n; i++) pos_idx[i] = i;
  #pragma omp parallel for
  for (int i = 0; i < neg_n; i++) neg_idx[i] = i;

  // split node from root with idx = 1, why 1? see binary tree in sequence
  SplitNode(pos, pos_idx, neg, neg_idx, 1);
}

void Cart::SplitNode(const DataSet& pos, const vector<int>& pos_idx, \
                     const DataSet& neg, const vector<int>& neg_idx, \
                     int node_idx) {
  Config& c = Config::GetInstance();
  const int pos_n = pos_idx.size();
  const int neg_n = neg_idx.size();
  if (node_idx >= nodes_n / 2) {
    // we are on a leaf node
    const int idx = node_idx - nodes_n / 2;
    double pos_w, neg_w;
    pos_w = neg_w = c.esp;

    #pragma omp parallel sections
    {
      #pragma omp section
      {
        for (int i = 0; i < pos_n; i++) {
          pos_w += pos.weights[pos_idx[i]];
        }
      }
      #pragma omp section
      {
        for (int i = 0; i < neg_n; i++) {
          neg_w += neg.weights[neg_idx[i]];
        }
      }
    }

    scores[idx] = 0.5*(log(pos_w) - log(neg_w));
    printf("Leaf % 3d has % 7d pos and % 7d neg. Score is %.4lf\n", \
           node_idx, pos_n, neg_n, scores[idx]);
    return;
  }

  printf("Node % 3d has % 7d pos and % 7d neg.", node_idx, pos_n, neg_n);

  // feature pool
  vector<Feature> feature_pool;
  Mat_<int> pos_feature, neg_feature;
  GenFeaturePool(feature_pool);
  pos_feature = pos.CalcFeatureValues(feature_pool, pos_idx);
  neg_feature = neg.CalcFeatureValues(feature_pool, neg_idx);
  // classification or regression
  RNG rng(cv::getTickCount());
  bool is_classification = (rng.uniform(0., 1.) < c.probs[stage]) ? true : false;
  int feature_idx, threshold;
  if (is_classification) {
    printf(" Split by Classification\n");
    SplitNodeWithClassification(pos, pos_idx, neg, neg_idx, \
                                pos_feature, neg_feature, \
                                feature_idx, threshold);
  }
  else {
    printf(" Split by Regression\n");
    Mat_<double> shape_residual = pos.CalcShapeResidual(pos_idx, landmark_id);
    SplitNodeWithRegression(pos, pos_idx, neg, neg_idx, \
                            pos_feature, shape_residual, \
                            feature_idx, threshold);
  }
  // split training data into left and right if any more
  vector<int> left_pos_idx, left_neg_idx;
  vector<int> right_pos_idx, right_neg_idx;

  #pragma omp parallel sections
  {
    // pos
    #pragma omp section
    {
      left_pos_idx.reserve(pos_n);
      right_pos_idx.reserve(pos_n);
      for (int i = 0; i < pos_n; i++) {
        if (pos_feature(feature_idx, i) <= threshold) {
          left_pos_idx.push_back(pos_idx[i]);
        }
        else {
          right_pos_idx.push_back(pos_idx[i]);
        }
      }
    }
    // neg
    #pragma omp section
    {
      left_neg_idx.reserve(neg_n);
      right_neg_idx.reserve(neg_n);
      for (int i = 0; i < neg_n; i++) {
        if (neg_feature(feature_idx, i) <= threshold) {
          left_neg_idx.push_back(neg_idx[i]);
        }
        else {
          right_neg_idx.push_back(neg_idx[i]);
        }
      }
    }
  }

  // save parameters on this node
  features[node_idx] = feature_pool[feature_idx];
  thresholds[node_idx] = threshold;
  // manually release to reduce memory usage
  feature_pool.clear();
  pos_feature.release();
  neg_feature.release();
  // split node in DFS way
  SplitNode(pos, left_pos_idx, neg, left_neg_idx, 2 * node_idx);
  SplitNode(pos, right_pos_idx, neg, right_neg_idx, 2 * node_idx + 1);
}

/*!
 * \breif Calculate Entropy
 * \param p   p
 * \return    entropy
 */
static inline double calcEntropy(double p) {
  if (isZero(p) || isZero(1. - p)) return 0;
  double entropy = -(p)*std::log(p) - (1. - p)*std::log(1. - p);
  entropy /= std::log(2.);
  return entropy;
}

void Cart::SplitNodeWithClassification(const DataSet& pos, const vector<int>& pos_idx, \
                                       const DataSet& neg, const vector<int>& neg_idx, \
                                       const Mat_<int>& pos_feature, \
                                       const Mat_<int>& neg_feature, \
                                       int& feature_idx, int& threshold) {
  const Config& c = Config::GetInstance();
  const int feature_n = pos_feature.rows;
  const int pos_n = pos_feature.cols;
  const int neg_n = neg_feature.cols;
  const int total_n = pos_n + neg_n;
  feature_idx = 0;
  threshold = -256; // all data will go to right child tree

  // select a feature that has minimum entropy
  vector<double> es_(feature_n);
  vector<int> ths_(feature_n);

  #pragma omp parallel for
  for (int i = 0; i < feature_n; i++) {
    double wp_l, wp_r, wn_l, wn_r;
    wp_l = wp_r = wn_l = wn_r = 0;
    vector<double> wp(511, 0), wn(511, 0);
    vector<int> p_n(511, 0), n_n(511, 0);
    for (int j = 0; j < pos_n; j++) {
      wp[pos_feature(i, j) + 255] += pos.weights[pos_idx[j]];
      wp_r += pos.weights[pos_idx[j]];
      p_n[pos_feature(i, j) + 255]++;
    }
    for (int j = 0; j < neg_n; j++) {
      wn[neg_feature(i, j) + 255] += neg.weights[neg_idx[j]];
      wn_r += neg.weights[neg_idx[j]];
      n_n[neg_feature(i, j) + 255]++;
    }

    int current_p = 0;
    int current_n = 0;
    double w = wp_r + wn_r;

    int threshold_ = -256;
    double entropy = calcEntropy(wp_r / w);
    for (int th = -255; th <= 255; th++) {
      const int idx = th + 255;
      wp_l += wp[idx];
      wn_l += wn[idx];
      wp_r -= wp[idx];
      wn_r -= wn[idx];
      current_p += p_n[idx];
      current_n += n_n[idx];

      const double p_ratio = double(current_p) / pos_n;
      const double n_ratio = double(current_n) / neg_n;
      if (p_ratio < 0.1 || p_ratio > 0.9) continue;
      if (n_ratio < 0.1 || n_ratio > 0.9) continue;

      double w_l = wp_l + wn_l;
      double w_r = wp_r + wn_r;
      double e = (w_l / w)*calcEntropy(wp_l / w_l) + \
                 (w_r / w)*calcEntropy(wp_r / w_r);
      if (e < entropy) {
        entropy = e;
        threshold_ = th;
      }
    }
    es_[i] = entropy;
    ths_[i] = threshold_;
  }

  double entropy_min = numeric_limits<double>::max();
  for (int i = 0; i < feature_n; i++) {
    if (es_[i] < entropy_min) {
      entropy_min = es_[i];
      threshold = ths_[i];
      feature_idx = i;
    }
  }
  // Done
}

/*!
 * \breif Calculate Variance of vector
 */
static double calcVariance(const vector<double>& vec) {
  if (vec.size() == 0) return 0.;
  Mat_<double> vec_(vec);
  double m1 = cv::mean(vec_)[0];
  double m2 = cv::mean(vec_.mul(vec_))[0];
  double variance = m2 - m1*m1;
  return variance;
}

template<typename T>
static void _qsort(T* a, int l, int r) {
  int i, j;
  T t, tmp;
  i = l; j = r;
  t = a[(i + j) / 2];
  do {
    while (a[i] < t) i++;
    while (a[j] > t) j--;
    if (i <= j) {
      tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
      i++; j--;
    }
  } while (i <= j);
  if (l < j) _qsort(a, l, j);
  if (i < r) _qsort(a, i, r);
}

void Cart::SplitNodeWithRegression(const DataSet& pos, const vector<int>& pos_idx, \
                                   const DataSet& neg, const vector<int>& neg_idx, \
                                   const Mat_<int>& pos_feature, \
                                   const Mat_<double>& shape_residual, \
                                   int& feature_idx, int& threshold) {
  Config& c = Config::GetInstance();
  const int feature_n = pos_feature.rows;
  const int pos_n = pos_feature.cols;
  feature_idx = 0;
  threshold = -256; // all data will go to right child tree

  if (pos_n == 0) {
    return;
  }

  //Mat_<int> pos_feature_sorted;
  //cv::sort(pos_feature, pos_feature_sorted, SORT_EVERY_ROW + SORT_ASCENDING);

  // select a feature reduce maximum variance
  vector<double> vs_(feature_n);
  vector<int> ths_(feature_n);

  #pragma omp parallel for
  for (int i = 0; i < feature_n; i++) {
    RNG& rng = c.rng_pool[omp_get_thread_num() + 1];

    Mat_<int> pos_feature_sorted = pos_feature.row(i).clone();
    _qsort<int>(pos_feature_sorted.ptr<int>(0), 0, pos_n - 1);

    vector<double> left_x, left_y, right_x, right_y;
    left_x.reserve(pos_n); left_y.reserve(pos_n);
    right_x.reserve(pos_n); right_y.reserve(pos_n);
    int threshold_ = pos_feature_sorted(0, int(pos_n*rng.uniform(0.1, 0.9)));
    for (int j = 0; j < pos_n; j++) {
      if (pos_feature(i, j) <= threshold_) {
        left_x.push_back(shape_residual(j, 0));
        left_y.push_back(shape_residual(j, 1));
      }
      else {
        right_x.push_back(shape_residual(j, 0));
        right_y.push_back(shape_residual(j, 1));
      }
    }
    double variance_ = (calcVariance(left_x) + calcVariance(left_y))*left_x.size() + \
                       (calcVariance(right_x) + calcVariance(right_y))*right_x.size();
    vs_[i] = variance_;
    ths_[i] = threshold_;
  }

  double variance_min = std::numeric_limits<double>::max();
  for (int i = 0; i < feature_n; i++) {
    if (vs_[i] < variance_min) {
      variance_min = vs_[i];
      threshold = ths_[i];
      feature_idx = i;
    }
  }
  // Done
}

void Cart::GenFeaturePool(vector<Feature>& feature_pool) {
  Config& c = Config::GetInstance();
  const int landmark_n = c.landmark_n;
  feature_pool.resize(featNum);

  #pragma omp parallel for
  for (int i = 0; i < featNum; i++) {
    RNG& rng = c.rng_pool[omp_get_thread_num() + 1];

    double x1, y1, x2, y2;
    x1 = y1 = x2 = y2 = 1.;
    // needs to be in a circle
    while (x1*x1 + y1*y1 > 1. || x2*x2 + y2*y2 > 1.) {
      x1 = rng.uniform(-1., 1.); y1 = rng.uniform(-1., 1.);
      x2 = rng.uniform(-1., 1.); y2 = rng.uniform(-1., 1.);
    }
    Feature& feat = feature_pool[i];
    switch (rng.uniform(0, 3)) {
    case 0:
      feat.scale = Feature::ORIGIN; break;
    case 1:
      feat.scale = Feature::HALF; break;
    case 2:
      feat.scale = Feature::QUARTER; break;
    default:
      feat.scale = Feature::ORIGIN; break;
    }

    // may be no multi scale
    if (!c.multi_scale) feat.scale = Feature::ORIGIN;

    feat.landmark_id1 = rng.uniform(0, landmark_n);
    feat.landmark_id2 = rng.uniform(0, landmark_n);
    feat.offset1_x = x1*radius;
    feat.offset1_y = y1*radius;
    feat.offset2_x = x2*radius;
    feat.offset2_y = y2*radius;
  }
}

int Cart::Forward(const Mat& img, const Mat& img_h, const Mat& img_q, \
                  const Mat_<double>& shape) const {
  int node_idx = 1;
  int len = depth - 1;
  while (len--) {
    const Feature& feature = features[node_idx];
    //int val = feature.CalcFeatureValue(img, img_h, img_q, shape);

    const Mat* img_ptr;
    switch (feature.scale) {
    case Feature::ORIGIN:
      img_ptr = &img; // ref
      break;
    case Feature::HALF:
      img_ptr = &img_h; // ref
      break;
    case Feature::QUARTER:
      img_ptr = &img_q; // ref
      break;
    default:
      dieWithMsg("Unsupported SCALE");
      break;
    }

    double x1, y1, x2, y2;
    const int width = img_ptr->cols;
    const int height = img_ptr->rows;
    x1 = (shape(0, 2 * feature.landmark_id1) + feature.offset1_x)*width;
    y1 = (shape(0, 2 * feature.landmark_id1 + 1) + feature.offset1_y)*height;
    x2 = (shape(0, 2 * feature.landmark_id2) + feature.offset2_x)*width;
    y2 = (shape(0, 2 * feature.landmark_id2 + 1) + feature.offset2_y)*height;
    int x1_ = int(round(x1));
    int y1_ = int(round(y1));
    int x2_ = int(round(x2));
    int y2_ = int(round(y2));

    checkBoundaryOfImage(width, height, x1_, y1_);
    checkBoundaryOfImage(width, height, x2_, y2_);

    int val = int(img_ptr->at<uchar>(y1_, x1_)) - int(img_ptr->at<uchar>(y2_, x2_));
    if (val <= thresholds[node_idx]) node_idx = 2 * node_idx;
    else node_idx = 2 * node_idx + 1;
  }
  const int bias = 1 << (depth - 1);
  return node_idx - bias;
}

void Cart::SerializeFrom(FILE* fd) {
  // only non leaf node need to save parameters
  for (int i = 1; i < nodes_n / 2; i++) {
    Feature& feature = features[i];
    fread(&feature.scale, sizeof(int), 1, fd);
    fread(&feature.landmark_id1, sizeof(int), 1, fd);
    fread(&feature.landmark_id2, sizeof(int), 1, fd);
    fread(&feature.offset1_x, sizeof(double), 1, fd);
    fread(&feature.offset1_y, sizeof(double), 1, fd);
    fread(&feature.offset2_x, sizeof(double), 1, fd);
    fread(&feature.offset2_y, sizeof(double), 1, fd);
    fread(&thresholds[i], sizeof(int), 1, fd);
  }
  // leaf node has scores
  for (int i = 0; i < nodes_n / 2; i++) {
    fread(&scores[i], sizeof(double), 1, fd);
  }
  // threshold
  fread(&th, sizeof(double), 1, fd);
}

void Cart::SerializeTo(FILE* fd) const {
  // only non leaf node need to save parameters
  for (int i = 1; i < nodes_n / 2; i++) {
    const Feature& feature = features[i];
    fwrite(&feature.scale, sizeof(int), 1, fd);
    fwrite(&feature.landmark_id1, sizeof(int), 1, fd);
    fwrite(&feature.landmark_id2, sizeof(int), 1, fd);
    fwrite(&feature.offset1_x, sizeof(double), 1, fd);
    fwrite(&feature.offset1_y, sizeof(double), 1, fd);
    fwrite(&feature.offset2_x, sizeof(double), 1, fd);
    fwrite(&feature.offset2_y, sizeof(double), 1, fd);
    fwrite(&thresholds[i], sizeof(int), 1, fd);
  }
  // leaf node has scores
  for (int i = 0; i < nodes_n / 2; i++) {
    fwrite(&scores[i], sizeof(double), 1, fd);
  }
  // threshold
  fwrite(&th, sizeof(double), 1, fd);
}

void Cart::PrintSelf() {
  const Config& c = Config::GetInstance();
  printf("\nSummary of this Cart\n");
  printf("node parameters\n");
  for (int i = 1; i < nodes_n / 2; i++) {
    const Feature& f = features[i];
    const int threshold = thresholds[i];
    printf("  node %d: [scale = %d, th = %d, landmark_1 = (%d, %.4lf, %.4lf), "
           "landmark_2 = (%d, %.4lf, %.4lf)]\n", i, f.scale, threshold, \
           f.landmark_id1 + c.landmark_offset, f.offset1_x, f.offset1_y, \
           f.landmark_id2 + c.landmark_offset, f.offset2_x, f.offset2_y);
  }
  printf("leaf scores\n[");
  for (int i = 0; i < leafNum; i++) {
    printf("%.4lf, ", scores[i]);
  }
  printf("]\n");
  printf("threshold = %.4lf\n\n", th);
}

} // namespace jda
