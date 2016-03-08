#include <cstdio>
#include <cstdlib>
#include <liblinear/linear.h>
#include <jda/common.hpp>
#include "jda/postf.hpp"

using namespace cv;
using namespace std;

namespace jda {

/*!
 * \breif Fully Free Model from liblinear
 */
static inline void freeModel(struct model* model) {
  free(model->w);
  free(model->label);
  free(model);
}

PostFilter::PostFilter()
  : sift() {
}

Mat_<double> PostFilter::SiftFeature(const Mat& img, const Mat_<double>& shape) const {
  const int n = shape.cols / 2;
  const double sift_region_size = 2;
  vector<KeyPoint> kps;
  kps.reserve(n);
  for (int i = 0; i < n; i++) {
    kps.push_back(KeyPoint(shape(0, 2 * i), shape(0, 2 * i + 1), sift_region_size));
  }

  Mat_<double> feature;
  sift(img, Mat(), kps, feature, true);
  return feature;
}

Mat_<double> PostFilter::LbpFeature(const Mat& img, const Mat_<double>& shape) const {
  // not implement
  throw Exception();
}

void PostFilter::Train(const vector<Mat>& imgs_p, const vector<Mat_<double> >& shapes_p, \
                       const vector<Mat>& imgs_n, const vector<Mat_<double> >& shapes_n) {
  const Config& c = Config::GetInstance();
  // load cascador
  FILE* fin = fopen("../model/jda.model", "rb");
  JDA_Assert(fin != NULL, "Can not open ../model/jda.model");
  cascador_.SerializeFrom(fin);
  fclose(fin);
  // sift
  LOG("Extract Sift for Positive DataSet");
  const int pos_n = imgs_p.size();
  const int neg_n = imgs_n.size();
  const int m = SiftFeature(imgs_p[0], shapes_p[0]).cols;
  Mat_<double> pos_f(pos_n, m);
  Mat_<double> neg_f(neg_n, m);

  #pragma omp parallel for
  for (int i = 0; i < pos_n; i++) {
    Mat_<double> f = SiftFeature(imgs_p[i], shapes_p[i]);
    double* ptr = pos_f.ptr<double>(i);
    for (int j = 0; j < m; j++) {
      ptr[j] = f(0, j);
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < neg_n; i++) {
    Mat_<double> f = SiftFeature(imgs_n[i], shapes_n[i]);
    double* ptr = neg_f.ptr<double>(i);
    for (int j = 0; j < m; j++) {
      ptr[j] = f(0, j);
    }
  }

  // svm
  const int n = pos_n + neg_n;
  struct feature_node** X = (struct feature_node**)malloc(n*sizeof(struct feature_node*));
  double* Y = (double*)malloc(n*sizeof(double));
  for (int i = 0; i < pos_n; i++) {
    X[i] = (struct feature_node*)malloc((m + 1)*sizeof(struct feature_node));
    double* ptr = pos_f.ptr<double>(i);
    for (int j = 0; j < m; j++) {
      X[i][j].index = j + 1;
      X[i][j].value = ptr[j];
    }
    X[i][m].index = -1;
    X[i][m].value = -1.;
    Y[i] = 1.;
  }
  for (int i = 0; i < neg_n; i++) {
    const int index = i + pos_n;
    X[index] = (struct feature_node*)malloc((m + 1)*sizeof(struct feature_node));
    double* ptr = neg_f.ptr<double>(i);
    for (int j = 0; j < m; j++) {
      X[index][j].index = j + 1;
      X[index][j].value = ptr[j];
    }
    X[index][m].index = -1;
    X[index][m].value = -1.;
    Y[index] = -1.;
  }

  struct problem prob;
  struct parameter param;
  prob.l = n;
  prob.n = m;
  prob.x = X;
  prob.y = Y;
  prob.bias = -1;
  param.solver_type = L2R_L2LOSS_SVC_DUAL;
  param.C = 1. / n;
  param.p = 0;
  param.eps = 0.0001;

  check_parameter(&prob, &param);
  struct model *model = train(&prob, &param);
  w.resize(m);
  for (int j = 0; j < m; j++) w[j] = get_decfun_coef(model, j + 1, 0);
  freeModel(model);

  // release
  for (int i = 0; i < n; i++) {
    free(X[i]);
  }
  free(X);
  free(Y);
}

vector<int> PostFilter::Filter(const vector<Mat>& imgs, const vector<Mat_<double> >& shapes) const {
  const int n = imgs.size();
  const int m = w.size();
  JDA_Assert(m > 0, "PostFilter not trained yet");
  vector<int> res(n, 0);

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    Mat_<double> f = SiftFeature(imgs[i], shapes[i]);
    double* f_ptr = f.ptr<double>(0);
    double y = 0;
    for (int j = 0; j = m; j++) {
      y += w[j] * f_ptr[j];
    }
    if (y > 0) res[i] = 1;
    else res[i] = 0;
  }

  return res;
}

} // namespace jda
