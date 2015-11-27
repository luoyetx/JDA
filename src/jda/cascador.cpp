#include <ctime>
#include <cstdio>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/data.hpp"
#include "jda/cart.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;

namespace jda {

static int YO = 0;

JoinCascador::JoinCascador() {
  const Config& c = Config::GetInstance();
  T = c.T;
  K = c.K;
  landmark_n = c.landmark_n;
  tree_depth = c.tree_depth;
  current_stage_idx = 0;
  current_cart_idx = -1;
  btcarts.reserve(T);
  for (int t = 0; t < T; t++) {
    btcarts.push_back(BoostCart(t));
  }
}
JoinCascador::~JoinCascador() {
}

void JoinCascador::Train(DataSet& pos, DataSet& neg) {
  this->pos = &pos;
  this->neg = &neg;
  const int start = current_stage_idx;
  for (int t = start; t < T; t++) {
    current_stage_idx = t;
    current_cart_idx = -1;
    LOG("Train %d th stages", t + 1);
    TIMER_BEGIN
      btcarts[t].Train(pos, neg);
      LOG("End of train %d th stages, costs %.4lf s", t + 1, TIMER_NOW);
    TIMER_END
    LOG("Snapshot current Training Status");
    Snapshot();
  }
}

void JoinCascador::Snapshot() {
  int stage_idx = current_stage_idx;
  int cart_idx = current_cart_idx;
  char buff1[256];
  char buff2[256];
  time_t t = time(NULL);
  strftime(buff1, sizeof(buff1), "%Y%m%d-%H%M%S", localtime(&t));
  sprintf(buff2, "../model/jda_tmp_%s_stage_%d_cart_%d.model", \
          buff1, stage_idx + 1, cart_idx + 1);

  FILE* fd = fopen(buff2, "wb");
  JDA_Assert(fd, "Can not open a temp file to save the model");

  SerializeTo(fd);

  fclose(fd);
}

void JoinCascador::Resume(FILE* fd) {
  SerializeFrom(fd);
}

void JoinCascador::SerializeTo(FILE* fd) const {
  fwrite(&YO, sizeof(YO), 1, fd);
  fwrite(&T, sizeof(int), 1, fd); // number of stages
  fwrite(&K, sizeof(int), 1, fd); // number of trees per stage
  fwrite(&landmark_n, sizeof(int), 1, fd); // number of landmarks
  fwrite(&tree_depth, sizeof(int), 1, fd); // tree depth
  // mean shape
  fwrite(mean_shape.ptr<double>(0), sizeof(double), mean_shape.cols, fd);
  // btcarts
  for (int t = 0; t < T; t++) {
    const BoostCart& btcart = btcarts[t];
    for (int k = 0; k < K; k++) {
      const Cart& cart = btcart.carts[k];
      cart.SerializeTo(fd);
    }
    // global regression parameters
    const double* w_ptr;
    const int rows = btcart.w.rows;
    const int cols = btcart.w.cols;
    for (int i = 0; i < rows; i++) {
      w_ptr = btcart.w.ptr<double>(i);
      fwrite(w_ptr, sizeof(double), cols, fd);
    }
  }
  fwrite(&YO, sizeof(YO), 1, fd);
}

void JoinCascador::SerializeFrom(FILE* fd) {
  int tmp;
  fread(&YO, sizeof(YO), 1, fd);
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(tmp == T, "T is wrong!");
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(tmp == K, "K is wrong!");
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(tmp == landmark_n, "landmark_n is wrong!");
  fread(&tmp, sizeof(int), 1, fd);
  JDA_Assert(tmp == tree_depth, "tree_depth is wrong!");

  // mean shape
  mean_shape.create(1, 2 * landmark_n);
  fread(mean_shape.ptr<double>(0), sizeof(double), mean_shape.cols, fd);

  for (int t = 0; t < T; t++) {
    BoostCart& btcart = btcarts[t];
    for (int k = 0; k < K; k++) {
      Cart& cart = btcart.carts[k];
      cart.SerializeFrom(fd);
    }
    // global regression parameters
    double* w_ptr;
    const int w_rows = landmark_n * 2;
    const int w_cols = K * (1 << (tree_depth - 1));
    for (int i = 0; i < w_rows; i++) {
      w_ptr = btcart.w.ptr<double>(i);
      fread(w_ptr, sizeof(double), w_cols, fd);
    }
  }
  fread(&YO, sizeof(YO), 1, fd);
}

bool JoinCascador::Validate(const Mat& img, double& score, Mat_<double>& shape) const {
  const Config& c = Config::GetInstance();
  Mat img_h, img_q;
  cv::resize(img, img_h, Size(c.img_h_width, c.img_h_height));
  cv::resize(img, img_q, Size(c.img_q_width, c.img_q_height));
  DataSet::RandomShape(mean_shape, shape);
  score = 0;
  Mat_<int> lbf(1, c.K);
  int* lbf_ptr = lbf.ptr<int>(0);
  const int base = 1 << (c.tree_depth - 1);
  int offset = 0;
  // stage [0, current_stage_idx)
  for (int t = 0; t < current_stage_idx; t++) {
    const BoostCart& btcart = btcarts[t];
    offset = 0;
    for (int k = 0; k < c.K; k++) {
      const Cart& cart = btcart.carts[k];
      int idx = cart.Forward(img, img_h, img_q, shape);
      score += cart.scores[idx];
      if (score < cart.th) {
        // not a face
        return false;
      }
      lbf_ptr[k] = offset + idx;
      offset += base;
    }
    // global regression
    shape += btcart.GenDeltaShape(lbf);
  }
  // current stage, cart [0, current_cart_idx]
  for (int k = 0; k <= current_cart_idx; k++) {
    const Cart& cart = btcarts[current_stage_idx].carts[k];
    int idx = cart.Forward(img, img_h, img_q, shape);
    score += cart.scores[idx];
    if (score < cart.th) {
      // not a face
      return false;
    }
  }
  return true;
}

/*!
 * \breif detect single scale
 */
static void detectSingleScale(const JoinCascador& joincascador, const Mat& img, \
                              vector<Rect>& rects, vector<double>& scores, \
                              vector<Mat_<double> >& shapes) {
  const Config& c = Config::GetInstance();
  const int win_w = c.img_o_width;
  const int win_h = c.img_o_height;
  const int x_max = img.cols - win_w;
  const int y_max = img.rows - win_h;
  const int x_step = 20;
  const int y_step = 20;
  int x = 0;
  int y = 0;

  rects.clear();
  scores.clear();
  shapes.clear();

  while (y <= y_max) {
    while (x <= x_max) {
      Rect roi(x, y, win_w, win_h);
      double score;
      Mat_<double> shape;
      bool is_face = joincascador.Validate(img(roi), score, shape);
      if (is_face) {
        rects.push_back(roi);
        scores.push_back(score);
        shapes.push_back(shape);
      }
      x += x_step;
    }
    x = 0;
    y += y_step;
  }
}

/*!
 * \breif detect multi scale
 */
static void detectMultiScale(const JoinCascador& joincascador, const Mat& img, \
                             vector<Rect>& rects, vector<double>& scores, \
                             vector<Mat_<double> >& shapes) {
  const Config& c = Config::GetInstance();
  const int win_w = c.img_o_width;
  const int win_h = c.img_o_height;
  int width = img.cols;
  int height = img.rows;
  const double factor = 1.3;
  double scale = 1.;
  Mat img_ = img.clone();

  rects.clear();
  scores.clear();
  shapes.clear();

  while ((width >= win_w) && (height >= win_h)) {
    vector<Rect> rects_;
    vector<double> scores_;
    vector<Mat_<double> > shapes_;
    detectSingleScale(joincascador, img_, rects_, scores_, shapes_);
    const int n = rects_.size();
    for (int i = 0; i < n; i++) {
      Rect& r = rects_[i];
      r.x *= scale; r.y *= scale;
      r.width *= scale; r.height *= scale;
      shapes_[i] *= scale;
    }
    rects.insert(rects.end(), rects_.begin(), rects_.end());
    scores.insert(scores.end(), scores_.begin(), scores_.end());
    shapes.insert(shapes.end(), shapes_.begin(), shapes_.end());

    scale *= factor;
    width = int(width / factor + 0.5);
    height = int(height / factor + 0.5);
    cv::resize(img_, img_, Size(width, height));
  }
}

/*!
 * \breif nms Non-maximum suppression
 * the algorithm is from https://github.com/ShaoqingRen/SPP_net/blob/master/nms%2Fnms_mex.cpp
 *
 * \param rects     area of faces
 * \param scores    score of faces
 * \param overlap   overlap threshold
 * \return          picked index
 */
static vector<int> nms(const vector<Rect>& rects, const vector<double>& scores, \
                       double overlap) {
  const int n = rects.size();
  vector<double> areas(n);

  typedef std::multimap<double, int> ScoreMapper;
  ScoreMapper map;
  for (int i = 0; i < n; i++) {
    map.insert(ScoreMapper::value_type(scores[i], i));
    areas[i] = rects[i].width*rects[i].height;
  }

  int picked_n = 0;
  vector<int> picked(n);
  while (map.size() != 0) {
    int last = map.rbegin()->second; // get the index of maximum score value
    picked[picked_n] = last;
    picked_n++;

    for (ScoreMapper::iterator it = map.begin(); it != map.end();) {
      int idx = it->second;
      double x1 = std::max(rects[idx].x, rects[last].x);
      double y1 = std::max(rects[idx].y, rects[last].y);
      double x2 = std::min(rects[idx].x + rects[idx].width, rects[last].x + rects[last].width);
      double y2 = std::min(rects[idx].y + rects[idx].height, rects[last].y + rects[last].height);
      double w = std::max(0., x2 - x1);
      double h = std::max(0., y2 - y1);
      double ov = w*h / (areas[idx] + areas[last] - w*h);
      if (ov > overlap) {
        ScoreMapper::iterator tmp = it;
        tmp++;
        map.erase(it);
        it = tmp;
      }
      else{
        it++;
      }
    }
  }

  picked.resize(picked_n);
  return picked;
}

int JoinCascador::Detect(const Mat& img, vector<Rect>& rects, vector<double>& scores, \
                         vector<Mat_<double> >& shapes) {
  vector<Rect> rects_;
  vector<double> scores_;
  vector<Mat_<double> > shapes_;
  detectMultiScale(*this, img, rects_, scores_, shapes_);
  
  const double overlap = 0.3;
  vector<int> picked = nms(rects_, scores_, overlap);
  const int n = picked.size();
  rects.resize(n);
  scores.resize(n);
  shapes.resize(n);

  // relocate the shape points
  for (int i = 0; i < n; i++) {
    const int index = picked[i];
    Rect& rect = rects_[index];
    Mat_<double>& shape = shapes_[index];
    const int landmark_n = shape.cols / 2;
    for (int j = 0; j < landmark_n; j++) {
      shape(0, 2 * j) += rect.x;
      shape(0, 2 * j + 1) += rect.y;
    }
    rects[i] = rect;
    shapes[i] = shape;
    scores[i] = scores_[index];
  }

  return n;
}

} // namespace jda
