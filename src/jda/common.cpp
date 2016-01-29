#include <omp.h>
#include <ctime>
#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <jsmn.hpp>
#include "jda/common.hpp"

using namespace cv;
using namespace std;
using namespace jsmn;

namespace jda {

int Feature::CalcFeatureValue(const Mat& o, const Mat& h, const Mat& q, \
                              const Mat_<double>& s) const {
  Mat img;
  switch (scale) {
  case ORIGIN:
    img = o; // ref
    break;
  case HALF:
    img = h; // ref
    break;
  case QUARTER:
    img = q; // ref
    break;
  default:
    dieWithMsg("Unsupported SCALE");
    break;
  }

  double x1, y1, x2, y2;
  const int width = img.cols;
  const int height = img.rows;
  x1 = (s(0, 2 * landmark_id1) + offset1_x)*width;
  y1 = (s(0, 2 * landmark_id1 + 1) + offset1_y)*height;
  x2 = (s(0, 2 * landmark_id2) + offset2_x)*width;
  y2 = (s(0, 2 * landmark_id2 + 1) + offset2_y)*height;
  int x1_ = int(round(x1));
  int y1_ = int(round(y1));
  int x2_ = int(round(x2));
  int y2_ = int(round(y2));

  checkBoundaryOfImage(width, height, x1_, y1_);
  checkBoundaryOfImage(width, height, x2_, y2_);

  int val = int(img.at<uchar>(y1_, x1_)) - int(img.at<uchar>(y2_, x2_));
  return val;
}

void LOG(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  char msg[256];
  vsprintf(msg, fmt, args);
  va_end(args);

  char buff[256];
  time_t t = time(NULL);
  strftime(buff, sizeof(buff), "[%x - %X]", localtime(&t));

  // log
  Config& c = Config::GetInstance();
  if (c.log_to_console) printf("%s %s\n", buff, msg);
  if (c.log_to_file) {
    if (!c.log_file) { // lazy open
      c.log_file = fopen(c.log_file_path.c_str(), "w");
      if (!c.log_file) dieWithMsg("Can not open log file to write");
    }
    fprintf(c.log_file, "%s %s\n", buff, msg);
  }
}

void dieWithMsg(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  char msg[256];
  vsprintf(msg, fmt, args);
  va_end(args);

  LOG(msg);
  exit(-1);
}

double calcMeanError(const vector<Mat_<double> >& gt_shapes, \
                     const vector<Mat_<double> >& current_shapes) {
  const Config& c = Config::GetInstance();
  const int N = gt_shapes.size();
  const int landmark_n = c.landmark_n;
  double e = 0.;
  Mat_<double> delta_shape;
  for (int i = 0; i < N; i++) {
    delta_shape = gt_shapes[i] - current_shapes[i];
    for (int j = 0; j < landmark_n; j++) {
      e += std::sqrt(std::pow(delta_shape(0, 2 * j), 2) + \
                     std::pow(delta_shape(0, 2 * j + 1), 2));
    }
  }
  e /= landmark_n * N;
  e /= c.img_o_size;
  return e;
}

Mat drawShape(const Mat& img, const Mat_<double>& shape) {
  Mat img_ = img.clone();
  const int landmark_n = shape.cols / 2;
  for (int i = 0; i < landmark_n; i++) {
    circle(img_, Point(shape(0, 2 * i), shape(0, 2 * i + 1)), 1, Scalar(0, 255, 0), -1);
  }
  return img_;
}
Mat drawShape(const Mat& img, const Mat_<double>& shape, const Rect& bbox) {
  Mat img_ = img.clone();
  const int landmark_n = shape.cols / 2;
  rectangle(img_, bbox, Scalar(0, 0, 255), 2);
  for (int i = 0; i < landmark_n; i++) {
    circle(img_, Point(shape(0, 2 * i), shape(0, 2 * i + 1)), 1, Scalar(0, 255, 0), -1);
  }
  return img_;
}

void showImage(const Mat& img) {
  cv::imshow("img", img);
  cv::waitKey(0);
}

Config::Config() {
  jsmn::Object json_config = jsmn::parse("../config.json");

  // model meta data
  T = json_config["T"].unwrap<Number>();
  K = json_config["K"].unwrap<Number>();
  landmark_n = json_config["landmark_n"].unwrap<Number>();
  tree_depth = json_config["tree_depth"].unwrap<Number>();
  shift_size = json_config["random_shift"].unwrap<Number>();

  // image size
  jsmn::Object& image_size_config = json_config["image_size"].unwrap<Object>();
  multi_scale = image_size_config["multi_scale"].unwrap<Boolean>();
  img_o_size = image_size_config["origin_size"].unwrap<Number>();
  img_h_size = image_size_config["half_size"].unwrap<Number>();
  img_q_size = image_size_config["quarter_size"].unwrap<Number>();

  // hard negative mining
  jsmn::Object& mining_config = json_config["hard_negative_mining"].unwrap<Object>();
  x_step = mining_config["x_step"].unwrap<Number>();
  y_step = mining_config["y_step"].unwrap<Number>();
  scale_factor = mining_config["scale"].unwrap<Number>();
  mining_pool_size = omp_get_max_threads();
  esp = 2.2e-16;

  // stage parameters
  jsmn::Object& stages = json_config["stages"].unwrap<Object>();
  this->feats.clear();
  this->radius.clear();
  this->probs.clear();
  for (int i = 0; i < T; i++) {
    this->feats.push_back(stages["feature_pool_size"][i].unwrap<Number>());
    this->nps.push_back(stages["neg_pos_ratio"][i].unwrap<Number>());
    this->radius.push_back(stages["random_sample_radius"][i].unwrap<Number>());
    this->probs.push_back(stages["classification_p"][i].unwrap<Number>());
    this->recall.push_back(stages["recall"][i].unwrap<Number>());
  }

  // data
  jsmn::Object& data = json_config["data"].unwrap<Object>();
  face_txt = data["face"].unwrap<jsmn::String>();
  test_txt = data["test"].unwrap<jsmn::String>();
  jsmn::Array& neg_list = data["background"].unwrap<jsmn::Array>();
  bg_txts.resize(neg_list.size());
  for (int i = 0; i < neg_list.size(); i++) {
    bg_txts[i] = neg_list[i].unwrap<jsmn::String>();
  }

  // status
  resume_model = json_config["resume_model"].unwrap<jsmn::String>();
  snapshot_iter = json_config["snapshot_iter"].unwrap<Number>();

  // fddb benchmark
  jsmn::Object& fddb = json_config["fddb"].unwrap<Object>();
  fddb_dir = fddb["dir"].unwrap<jsmn::String>();
  fddb_result = fddb["out"].unwrap<Boolean>();
  fddb_nms = fddb["nms"].unwrap<Boolean>();
  fddb_minimum_size = fddb["minimum_size"].unwrap<Number>();
  fddb_step = fddb["step"].unwrap<Number>();
  fddb_scale_factor = fddb["scale"].unwrap<Number>();
  fddb_overlap = fddb["overlap"].unwrap<Number>();

  // cart
  jsmn::Object& cart = json_config["cart"].unwrap<Object>();
  restart_on = cart["restart"]["on"].unwrap<Boolean>();
  restart_th = cart["restart"]["th"].unwrap<Number>();
  restart_times = cart["restart"]["times"].unwrap<Number>();

  // face augment
  jsmn::Object& face = json_config["face"].unwrap<Object>();
  face_augment_on = face["online_augment"].unwrap<Boolean>();
  symmetric_landmarks.resize(2);
  int offset = face["symmetric_landmarks"]["offset"].unwrap<Number>();
  jsmn::Array& left = face["symmetric_landmarks"]["left"].unwrap<Array>();
  jsmn::Array& right = face["symmetric_landmarks"]["right"].unwrap<Array>();
  JDA_Assert(left.size() == right.size(), "Symmetric left and right landmarks are not equal size");
  symmetric_landmarks[0].resize(left.size());
  symmetric_landmarks[1].resize(right.size());
  for (int i = 0; i < left.size(); i++) {
    symmetric_landmarks[0][i] = left[i].unwrap<Number>() - offset;
    symmetric_landmarks[1][i] = right[i].unwrap<Number>() - offset;
  }

  // log
  jsmn::Object& log = json_config["log"].unwrap<Object>();
  log_to_console = log["console"].unwrap<Boolean>();
  log_to_file = log["file"].unwrap<Boolean>();
  if (log_to_file) {
    string dir = log["directory"].unwrap<jsmn::String>();
    char buff[300];
    char path[300];
    time_t t = time(NULL);
    strftime(buff, sizeof(buff), "%Y%m%d-%H%M%S", localtime(&t));
    if (dir[dir.length() - 1] != '/') {
      sprintf(path, "%s/%s.log", dir.c_str(), buff);
    }
    else {
      sprintf(path, "%s%s.log", dir.c_str(), buff);
    }
    log_file_path = path;
    log_file = NULL;
  }
}

Config::~Config() {
  if (log_to_file && log_file) {
    fclose(log_file);
  }
}

} // namespace jda
