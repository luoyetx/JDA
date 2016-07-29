#include <omp.h>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cassert>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/data.hpp"
#include "jda/cart.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;

namespace jda {

DataSet::DataSet() {}
DataSet::~DataSet() {}

Mat_<int> DataSet::CalcFeatureValues(const vector<Feature>& feature_pool, \
                                     const vector<int>& idx) const {
  const int n = feature_pool.size();
  const int m = idx.size();

  if (m == 0) {
    return Mat_<int>();
  }

  Mat_<int> features(n, m);

  for (int i = 0; i < n; i++) {
    const Feature& feature = feature_pool[i];
    int* ptr = features.ptr<int>(i);

    #pragma omp parallel for
    for (int j = 0; j < m; j++) {
      const Mat& img = imgs[idx[j]];
      const Mat& img_half = imgs_half[idx[j]];
      const Mat& img_quarter = imgs_quarter[idx[j]];
      const Mat_<double>& shape = current_shapes[idx[j]];
      ptr[j] = feature.CalcFeatureValue(img, img_half, img_quarter, shape);
    }
  }

  return features;
}

Mat_<double> DataSet::CalcShapeResidual(const vector<int>& idx) const {
  JDA_Assert(is_pos == true, "Negative Dataset can not use `CalcShapeResidual`");
  const int n = idx.size();
  // all landmark
  const int landmark_n = gt_shapes[0].cols / 2;
  Mat_<double> shape_residual(n, landmark_n * 2);

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    shape_residual.row(i) = gt_shapes[idx[i]] - current_shapes[idx[i]];
  }
  return shape_residual;
}
Mat_<double> DataSet::CalcShapeResidual(const vector<int>& idx, int landmark_id) const {
  JDA_Assert(is_pos == true, "Negative Dataset can not use `CalcShapeResidual`");
  const int n = idx.size();
  // specific landmark
  Mat_<double> shape_residual(n, 2);

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    shape_residual(i, 0) = gt_shapes[idx[i]](0, 2 * landmark_id) - \
                           current_shapes[idx[i]](0, 2 * landmark_id);
    shape_residual(i, 1) = gt_shapes[idx[i]](0, 2 * landmark_id + 1) - \
                           current_shapes[idx[i]](0, 2 * landmark_id + 1);
  }
  return shape_residual;
}

Mat_<double> DataSet::CalcMeanShape() {
  mean_shape = gt_shapes[0].clone();
  const int n = gt_shapes.size();
  for (int i = 1; i < n; i++) {
    mean_shape += gt_shapes[i];
  }
  mean_shape /= n;
  return mean_shape;
}

void DataSet::RandomShape(const Mat_<double>& mean_shape, Mat_<double>& shape) {
  const Config& c = Config::GetInstance();
  RNG rng = RNG(getTickCount());
  double x = rng.uniform(-c.shift_size, c.shift_size);
  double y = rng.uniform(-c.shift_size, c.shift_size);
  shape.create(mean_shape.rows, mean_shape.cols);
  // only apply a global shift
  for (int j = 0; j < c.landmark_n; j++) {
    shape(0, 2 * j) = mean_shape(0, 2 * j) + x;
    shape(0, 2 * j + 1) = mean_shape(0, 2 * j + 1) + y;
  }
}
void DataSet::RandomShapes(const Mat_<double>& mean_shape, vector<Mat_<double> >& shapes) {
  Config& c = Config::GetInstance();
  const int n = shapes.size();
  const int landmark_n = c.landmark_n;
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    RNG& rng = c.rng_pool[omp_get_thread_num() + 1];
    double x = rng.uniform(-c.shift_size, c.shift_size);
    double y = rng.uniform(-c.shift_size, c.shift_size);
    shapes[i].create(mean_shape.rows, mean_shape.cols);
    // only apply a global shift
    for (int j = 0; j < landmark_n; j++) {
      shapes[i](0, 2 * j) = mean_shape(0, 2 * j) + x;
      shapes[i](0, 2 * j + 1) = mean_shape(0, 2 * j + 1) + y;
    }
  }
}

void DataSet::UpdateWeights() {
  const double flag = -(is_pos ? 1 : -1);

  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    double tmp = flag*scores[i];
    //if (tmp > 15.) tmp = 15.;
    weights[i] = exp(tmp);
  }
}
void DataSet::UpdateWeights(DataSet& pos, DataSet& neg) {
  pos.UpdateWeights();
  neg.UpdateWeights();
  // normalize to 1
  const int pos_n = pos.size;
  const int neg_n = neg.size;
  double sum_pos_w = 0.;
  double sum_neg_w = 0.;
  double sum_w = 0.;

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      for (int i = 0; i < pos_n; i++) {
        sum_pos_w += pos.weights[i];
      }
    }
    #pragma omp section
    {
      for (int i = 0; i < neg_n; i++) {
        sum_neg_w += neg.weights[i];
      }
    }
  }

  sum_w = sum_pos_w + sum_neg_w;
  double sum_w_ = 1. / sum_w;

  #pragma omp parallel for
  for (int i = 0; i < pos_n; i++) {
    pos.weights[i] *= sum_w_;
  }

  #pragma omp parallel for
  for (int i = 0; i < neg_n; i++) {
    neg.weights[i] *= sum_w_;
  }
}

void DataSet::UpdateScores(const Cart& cart) {
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    const Mat& img = imgs[i];
    const Mat& img_h = imgs_half[i];
    const Mat& img_q = imgs_quarter[i];
    const Mat_<double>& shape = current_shapes[i];
    int leaf_node_idx = cart.Forward(img, img_h, img_q, shape);
    last_scores[i] = scores[i]; // cache
    scores[i] += cart.scores[leaf_node_idx];
  }
  is_sorted = false;
}

void DataSet::Swap(int i, int j) {
  std::swap(imgs[i], imgs[j]);
  std::swap(imgs_half[i], imgs_half[j]);
  std::swap(imgs_quarter[i], imgs_quarter[j]);
  if (is_pos) std::swap(gt_shapes[i], gt_shapes[j]);
  std::swap(current_shapes[i], current_shapes[j]);
  std::swap(scores[i], scores[j]);
  std::swap(last_scores[i], last_scores[j]);
  std::swap(weights[i], weights[j]);
}

double DataSet::CalcThresholdByRate(double rate) {
  if (!is_sorted) QSort();
  int offset = size - 1 - int(rate*size);
  return scores[offset];
}
double DataSet::CalcThresholdByNumber(int remove) {
  if (!is_sorted) QSort();
  int offset = size - 1 - remove;
  if (offset < 0) offset = 0;
  return scores[offset];
}

void DataSet::Remove(double th) {
  if (!is_sorted) QSort();
  int offset = size - 1;
  const int upper = scores.size();
  // get offset
  while (offset >=0 && scores[offset] < th) offset--;
  offset++;
  imgs.resize(offset);
  imgs_half.resize(offset);
  imgs_quarter.resize(offset);
  if (is_pos) gt_shapes.resize(offset);
  current_shapes.resize(offset);
  scores.resize(offset);
  last_scores.resize(offset);
  weights.resize(offset);
  // new size
  size = offset;
}

int DataSet::PreRemove(double th) {
  if (!is_sorted) QSort();
  int offset = size - 1;
  const int upper = scores.size();
  // get offset
  while (offset >= 0 && scores[offset] < th) offset--;
  return size - 1 - offset;
}

void DataSet::QSort() {
  if (is_sorted) return;
  _QSort_(0, size - 1);
  is_sorted = true;
}
void DataSet::_QSort_(int left, int right) {
  int i = left;
  int j = right;
  double t = scores[(left + right) / 2];
  do {
    while (scores[i] > t) i++;
    while (scores[j] < t) j--;
    if (i <= j) {
      // swap data point
      Swap(i, j);
      i++; j--;
    }
  } while (i <= j);

  #pragma omp parallel sections
  {
    #pragma omp section
    {
      if (left < j) _QSort_(left, j);
    }
    #pragma omp section
    {
      if (i < right) _QSort_(i, right);
    }
  }
}

void DataSet::ResetScores() {
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    scores[i] = last_scores[i];
  }
  is_sorted = false;
}

void DataSet::CalcMeanAndStd(const DataSet& pos, const DataSet& neg, double& mean, double& std) {
  mean = 0.;
  std = 1.;
  // calc mean
  for (int i = 0; i < pos.size; i++) {
    mean += pos.scores[i];
  }
  for (int i = 0; i < neg.size; i++) {
    mean += neg.scores[i];
  }
  mean /= pos.size + neg.size;
  // calc std
  double var = 0.;
  for (int i = 0; i < pos.size; i++) {
    var += std::pow(pos.scores[i] - mean, 2);
  }
  for (int i = 0; i < neg.size; i++) {
    var += std::pow(neg.scores[i] - mean, 2);
  }
  var /= pos.size + neg.size;
  std = std::sqrt(var);
}

void DataSet::ApplyMeanAndStd(const double mean, const double std) {
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    scores[i] = (scores[i] - mean) / std;
  }
}

void DataSet::Clear() {
  imgs.clear();
  imgs_half.clear();
  imgs_quarter.clear();
  current_shapes.clear();
  gt_shapes.clear();
  scores.clear();
  last_scores.clear();
  weights.clear();
  is_sorted = false;
  size = 0;
}

void DataSet::Dump(const string& dir) const {
  const int n = size;

  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    char buff[300];
    sprintf(buff, "%s/%06d.jpg", dir.c_str(), i);
    Mat img = drawShape(imgs[i], current_shapes[i] * Config::GetInstance().img_o_size);
    cv::imwrite(buff, img);
  }
}

void DataSet::MoreNegSamples(int pos_size, double rate) {
  JDA_Assert(is_pos == false, "Positive Dataset can not use `MoreNegSamples`");
  const Config& c = Config::GetInstance();
  // get the size of negative to generate
  const int size_ = rate*pos_size - this->size;
  if (size_ <= 0) {
    // generation is not needed
    return;
  }
  LOG("Negative Samples are insufficient");
  LOG("Use hard negative mining for size = %d", size_);
  vector<Mat> imgs_;
  vector<double> scores_;
  vector<Mat_<double> > shapes_;
  int extra_size;

  TIMER_BEGIN
    extra_size = neg_generator.Generate(*c.joincascador, size_, \
                                        imgs_, scores_, shapes_);
    LOG("We have mined %d hard negative samples, current_bg_idx = %d, it costs %.2lf s", \
        extra_size, neg_generator.current_idx, TIMER_NOW);
  TIMER_END

  const int expanded = imgs.size() + imgs_.size();
  imgs.reserve(expanded);
  imgs_half.reserve(expanded);
  imgs_quarter.reserve(expanded);
  //gt_shapes.reserve(expanded);
  current_shapes.reserve(expanded);
  scores.reserve(expanded);
  last_scores.reserve(expanded);
  weights.reserve(expanded);
  for (int i = 0; i < extra_size; i++) {
    Mat half, quarter;
    cv::resize(imgs_[i], imgs_[i], Size(c.img_o_size, c.img_o_size));
    cv::resize(imgs_[i], half, Size(c.img_h_size, c.img_h_size));
    cv::resize(imgs_[i], quarter, Size(c.img_q_size, c.img_q_size));
    imgs.push_back(imgs_[i]);
    imgs_half.push_back(half);
    imgs_quarter.push_back(quarter);
    current_shapes.push_back(shapes_[i]);
    scores.push_back(scores_[i]);
    last_scores.push_back(0);
    weights.push_back(0); // all weights will be updated by calling `UpdateWeights`
  }
  size = expanded;
  is_sorted = false;
}

/*!
 * \breif get face from original image using bbox
 * \note  if bbox out of range, fill the rest with black
 *
 * \param img     original image
 * \param bbox    face bbox
 * \return        face
 */
static Mat getFace(const Mat& img, const Rect& bbox) {
  const int rows = img.rows;
  const int cols = img.cols;
  if ((bbox.x >= 0) && (bbox.y >= 0) && \
      (bbox.x + bbox.width < cols) && (bbox.y + bbox.height < rows)) {
    return img(bbox).clone();
  }

  // out of range, large origin image and fill with black
  const int rows_ = 3 * img.rows;
  const int cols_ = 3 * img.cols;
  Mat img_(rows_, cols_, CV_8UC1);
  img_.setTo(0);

  const int x = cols / 2;
  const int y = rows / 2;
  const int w = cols;
  const int h = rows;
  Rect roi(x, y, w, h);
  img.copyTo(img_(roi));

  Rect bbox_(bbox.x + x, bbox.y + y, bbox.width, bbox.height);
  return img_(bbox_).clone();
}

void DataSet::LoadPositiveDataSet(const string& positive) {
  const Config& c = Config::GetInstance();
  const int landmark_n = c.landmark_n;
  FILE* file = fopen(positive.c_str(), "r");
  JDA_Assert(file, "Can not open positive dataset file");

  char buff[300];
  vector<string> path;
  vector<Rect> bboxes;
  imgs.clear();
  imgs_half.clear();
  imgs_quarter.clear();
  gt_shapes.clear();
  // read all meta data
  while (fscanf(file, "%s", buff) > 0) {
    path.push_back(string(buff));
    // bbox
    Rect bbox;
    fscanf(file, "%d%d%d%d", &bbox.x, &bbox.y, &bbox.width, &bbox.height);
    bboxes.push_back(bbox);
    // shape
    Mat_<double> shape(1, 2 * landmark_n);
    const double* ptr = shape.ptr<double>(0);
    for (int i = 0; i < 2 * landmark_n; i++) {
      fscanf(file, "%lf", ptr + i);
    }
    gt_shapes.push_back(shape);
  }
  fclose(file);

  const int n = path.size();
  size = c.face_augment_on ? 2 * n : n;
  imgs.resize(size);
  imgs_half.resize(size);
  imgs_quarter.resize(size);
  gt_shapes.resize(size);

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    // face image should be a sqaure
    Mat origin = imread(path[i]); // some jpeg file can not be loaded if using CV_LOAD_IMAGE_GRAYSCALE
    if (!origin.data) {
      dieWithMsg("Can not open %s", path[i].c_str());
    }
    cvtColor(origin, origin, CV_BGR2GRAY);
    // get face
    Mat face = getFace(origin, bboxes[i]);
    // relocate landmarks
    for (int j = 0; j < landmark_n; j++) {
      gt_shapes[i](0, 2 * j) = (gt_shapes[i](0, 2 * j) - bboxes[i].x) / bboxes[i].width;
      gt_shapes[i](0, 2 * j + 1) = (gt_shapes[i](0, 2 * j + 1) - bboxes[i].y) / bboxes[i].height;
    }
    Mat img, half, quarter; // should be defined here due to the memory manager by OpenCV Mat
    cv::resize(face, img, Size(c.img_o_size, c.img_o_size));
    cv::resize(face, half, Size(c.img_h_size, c.img_h_size));
    cv::resize(face, quarter, Size(c.img_q_size, c.img_q_size));
    imgs[i] = img;
    imgs_half[i] = half;
    imgs_quarter[i] = quarter;

    if (c.face_augment_on) { // flip
      cv::flip(imgs[i], imgs[i + n], 1);
      cv::flip(imgs_half[i], imgs_half[i + n], 1);
      cv::flip(imgs_quarter[i], imgs_quarter[i + n], 1);
      gt_shapes[i + n] = gt_shapes[i].clone();
      // flip all landmarks
      for (int j = 0; j < c.landmark_n; j++) {
        gt_shapes[i + n](0, 2 * j) = 1 - gt_shapes[i + n](0, 2 * j);
      }
      // swap symmetric landmarks
      for (int j = 0; j < c.symmetric_landmarks[0].size(); j++) {
        const int idx1 = c.symmetric_landmarks[0][j];
        const int idx2 = c.symmetric_landmarks[1][j];
        double x1, y1, x2, y2;
        x1 = gt_shapes[i + n](0, 2 * idx2);
        y1 = gt_shapes[i + n](0, 2 * idx2 + 1);
        x2 = gt_shapes[i + n](0, 2 * idx1);
        y2 = gt_shapes[i + n](0, 2 * idx1 + 1);
        gt_shapes[i + n](0, 2 * idx1) = x1;
        gt_shapes[i + n](0, 2 * idx1 + 1) = y1;
        gt_shapes[i + n](0, 2 * idx2) = x2;
        gt_shapes[i + n](0, 2 * idx2 + 1) = y2;
      }
    }
  }

  is_pos = true;
  current_shapes.resize(size);
  scores.resize(size);
  last_scores.resize(size);
  weights.resize(size);
  std::fill(weights.begin(), weights.end(), 0);
  std::fill(scores.begin(), scores.end(), 0);
  std::fill(last_scores.begin(), last_scores.end(), 0);

  CalcMeanShape();
  DataSet::RandomShapes(mean_shape, current_shapes);
}

void DataSet::LoadNegativeDataSet(const vector<string>& negative) {
  const Config& c = Config::GetInstance();
  neg_generator.Load(negative);
  Clear();
  is_pos = false;
}

void DataSet::LoadDataSet(DataSet& pos, DataSet& neg) {
  const Config& c = Config::GetInstance();
  pos.LoadPositiveDataSet(c.face_txt);
  neg.LoadNegativeDataSet(c.bg_txts);
}

/*!
 * \breif Write DataSet to binary file
 * \param data    DataSet
 * \param fout    binary file discriptor
 */
static void writeDataSet(const DataSet& data, FILE* fout) {
  int flag;
  if (data.is_pos) flag = 1;
  else flag = 0;

  fwrite(&flag, sizeof(int), 1, fout);
  int n = data.size;
  fwrite(&n, sizeof(int), 1, fout);
  if (data.is_pos) {
    fwrite(data.mean_shape.ptr<double>(0), sizeof(double), data.mean_shape.cols, fout);
  }
  for (int i = 0; i < n; i++) {
    // img
    const Mat& img_o = data.imgs[i];
    fwrite(&img_o.cols, sizeof(int), 1, fout);
    fwrite(&img_o.rows, sizeof(int), 1, fout);
    for (int j = 0; j < img_o.rows; j++) {
      fwrite(img_o.ptr<uchar>(j), sizeof(uchar), img_o.cols, fout);
    }
    const Mat& img_h = data.imgs_half[i];
    fwrite(&img_h.cols, sizeof(int), 1, fout);
    fwrite(&img_h.rows, sizeof(int), 1, fout);
    for (int j = 0; j < img_h.rows; j++) {
      fwrite(img_h.ptr<uchar>(j), sizeof(uchar), img_h.cols, fout);
    }
    const Mat& img_q = data.imgs_quarter[i];
    fwrite(&img_q.cols, sizeof(int), 1, fout);
    fwrite(&img_q.rows, sizeof(int), 1, fout);
    for (int j = 0; j < img_q.rows; j++) {
      fwrite(img_q.ptr<uchar>(j), sizeof(uchar), img_q.cols, fout);
    }
    // gt_shape if positive samples
    if (data.is_pos) {
      const Mat_<double>& gt_shape = data.gt_shapes[i];
      fwrite(gt_shape.ptr<double>(0), sizeof(double), gt_shape.cols, fout);
    }
    // current_shapes
    const Mat_<double>& current_shape = data.current_shapes[i];
    fwrite(current_shape.ptr<double>(0), sizeof(double), current_shape.cols, fout);
    // score
    double score = data.scores[i];
    fwrite(&score, sizeof(double), 1, fout);
    // weight
    double weight = data.weights[i];
    fwrite(&weight, sizeof(double), 1, fout);
  }
}

/*!
 * \breif Read DataSet from a binary file and initialize all memory
 * \note  nega_generator will be initialized in this function
 *
 * \param data    DataSet
 * \param fin     binary file discriptor
 */
static void readDataSet(DataSet& data, FILE* fin) {
  const Config& c = Config::GetInstance();

  int flag = 0;
  fread(&flag, sizeof(int), 1, fin);
  if (flag == 1) data.is_pos = true;
  else data.is_pos = false;

  int n;
  fread(&n, sizeof(int), 1, fin);
  if (data.is_pos) {
    data.mean_shape = Mat_<double>(1, 2 * c.landmark_n);
    fread(data.mean_shape.ptr<double>(0), sizeof(double), data.mean_shape.cols, fin);
  }
  // malloc and initialize
  data.imgs.resize(n);
  data.imgs_half.resize(n);
  data.imgs_quarter.resize(n);
  if (data.is_pos) data.gt_shapes.resize(n);
  data.current_shapes.resize(n);
  data.scores.resize(n);
  data.last_scores.resize(n);
  data.weights.resize(n);
  data.is_sorted = false;
  data.size = n;

  for (int i = 0; i < n; i++) {
    // img
    int rows, cols;
    fread(&cols, sizeof(int), 1, fin);
    fread(&rows, sizeof(int), 1, fin);
    data.imgs[i] = Mat(rows, cols, CV_8UC1);
    for (int j = 0; j < rows; j++) {
      fread(data.imgs[i].ptr<uchar>(j), sizeof(uchar), cols, fin);
    }
    fread(&cols, sizeof(int), 1, fin);
    fread(&rows, sizeof(int), 1, fin);
    data.imgs_half[i] = Mat(rows, cols, CV_8UC1);
    for (int j = 0; j < rows; j++) {
      fread(data.imgs_half[i].ptr<uchar>(j), sizeof(uchar), cols, fin);
    }
    fread(&cols, sizeof(int), 1, fin);
    fread(&rows, sizeof(int), 1, fin);
    data.imgs_quarter[i] = Mat(rows, cols, CV_8UC1);
    for (int j = 0; j < rows; j++) {
      fread(data.imgs_quarter[i].ptr<uchar>(j), sizeof(uchar), cols, fin);
    }
    // gt_shape if positive samples
    if (data.is_pos) {
      Mat_<double>& gt_shape = data.gt_shapes[i];
      gt_shape = Mat_<double>::zeros(1, 2 * c.landmark_n);
      fread(gt_shape.ptr<double>(0), sizeof(double), gt_shape.cols, fin);
    }
    // current_shapes
    Mat_<double>& current_shape = data.current_shapes[i];
    current_shape = Mat_<double>::zeros(1, 2 * c.landmark_n);
    fread(current_shape.ptr<double>(0), sizeof(double), current_shape.cols, fin);
    // score
    double& score = data.scores[i];
    fread(&score, sizeof(double), 1, fin);
    // weight
    double& weight = data.weights[i];
    fread(&weight, sizeof(double), 1, fin);

    // init
    data.last_scores[i] = 0;
  }

  // init nega_generator if data is negative dataset
  if (!data.is_pos) {
    data.neg_generator.Load(c.bg_txts);
  }
}

void DataSet::Snapshot(const DataSet& pos, const DataSet& neg) {
  const Config& c = Config::GetInstance();

  int stage_idx = c.joincascador->current_stage_idx;
  int cart_idx = c.joincascador->current_cart_idx;
  char buff1[256];
  char buff2[256];
  time_t t = time(NULL);
  strftime(buff1, sizeof(buff1), "%Y%m%d-%H%M%S", localtime(&t));
  sprintf(buff2, "../data/dump/jda_data_%s_stage_%d_cart_%d.data", \
          buff1, stage_idx + 1, cart_idx + 1);

  if (!EXISTS("../data/dump")) {
    MKDIR("../data/dump");
  }

  FILE* fout = fopen(buff2, "wb");
  if (fout == NULL) {
    LOG("Can not write to file, Skip DataSet::Snapshot()");
  }
  else{
    LOG("DataSet Snapshot Begin");
    LOG("Write all positive training samples");
    writeDataSet(pos, fout);
    LOG("Write all negative training samples");
    writeDataSet(neg, fout);
    LOG("DataSet Snapshot End");
  }
  fclose(fout);
}

void DataSet::Resume(const string& data_file, DataSet& pos, DataSet& neg) {
  JDA_Assert(EXISTS(data_file.c_str()), "No Data File, Please Check It");
  FILE* fin = fopen(data_file.c_str(), "rb");
  JDA_Assert(fin, "Can not open data file");

  readDataSet(pos, fin);
  readDataSet(neg, fin);

  fclose(fin);
}

// Negative Generator

NegGenerator::NegGenerator()
  : current_idx(0), current_hd_idx(0), x(0), y(0), transform_type(0), resets(0) {
}
NegGenerator::~NegGenerator() {
}

Mat NegGenerator::NextImage() {
  Mat patch;
  // use hard if any
  if (current_hd_idx < hds.size()) {
    patch = hds[current_hd_idx].clone();
    current_hd_idx++;
    if (current_hd_idx >= hds.size()) {
      hds.clear(); // release memory
    }
    return patch;
  }
  // background image
  x += step;
  if (x + win_size > bg_img.cols) {
    x = 0;
    y += step;
    if (y + win_size > bg_img.rows) {
      y = 0;
      win_size *= factor;
      if (win_size >= bg_img.cols || win_size >= bg_img.rows) {
        // next transform
        const Config& c = Config::GetInstance();
        win_size = c.img_o_size;
        // next image
        RNG rng(cv::getTickCount());
        factor = rng.uniform(1.1, 1.5);
        step = rng.uniform(2, c.img_q_size);
        while (true) {
          current_idx++;
          if (current_idx >= list.size()) {
            LOG("Run out of background images. Reset current_idx and restart.");
            current_idx = -1;
            transform_type = (transform_type + 1) % 8;
            resets++;
            LOG("Current reset = %d, Current transform type = %d", resets, transform_type);
            continue;
          }
          bg_img = cv::imread(list[current_idx], CV_LOAD_IMAGE_GRAYSCALE);
          if (!bg_img.data || bg_img.cols <= win_size || bg_img.rows <= win_size) {
            // continue
          }
          else {
            break;
          }
        }

        // perform transform to the bg image
        switch (transform_type){
        case 0:
          break;
        case 1:
          flip(bg_img, bg_img, 0);
          transpose(bg_img, bg_img);
          break;
        case 2:
          flip(bg_img, bg_img, -1);
          break;
        case 3:
          flip(bg_img, bg_img, 1);
          transpose(bg_img, bg_img);
          break;
        case 4:
          flip(bg_img, bg_img, 1);
          break;
        case 5:
          flip(bg_img, bg_img, -1);
          transpose(bg_img, bg_img);
          break;
        case 6:
          flip(bg_img, bg_img, -1);
          flip(bg_img, bg_img, 1);
          break;
        case 7:
          flip(bg_img, bg_img, 0);
          transpose(bg_img, bg_img);
          flip(bg_img, bg_img, 1);
          break;
        default:
          dieWithMsg("No such transform type for background image augmentation.");
          break;
        }
      }
    }
  }
  patch = bg_img(Rect(x, y, win_size, win_size)).clone();
  return patch;
}

int NegGenerator::Generate(const JoinCascador& joincascador, int size, \
                           vector<Mat>& imgs, vector<double>& scores, \
                           vector<Mat_<double> >& shapes) {
  const Config& c = Config::GetInstance();
  imgs.clear();
  scores.clear();
  shapes.clear();

  const int pool_size = c.thread_n;
  imgs.reserve(size + pool_size); // enough memory to overflow
  scores.reserve(size + pool_size);
  shapes.reserve(size + pool_size);

  vector<Mat> region_pool(pool_size);
  vector<Mat> region_h_pool(pool_size);
  vector<Mat> region_q_pool(pool_size);
  vector<double> score_pool(pool_size);
  vector<Mat_<double> > shape_pool(pool_size);
  vector<int> used(pool_size);
  vector<int> carts_go_through(pool_size);
  int nega_n = 0; // not hard nega
  double carts_n = 0; // number of carts go through by all not hard nega, type `int` may overflow

  double ratio = 0.1;
  while (imgs.size() < size) {
    // We generate a negative sample pool for validation
    for (int i = 0; i < pool_size; i++) {
      region_pool[i] = NextImage();
    }

    #pragma omp parallel for
    for (int i = 0; i < pool_size; i++) {
      cv::resize(region_pool[i], region_h_pool[i], Size(c.img_h_size, c.img_h_size));
      cv::resize(region_pool[i], region_q_pool[i], Size(c.img_q_size, c.img_q_size));
      cv::resize(region_pool[i], region_pool[i], Size(c.img_o_size, c.img_o_size));
      bool is_face = joincascador.Validate(region_pool[i], region_h_pool[i], region_q_pool[i], \
                                           score_pool[i], shape_pool[i], carts_go_through[i]);
      if (is_face) used[i] = 1;
      else used[i] = 0;
    }

    // collect
    for (int i = 0; i < pool_size; i++) {
      if (used[i] > 0) {
        imgs.push_back(region_pool[i].clone()); // may not need to copy
        scores.push_back(score_pool[i]);
        shapes.push_back(shape_pool[i].clone()); // must copy
      }
      else { // not hard enough
        nega_n++;
        carts_n += carts_go_through[i];
      }
    }

    if (imgs.size() >= ratio*size) {
      while (imgs.size() >= ratio*size) ratio += 0.1;
      LOG("We have mined %d%%, bg image remains %d%%", int(double(imgs.size())/size*100.), \
          int(100. - double(current_idx) / list.size() * 100.));
    }
  }

  if (nega_n > 0) {
    const int patch_n = imgs.size() + nega_n;
    const double fp_rate = double(imgs.size()) / patch_n;
    const double average_cart = double(carts_n) / nega_n;
    LOG("Done with mining, number of not hard enough is %d", nega_n);
    LOG("Average number of cart to reject is %.2lf, FP = %lf", average_cart, fp_rate);
  }
  else {
    LOG("Done with mining, all nega is hard enough");
  }

  return imgs.size();
}

void NegGenerator::Load(const vector<string>& path) {
  const Config& c = Config::GetInstance();
  char buff[300];
  FILE* file;
  // background images
  list.clear();
  current_idx = 0;
  for (int i = 1; i < path.size(); i++) {
    file = fopen(path[i].c_str(), "r");
    sprintf(buff, "Can not open negative dataset file list, %s", path[i].c_str());
    JDA_Assert(file, buff);
    while (fscanf(file, "%s", buff) > 0) {
      list.push_back(buff);
    }
    fclose(file);
  }
  RNG rng(cv::getTickCount());
  std::random_shuffle(list.begin(), list.end(), rng);
  // initial bg state
  resets = 0;
  x = y = 0;
  transform_type = 0;
  win_size = c.img_o_size;
  factor = rng.uniform(1.1, 1.5);
  step = rng.uniform(2, c.img_q_size);
  bg_img = cv::imread(list[current_idx], CV_LOAD_IMAGE_GRAYSCALE);
  if (!bg_img.data) {
    dieWithMsg("Load background image %s failed", list[current_idx].c_str());
  }
  // load all hard negative samples
  hds.clear();
  current_hd_idx = 0;
  if (!c.use_hard) return;

  string hard = path[0];
  if (hard.substr(hard.length() - 3, hard.length()) == "txt") {
    // text hard negative
    FILE* file = fopen(path[0].c_str(), "r");
    sprintf(buff, "Can not open negative dataset file list, %s", path[0].c_str());
    JDA_Assert(file, buff);

    vector<string> hd_list;
    while (fscanf(file, "%s", buff) > 0) {
      hd_list.push_back(buff);
    }
    fclose(file);

    int thread_n = c.thread_n;
    omp_set_num_threads(3 * thread_n);
    SLEEP(1000);
    LOG("Load All Hard Negative Samples from text file");

    const int n = hd_list.size();
    hds.resize(n);

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      Mat img = cv::imread(hd_list[i], CV_LOAD_IMAGE_GRAYSCALE);
      if (!img.data) {
        LOG("Can not open hard negative file %s, Skip it", hd_list[i].c_str());
        continue;
      }
      cv::resize(img, img, Size(c.img_o_size, c.img_o_size));
      hds[i] = img;
    }

    omp_set_num_threads(thread_n);
    SLEEP(1000);
    LOG("All hard negative samples Done");

    if (false) {
      hds.resize(2 * n);
      for (int i = 0; i < n; i++) {
        if (hds[i].data) cv::flip(hds[i], hds[i + n], 1);
      }
    }

    LOG("Snapshot hard negative");
    FILE* data = fopen("../data/dump/hard.data", "wb");
    int n2 = hds.size();
    fwrite(&n2, sizeof(int), 1, data);
    for (int i = 0; i < n2; i++) {
      Mat& img = hds[i];
      if (!img.data) {
        int t4 = 0;
        fwrite(&t4, sizeof(int), 1, data);
        fwrite(&t4, sizeof(int), 1, data);
        continue;
      }
      fwrite(&img.cols, sizeof(int), 1, data);
      fwrite(&img.rows, sizeof(int), 1, data);
      for (int r = 0; r < img.rows; r++) {
        fwrite(img.ptr<uchar>(r), sizeof(uchar), img.cols, data);
      }
    }
    fclose(data);
  }
  else {
    // binary hard negative
    LOG("Load hard negative data from binary file");
    FILE* data = fopen(hard.c_str(), "rb");
    if (!data) {
      dieWithMsg("Can not open hard negative data, %s", hard.c_str());
    }
    int n = 0;
    fread(&n, sizeof(int), 1, data);
    hds.reserve(n);
    for (int i = 0; i < n; i++) {
      int rows, cols;
      Mat img;
      fread(&cols, sizeof(int), 1, data);
      fread(&rows, sizeof(int), 1, data);
      if (rows == 0 || cols == 0) {
        continue;
      }
      img = Mat(rows, cols, CV_8UC1);
      for (int r = 0; r < rows; r++) {
        fread(img.ptr<uchar>(r), sizeof(uchar), img.cols, data);
      }
      hds.push_back(img);
    }
    fclose(data);
  }

  std::random_shuffle(hds.begin(), hds.end(), rng);
}

} // namespace jda
