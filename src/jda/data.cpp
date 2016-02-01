#ifdef WIN32
#include <io.h>
#include <direct.h>
#define EXISTS(path) (access(path, 0)!=-1)
#define MKDIR(path) mkdir(path)
#else
#include <unistd.h>
#include <sys/stat.h>
#define EXISTS(path) (access(path, 0)!=-1)
#define MKDIR(path) mkdir(path, 0775)
#endif

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

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    const Feature& feature = feature_pool[i];
    int* ptr = features.ptr<int>(i);
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
Mat_<double> DataSet::CalcShapeResidual(const vector<int>& idx, int8_t landmark_id) const {
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

Mat_<double> DataSet::CalcMeanShape() const {
  Mat_<double> mean_shape = gt_shapes[0].clone();
  const int n = gt_shapes.size();
  for (int i = 1; i < n; i++) {
    mean_shape += gt_shapes[i];
  }
  mean_shape /= n;
  return mean_shape;
}

void DataSet::RandomShape(const Mat_<double>& mean_shape, Mat_<double>& shape) {
  const Config& c = Config::GetInstance();
  const double shift_size = c.shift_size;
  RNG rng = RNG(getTickCount());
  Mat_<double> shift(mean_shape.rows, mean_shape.cols);
  // we use a uniform distribution over [-shift_size, shift_size]
  rng.fill(shift, RNG::UNIFORM, -shift_size, shift_size);
  shape = mean_shape + shift;
}
void DataSet::RandomShapes(const Mat_<double>& mean_shape, vector<Mat_<double> >& shapes) {
  const Config& c = Config::GetInstance();
  const double shift_size = c.shift_size;
  const int n = shapes.size();
  RNG rng = RNG(getTickCount());
  Mat_<double> shift(mean_shape.rows, mean_shape.cols);
  for (int i = 0; i < n; i++) {
    // we use a uniform distribution over [-shift_size, shift_size]
    rng.fill(shift, RNG::UNIFORM, -shift_size, shift_size);
    shapes[i] = mean_shape + shift;
  }
}

void DataSet::UpdateWeights() {
  const double flag = -(is_pos ? 1 : -1);
  for (int i = 0; i < size; i++) {
    weights[i] = exp(flag*scores[i]);
  }
}
void DataSet::UpdateWeights(DataSet& pos, DataSet& neg) {
  pos.UpdateWeights();
  neg.UpdateWeights();
  // normalize to 1
  const int pos_n = pos.size;
  const int neg_n = neg.size;
  double sum_w = 0.;
  for (int i = 0; i < pos_n; i++) {
    sum_w += pos.weights[i];
  }
  for (int i = 0; i < neg_n; i++) {
    sum_w += neg.weights[i];
  }
  for (int i = 0; i < pos_n; i++) {
    pos.weights[i] /= sum_w;
  }
  for (int i = 0; i < neg_n; i++) {
    neg.weights[i] /= sum_w;
  }
}

void DataSet::UpdateScores(const Cart& cart) {
  #pragma omp parallel for
  for (int i = 0; i < size; i++) {
    const Mat& img = imgs[i];
    const Mat& img_h = imgs_half[i];
    const Mat& img_q = imgs_quarter[i];
    Mat_<double>& shape = current_shapes[i];
    int leaf_node_idx = cart.Forward(img, img_h, img_q, shape);
    scores[i] += cart.scores[leaf_node_idx];
  }
  is_sorted = false;
}

double DataSet::CalcThresholdByRate(double rate) {
  if (!is_sorted) QSort();
  int offset = size - 1 - int(rate*size);
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
  weights.resize(offset);
  // new size
  size = offset;
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
      std::swap(imgs[i], imgs[j]);
      std::swap(imgs_half[i], imgs_half[j]);
      std::swap(imgs_quarter[i], imgs_quarter[j]);
      if (is_pos) std::swap(gt_shapes[i], gt_shapes[j]);
      std::swap(current_shapes[i], current_shapes[j]);
      std::swap(scores[i], scores[j]);
      std::swap(weights[i], weights[j]);
      i++; j--;
    }
  } while (i <= j);
  if (left < j) _QSort_(left, j);
  if (i < right) _QSort_(i, right);
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
  const int extra_size = neg_generator.Generate(*c.joincascador, size_, \
                                                imgs_, scores_, shapes_);
  LOG("We have mined %d hard negative samples", extra_size);
  const int expanded = imgs.size() + imgs_.size();
  imgs.reserve(expanded);
  imgs_half.reserve(expanded);
  imgs_quarter.reserve(expanded);
  //gt_shapes.reserve(expanded);
  current_shapes.reserve(expanded);
  scores.reserve(expanded);
  weights.reserve(expanded);
  for (int i = 0; i < extra_size; i++) {
    Mat half, quarter;
    cv::resize(imgs_[i], half, Size(c.img_h_size, c.img_h_size));
    cv::resize(imgs_[i], quarter, Size(c.img_q_size, c.img_q_size));
    imgs.push_back(imgs_[i]);
    imgs_half.push_back(half);
    imgs_quarter.push_back(quarter);
    current_shapes.push_back(shapes_[i]);
    scores.push_back(scores_[i]);
    weights.push_back(0); // all weights will be updated by calling `UpdateWeights`
  }
  size = expanded;
  is_sorted = false;
}

void DataSet::LoadPositiveDataSet(const string& positive) {
  const Config& c = Config::GetInstance();
  const int landmark_n = c.landmark_n;
  FILE* file = fopen(positive.c_str(), "r");
  JDA_Assert(file, "Can not open positive dataset file");

  char buff[300];
  vector<string> path;
  imgs.clear();
  imgs_half.clear();
  imgs_quarter.clear();
  gt_shapes.clear();
  // read all meta data
  while (fscanf(file, "%s", buff) > 0) {
    path.push_back(string(buff));
    Mat_<double> shape(1, 2 * landmark_n);
    const double* ptr = shape.ptr<double>(0);
    for (int i = 0; i < 2 * landmark_n; i++) {
      fscanf(file, "%lf", ptr + i);
    }
    gt_shapes.push_back(shape);
  }
  fclose(file);

  const int n = path.size();
  imgs.resize(n);
  imgs_half.resize(n);
  imgs_quarter.resize(n);

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    // face image should be a sqaure
    Mat origin = imread(path[i], CV_LOAD_IMAGE_GRAYSCALE);
    if ((!origin.data) || (origin.cols != origin.rows)) {
      char msg[300];
      sprintf(msg, "%s is not valide", buff);
      JDA_Assert(false, msg);
    }
    gt_shapes[i] /= origin.cols; // relative landmark position range in [0, 1]
    Mat img, half, quarter; // should be defined here due to the memory manager by OpenCV Mat
    cv::resize(origin, img, Size(c.img_o_size, c.img_o_size));
    cv::resize(origin, half, Size(c.img_h_size, c.img_h_size));
    cv::resize(origin, quarter, Size(c.img_q_size, c.img_q_size));
    imgs[i] = img;
    imgs_half[i] = half;
    imgs_quarter[i] = quarter;
  }

  size = n;
  is_pos = true;
  current_shapes.resize(size);
  scores.resize(size);
  weights.resize(size);
  std::fill(scores.begin(), scores.end(), 0);
}
void DataSet::LoadNegativeDataSet(const string& negative) {
  neg_generator.Load(negative);
  imgs.clear();
  gt_shapes.clear();
  current_shapes.clear();
  size = 0;
  is_pos = false;
}
void DataSet::LoadDataSet(DataSet& pos, DataSet& neg) {
  const Config& c = Config::GetInstance();
  pos.LoadPositiveDataSet(c.train_pos_txt);
  neg.LoadNegativeDataSet(c.train_neg_txt);
  Mat_<double> mean_shape = pos.CalcMeanShape();
  // for current_shapes
  DataSet::RandomShapes(mean_shape, pos.current_shapes);
  // for negative generator
  neg.neg_generator.mean_shape = mean_shape;
}


NegGenerator::NegGenerator() {}
NegGenerator::~NegGenerator() {}

int NegGenerator::Generate(const JoinCascador& joincascador, int size, \
                           vector<Mat>& imgs, vector<double>& scores, \
                           vector<Mat_<double> >& shapes) {
  imgs.clear();
  scores.clear();
  shapes.clear();

  const int pool_size = Config::GetInstance().mining_pool_size;
  imgs.reserve(size + pool_size); // enough memory to overflow
  scores.reserve(size + pool_size);
  shapes.reserve(size + pool_size);

  vector<Mat> region_pool(pool_size);
  vector<double> score_pool(pool_size);
  vector<Mat_<double> > shape_pool(pool_size);
  vector<int> used(pool_size);

  const int size_o = size;
  double ratio = 0.9;
  while (size > 0) {
    // We generate a negative sample pool for validation
    for (int i = 0; i < pool_size; i++) {
      region_pool[i] = NextImage();
    }

    #pragma omp parallel for
    for (int i = 0; i < pool_size; i++) {
      bool is_face = joincascador.Validate(region_pool[i], score_pool[i], shape_pool[i]);
      if (is_face) used[i] = 1;
      else used[i] = 0;
    }

    // collect
    for (int i = 0; i < pool_size; i++) {
      if (used[i] > 0) {
        imgs.push_back(region_pool[i]);
        scores.push_back(score_pool[i]);
        shapes.push_back(shape_pool[i]);
        size--;
      }
    }

    if (size < ratio*size_o) {
      LOG("We have mined %d%%, bg image remains %d%%", int((1. - ratio + 1e-6) * 100), \
          int(100. - double(current_idx) / list.size() * 100.));
      ratio -= 0.1;
    }
  }
  return imgs.size();
}

Mat NegGenerator::NextImage() {
  const Config& c = Config::GetInstance();
  const int w = c.img_o_size;
  const int h = c.img_o_size;

  NextState();

  Mat region;
  Rect roi(x, y, w, h);
  region = img(roi).clone();

  switch (transform_type) {
  case ORIGIN:
    break;
  case ORIGIN_R:
    flip(region, region, 0);
    transpose(region, region);
    break;
  case ORIGIN_RR:
    flip(region, region, -1);
    break;
  case ORIGIN_RRR:
    flip(region, region, 1);
    transpose(region, region);
    break;
  case ORIGIN_FLIP:
    flip(region, region, 1);
    break;
  case ORIGIN_FLIP_R:
    flip(region, region, -1);
    transpose(region, region);
    break;
  case ORIGIN_FLIP_RR:
    flip(region, region, -1);
    flip(region, region, 1);
    break;
  case ORIGIN_FLIP_RRR:
    flip(region, region, 0);
    transpose(region, region);
    flip(region, region, 1);
    break;
  default:
    dieWithMsg("Unsupported Transform of Negative Sample");
    break;
  }

  return region;
}

void NegGenerator::NextState() {
  const Config& c = Config::GetInstance();
  const double scale_factor = c.scale_factor;
  const int x_step = c.x_step;
  const int y_step = c.y_step;
  const int w = c.img_o_size;
  const int h = c.img_o_size;
  const double scale = c.scale_factor;

  const int width = img.cols;
  const int height = img.rows;

  switch (transform_type) {
  case ORIGIN:
    transform_type = ORIGIN_R;
    return;
  case ORIGIN_R:
    transform_type = ORIGIN_RR;
    return;
  case ORIGIN_RR:
    transform_type = ORIGIN_RRR;
    return;
  case ORIGIN_RRR:
    transform_type = ORIGIN_FLIP;
    return;
  case ORIGIN_FLIP:
    transform_type = ORIGIN_FLIP_R;
    return;
  case ORIGIN_FLIP_R:
    transform_type = ORIGIN_FLIP_RR;
    return;
  case ORIGIN_FLIP_RR:
    transform_type = ORIGIN_FLIP_RRR;
    return;
  case ORIGIN_FLIP_RRR:
    transform_type = ORIGIN;
    break;
  default:
    dieWithMsg("Unsupported Transform of Negative Sample");
    break;
  }

  x += x_step; // move x
  if (x + w >= width) {
    x = 0;
    y += y_step; // move y
    if (y + h >= height) {
      x = y = 0;
      int width_ = int(img.cols * scale_factor);
      int height_ = int(img.rows * scale_factor);
      cv::resize(img, img, Size(width_, height_)); // scale image
      if (img.cols < w || img.rows < h) {
        // next image
        while (true) {
          current_idx++; // next image
          if (current_idx >= list.size()) {
            // Add background image list online
            LOG("Run out of Negative Samples! :-(");
            SaveTheWorld();
            continue;
          }
          //LOG("Use %d th Nega Image %s", current_idx + 1, list[current_idx].c_str());
          img = cv::imread(list[current_idx], CV_LOAD_IMAGE_GRAYSCALE);
          if (!img.data || img.cols <= w || img.rows <= h) {
            if (!img.data) {
              //LOG("Can not open image %s, Skip it", list[current_idx].c_str());
            }
            else {
              //LOG("Image %s is too small, Skip it", list[current_idx].c_str());
            }
          }
          else {
            // successfully get another background image
            break;
          }
        }
      }
    }
  }
}

void NegGenerator::Load(const string& path) {
  FILE* file = fopen(path.c_str(), "r");
  JDA_Assert(file, "Can not open negative dataset file list");

  char buff[256];
  list.clear();
  while (fscanf(file, "%s", buff) > 0) {
    list.push_back(buff);
  }
  std::random_shuffle(list.begin(), list.end());

  // initialize
  x = y = 0;
  current_idx = 0;
  transform_type = ORIGIN;
  img = cv::imread(list[current_idx], CV_LOAD_IMAGE_GRAYSCALE);
  if (!img.data) dieWithMsg("Can not open image, the path is %s", list[current_idx].c_str());
}

void NegGenerator::SaveTheWorld() {
  const Config& c = Config::GetInstance();
  c.joincascador->Snapshot();

  char buff1[256];
  char buff2[256];
  time_t t = time(NULL);
  strftime(buff1, sizeof(buff1), "We now save all hard negative samples under"
                                  "../data/hd/%Y%m%d-%H%M%S", localtime(&t));
  LOG(buff1);
  if (!EXISTS("../data/hd")) {
    MKDIR("../data/hd");
  }
  strftime(buff1, sizeof(buff1), "../data/hd/%Y%m%d-%H%M%S", localtime(&t));
  if (!EXISTS(buff1)) {
    MKDIR(buff1);
  }

  const int size = c.joincascador->neg->size;
  LOG("We have %d images to save", size);
  for (int i = 0; i < size; i++) {
    sprintf(buff2, "%s/%05d.jpg", buff1, i + 1);
    imwrite(buff2, c.joincascador->neg->imgs[i]);
  }

  char path[256];
  LOG("We need more background images, "
      "Please input a text file which contains background image list: ");
  scanf("%s", path);

  FILE* file = fopen(path, "r");
  while (!file) {
    LOG("Can not open %s, Please Check it!", path);
    LOG("Please input a text file below:");
    scanf("%s", path);
    file = fopen(path, "r");
  }

  list.clear();
  while (fscanf(file, "%s", path) > 0) {
    list.push_back(path);
  }
  std::random_shuffle(list.begin(), list.end());
  // reset current_idx
  current_idx = -1;
}

} // namespace jda
