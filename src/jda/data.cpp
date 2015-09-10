#include <cmath>
#include <cassert>
#include <opencv2/highgui/highgui.hpp>
#include "jda/jda.hpp"

using namespace cv;
using namespace std;

namespace jda {

DataSet::DataSet() {}
DataSet::~DataSet() {}
DataSet::DataSet(const DataSet& other) {}
DataSet& DataSet::operator=(const DataSet& other) {
    if (this == &other) return *this;
    return *this;
}

Mat_<int> DataSet::CalcFeatureValues(vector<Feature>& feature_pool, vector<int>& idx) {
    const int n = feature_pool.size();
    const int m = idx.size();
    Mat_<int> features(n, m);

    for (int i = 0; i < n; i++) {
        const Feature& feature = feature_pool[i];
        int* ptr = features.ptr<int>(i);
        for (int j = 0; j < m; j++) {
            const Mat& img = imgs[idx[j]];
            const Mat_<double>& shape = current_shapes[idx[j]];
            const int width = img.cols - 1;
            const int height = img.rows - 1;

            double x1, y1, x2, y2, scale;
            switch (feature.scale) {
            case Feature::ORIGIN:
                scale = 1.0; break;
            case Feature::HALF:
                scale = 0.5; break;
            case Feature::QUARTER:
                scale = 0.25; break;
            default:
                scale = 1.0; break;
            }
            x1 = shape(0, 2 * feature.landmark_id1) + scale*feature.offset1_x;
            y1 = shape(0, 2 * feature.landmark_id1 + 1) + scale*feature.offset1_y;
            x2 = shape(0, 2 * feature.landmark_id2) + scale*feature.offset2_x;
            y2 = shape(0, 2 * feature.landmark_id2 + 1) + scale*feature.offset2_y;

            checkBoundaryOfImage(width, height, x1, y1);
            checkBoundaryOfImage(width, height, x2, y2);

            const int x1_ = int(round(x1));
            const int y1_ = int(round(y1));
            const int x2_ = int(round(x2));
            const int y2_ = int(round(y2));

            ptr[j] = img.at<uchar>(x1_, y1_) - img.at<uchar>(x2_, y2_);
        }
    }

    return features;
}

Mat_<double> DataSet::CalcShapeResidual(vector<int>& idx) {
    assert(is_pos == true);
    Mat_<double> shape_residual;
    const int n = idx.size();
    // all landmark
    const int landmark_n = gt_shapes[0].cols / 2;
    shape_residual.create(n, landmark_n * 2);
    for (int i = 0; i < n; i++) {
        shape_residual.row(i) = gt_shapes[idx[i]] - current_shapes[idx[i]];
    }
    return shape_residual;
}
Mat_<double> DataSet::CalcShapeResidual(vector<int>& idx, int landmark_id) {
    assert(is_pos == true);
    Mat_<double> shape_residual;
    const int n = idx.size();
    // specific landmark
    shape_residual.create(n, 2);
    for (int i = 0; i < n; i++) {
        shape_residual(i, 0) = gt_shapes[idx[i]](0, 2 * landmark_id) - \
            current_shapes[idx[i]](0, 2 * landmark_id);
        shape_residual(i, 1) = gt_shapes[idx[i]](0, 2 * landmark_id + 1) - \
            current_shapes[idx[i]](0, 2 * landmark_id + 1);
    }
    return shape_residual;
}

Mat_<double> DataSet::CalcMeanShape() {
    Mat_<double> mean_shape = gt_shapes[0].clone();
    const int n = gt_shapes.size();
    for (int i = 1; i < n; i++) {
        mean_shape += gt_shapes[i];
    }
    mean_shape /= n;
    return mean_shape;
}

void DataSet::RandomShapes(Mat_<double>& mean_shape, vector<Mat_<double> >& shapes) {
    // **TODO** random perturbation on mean_shapes
    const int n = shapes.size();
    for (int i = 0; i < n; i++) {

    }
}

void DataSet::UpdateWeights() {
    const double flag = -(is_pos ? 1 : -1);
    double sum_w = 0;
    for (int i = 0; i < size; i++) {
        weights[i] = exp(flag*scores[i]);
        sum_w += weights[i];
    }
    // normalize to 1
    for (int i = 0; i < size; i++) {
        weights[i] /= sum_w;
    }
}

void DataSet::UpdateScores(Cart& cart) {
    for (int i = 0; i < size; i++) {
        Mat& img = imgs[i];
        Mat_<double>& shape = current_shapes[i];
        int leaf_node_idx = cart.Forward(img, shape);
        scores[i] += cart.scores[leaf_node_idx];
    }
    is_sorted = false;
}

double DataSet::CalcThresholdByRate(double rate) {
    if (!is_sorted) QSort();
    int offset = int(rate*size);
    return scores[offset];
}

void DataSet::Remove(double th) {
    if (!is_sorted) QSort();
    int offset = 0;
    const int upper = scores.size();
    // get offset
    while (offset < upper && scores[offset] < th) offset++;
    imgs.resize(offset);
    gt_shapes.resize(offset);
    current_shapes.resize(offset);
    scores.resize(offset);
    weights.resize(offset);
    // renew size
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
        while (scores[i] < t) i++;
        while (scores[j] > t) j--;
        if (i <= j) {
            // swap data point
            std::swap(imgs[i], imgs[j]);
            std::swap(gt_shapes[i], gt_shapes[j]);
            std::swap(current_shapes[i], current_shapes[j]);
            std::swap(scores[i], scores[j]);
            std::swap(weights[i], weights[j]);
            i++; j--;
        }
    } while (i <= j);
    if (left < j) _QSort_(left, j);
    if (i < right) _QSort_(i, right);
}

void DataSet::MoreNegSamples(int stage, int size) {
    assert(is_pos == false);
    // **TODO** calculate size_ for size of negative samples to generate
    int size_ = 0;
    vector<Mat> imgs_;
    vector<double> scores_;
    vector<Mat_<double> > shapes_;
    const int extra_size = neg_generator.Generate(*joincascador, size_, stage, \
                                                  imgs_, scores_, shapes_);
    const int expanded = extra_size+ imgs_.size();
    imgs.reserve(expanded);
    current_shapes.reserve(expanded);
    scores.reserve(expanded);
    weights.reserve(expanded);
    for (int i = 0; i < extra_size; i++) {
        imgs.push_back(imgs_[i]);
        current_shapes.push_back(shapes_[i]);
        scores.push_back(scores_[i]);
        weights.push_back(0); // all weights will be updated by calling `UpdataWeights`
    }
}

void DataSet::LoadPositiveDataSet(const string& positive) {
    const Config& c = Config::GetInstance();
    const int landmark_n = c.landmark_n;
    FILE* file = fopen(positive.c_str(), "r");
    assert(file);

    char buff[300];
    imgs.clear();
    gt_shapes.clear();
    while (fscanf(file, "%s", buff) > 0) {
        Mat_<double> shape(1, 2 * landmark_n);
        const double* ptr = shape.ptr<double>(0);
        for (int i = 0; i < 2 * landmark_n; i++) {
            fscanf(file, "%lf", ptr + i);
        }
        Mat img = imread(buff, CV_LOAD_IMAGE_GRAYSCALE);
        imgs.push_back(img);
        gt_shapes.push_back(shape);
    }
    size = imgs.size();
    is_pos = true;
    current_shapes.resize(size);

    fclose(file);
}
void DataSet::LoadNegativeDataSet(const string& negative) {
    neg_generator.SetOriginList(negative);
    imgs.clear();
    gt_shapes.clear();
    current_shapes.clear();
    size = 0;
    is_pos = false;
}
void DataSet::LoadDataSet(DataSet& pos, DataSet& neg) {
    const Config& c = Config::GetInstance();
    pos.LoadPositiveDataSet(c.positive_dataset);
    neg.LoadNegativeDataSet(c.negative_dataset);
    Mat_<double> mean_shape = pos.CalcMeanShape();
    // for current_shapes
    DataSet::RandomShapes(mean_shape, pos.current_shapes);
    // for negative generator
    neg.neg_generator.mean_shape = mean_shape;
    // **TODO** calculate size_ for size of negative samples to generate
    const int size = 0;
    neg.MoreNegSamples(0, size);
}


NegGenerator::NegGenerator() {}
NegGenerator::~NegGenerator() {}
NegGenerator::NegGenerator(const NegGenerator& other) {}
NegGenerator& NegGenerator::operator=(const NegGenerator& other) {
    if (this == &other) return *this;
    return *this;
}

int NegGenerator::Generate(JoinCascador& joincascador, int size, int stage, \
                           vector<Mat>& imgs, vector<double>& scores, \
                           vector<Mat_<double> >& shapes) {
    // **TODO** generate negative samples with a strategy
    return 0;
}

void NegGenerator::SetOriginList(const string& path) {
    FILE* file = fopen(path.c_str(), "r");
    assert(file);

    char buff[300];
    list.clear();
    while (fscanf(file, "%s", buff) > 0) {
        list.push_back(buff);
    }
}

} // namespace jda
