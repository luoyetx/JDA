#include <cmath>
#include "jda/data.hpp"
#include "jda/cart.hpp"
#include "jda/common.hpp"

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

Mat DataSet::CalcFeatureValues(vector<Feature>& feature_pool, vector<int>& idx) {
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

            ptr[j] = img.at<uchar>(x1_, y1_) - img.at<uchar>(x2, y2);
        }
    }

    return features;
}

Mat DataSet::CalcShapeResidual(vector<int>& idx, int landmark_id) {
    assert(is_pos == true);
    Mat_<double> shape_residual;
    const int n = idx.size();
    if (landmark_id < 0) {
        // all landmark
        const int landmark_n = gt_shapes[0].cols / 2;
        shape_residual.create(n, landmark_n * 2);
        for (int i = 0; i < n; i++) {
            shape_residual.row(i) = gt_shapes[idx[i]] - current_shapes[idx[i]];
        }
    }
    else {
        // specific landmark
        shape_residual.create(n, 2);
        for (int i = 0; i < n; i++) {
            shape_residual(i, 0) = gt_shapes[idx[i]](0, 2 * landmark_id) - \
                                   current_shapes[idx[i]](0, 2 * landmark_id);
            shape_residual(i, 1) = gt_shapes[idx[i]](0, 2 * landmark_id + 1) - \
                                   current_shapes[idx[i]](0, 2 * landmark_id + 1);
        }
    }
    return shape_residual;
}

void DataSet::UpdateWeights() {
    const double flag = -(is_pos ? 1 : -1);
    for (int i = 0; i < size; i++) {
        weights[i] = exp(flag*scores[i]);
    }
}

void DataSet::UpdateScores(Cart& cart) {
    for (int i = 0; i < size; i++) {
        Mat& img = imgs[i];
        Mat_<double>& shape = current_shapes[i];
        int leaf_node_idx = cart.Forward(img, shape);
        scores[i] += cart.scores[leaf_node_idx];
    }
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

} // namespace jda
