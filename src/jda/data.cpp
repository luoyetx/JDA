#include "jda/data.hpp"
#include "jda/cart.hpp"

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
    // **TODO** calculate feature values
    return Mat();
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
        Mat& shape = current_shapes[i];
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
