#include <cstdio>
#include "jda/jda.hpp"

using namespace cv;
using namespace std;

namespace jda {

struct jdaCascador::jdaCart {
    // raw memory points ?
    vector<int> scales;
    vector<int> landmark_id1;
    vector<int> landmark_id2;
    vector<double> offset1_x;
    vector<double> offset1_y;
    vector<double> offset2_x;
    vector<double> offset2_y;
    vector<double> thresholds;
    vector<double> scores;
    double th;
};

void jdaCascador::SerializeFrom(FILE* fd) {
    int YO;
    fread(&YO, sizeof(YO), 1, fd);
    fread(&T, sizeof(int), 1, fd);
    fread(&K, sizeof(int), 1, fd);
    fread(&landmark_n, sizeof(int), 1, fd);
    fread(&depth, sizeof(int), 1, fd);
    mean_shape.create(1, 2 * landmark_n);
    fread(mean_shape.ptr<double>(0), sizeof(double), mean_shape.cols, fd);

    T++; // training stage rang in [0, c.T), we should plus one
    node_n = 1 << depth; // nodes
    leaf_n = 1 << (depth - 1); // leaves
    //int non_leaf_n = leaf_n;
    carts.resize(T*K);
    ws.resize(T);
    const int w_rows = 2 * landmark_n;
    const int w_cols = leaf_n*K;
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < K; j++) {
            jdaCart& cart = carts[i*K + j];
            // non-leaf, index start from 1, 0 will never be used
            cart.scales.resize(leaf_n);
            cart.landmark_id1.resize(leaf_n);
            cart.landmark_id2.resize(leaf_n);
            cart.offset1_x.resize(leaf_n);
            cart.offset1_y.resize(leaf_n);
            cart.offset2_x.resize(leaf_n);
            cart.offset2_y.resize(leaf_n);
            cart.thresholds.resize(leaf_n);
            for (int q = 1; q < leaf_n; q++) {
                fread(&cart.scales[q], sizeof(int), 1, fd);
                fread(&cart.landmark_id1[q], sizeof(int), 1, fd);
                fread(&cart.landmark_id2[q], sizeof(int), 1, fd);
                fread(&cart.offset1_x[q], sizeof(double), 1, fd);
                fread(&cart.offset1_y[q], sizeof(double), 1, fd);
                fread(&cart.offset2_x[q], sizeof(double), 1, fd);
                fread(&cart.offset2_y[q], sizeof(double), 1, fd);
                fread(&cart.thresholds[q], sizeof(int), 1, fd);
            }
            // leaf
            cart.scores.resize(leaf_n);
            for (int q = 0; q < leaf_n; q++) {
                fread(&cart.scores[q], sizeof(double), 1, fd);
            }
            fread(&cart.th, sizeof(double), 1, fd);
        }
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
