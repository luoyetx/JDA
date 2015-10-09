#include <liblinear/linear.h>
#include "jda/jda.hpp"

#include <iostream>

using namespace cv;
using namespace std;

namespace jda {

BoostCart::BoostCart() {}
BoostCart::~BoostCart() {}
//BoostCart::BoostCart(const BoostCart& other) {}
//BoostCart& BoostCart::operator=(const BoostCart& other) {
//    if (this == &other) return *this;
//    return *this;
//}
void BoostCart::Initialize(int stage) {
    const Config& c = Config::GetInstance();
    this->stage = stage;
    K = c.K;
    carts.resize(K);
    const int landmark_n = c.landmark_n;
    const int m = K*(1 << c.tree_depth);
    w = Mat_<double>(2 * landmark_n, m);
}

void BoostCart::Train(DataSet& pos, DataSet& neg) {
    assert(carts.size() == K);

    const Config& c = Config::GetInstance();
    const int landmark_n = c.landmark_n;
    RNG rng(getTickCount());
    // Real Boost
    for (int k = 0; k < K; k++) {
        Cart& cart = carts[k];
        // more neg if needed
        neg.MoreNegSamples(pos.size);
        // update weights
        pos.UpdateWeights();
        neg.UpdateWeights();
        int landmark_id = k % landmark_n;
        cart.Initialize(stage, landmark_id);
        // train cart
        TIMER_BEGIN
            LOG("Train %d th Cart", k);
            cart.Train(pos, neg);
            LOG("Done with %d th Cart, costs %.4lf s", k, TIMER_NOW);
        TIMER_END
        joincascador->current_cart_idx = k;
        // update score
        pos.UpdateScores(cart);
        neg.UpdateScores(cart);
        // select th for tp_rate and nf_rate
        double th1 = pos.CalcThresholdByRate(1 - c.accept_rate);
        double th2 = neg.CalcThresholdByRate(c.reject_rate);
        // expect th2 < th < th1
        cart.th = (th2 < th1) ? rng.uniform(th2, th1) : th1;
        int pos_n = pos.size;
        int neg_n = neg.size;
        pos.Remove(cart.th);
        neg.Remove(cart.th);
        double pos_drop_rate = double(pos_n - pos.size) / double(pos_n) * 100.;
        double neg_drop_rate = double(neg_n - neg.size) / double(neg_n) * 100.;
        LOG("Pos drop rate = %.2lf%%, Neg drop rate = %.2lf%%", pos_drop_rate, neg_drop_rate);
        LOG("Current Positive DataSet Size is %d", pos.size);
    }
    // Global Regression with LBF
    // generate lbf
    const int pos_n = pos.size;
    const int neg_n = neg.size;
    const int m = K;
    vector<Mat_<int> > pos_lbf(pos_n);
    vector<Mat_<int> > neg_lbf(neg_n);
    for (int i = 0; i < pos_n; i++) {
        pos_lbf[i] = GenLBF(pos.imgs[i], pos.current_shapes[i]);
    }
    for (int i = 0; i < neg_n; i++) {
        neg_lbf[i] = GenLBF(neg.imgs[i], neg.current_shapes[i]);
    }
    // regression
    vector<int> pos_idx(pos.size);
    for (int i = 0; i < pos.size; i++) pos_idx[i] = i;
    Mat_<double> shape_residual = pos.CalcShapeResidual(pos_idx);
    GlobalRegression(pos_lbf, shape_residual);
    // update shapes
    for (int i = 0; i < pos_n; i++) {
        pos.current_shapes[i] += GenDeltaShape(pos_lbf[i]);
    }
    for (int i = 0; i < neg_n; i++) {
        neg.current_shapes[i] += GenDeltaShape(neg_lbf[i]);
    }
    // regression error
    double e = calcMeanError(pos.gt_shapes, pos.current_shapes);
    LOG("Regression Mean Error = %.4lf", e);
    // Done
}

/**
 * Fully Free Model from liblinear
 */
static inline void freeModel(struct model* model) {
    free(model->w);
    free(model->label);
    free(model);
}

void BoostCart::GlobalRegression(const vector<Mat_<int> >& lbf, const Mat_<double>& shape_residual) {
    Config& c = Config::GetInstance();
    const int landmark_n = c.landmark_n;
    const int n = lbf.size();
    const int m = K; // true size of local binary feature
    const int f = m*carts[0].leafNum; // full size of local binary feature
    vector<int> idx;
    // prepare linear regression X, Y
    struct feature_node** X = (struct feature_node**)malloc(n*sizeof(struct feature_node *));
    double** Y = (double**)malloc(2 * landmark_n*sizeof(double *));
    for (int i = 0; i < n; i++) {
        X[i] = (struct feature_node *)malloc((m + 1)*sizeof(struct feature_node));
        for (int j = 0; j < m; j++) {
            X[i][j].index = lbf[i](0, j) + 1; // index starts from 1
            X[i][j].value = 1;
        }
        X[i][m].index = X[i][m].value = -1;
    }
    for (int i = 0; i < landmark_n; i++) {
        Y[2 * i] = (double*)malloc(n*sizeof(double));
        Y[2 * i + 1] = (double*)malloc(n*sizeof(double));
        for (int j = 0; j < n; j++) {
            Y[2 * i][j] = shape_residual(j, 2 * i);
            Y[2 * i + 1][j] = shape_residual(j, 2 * i + 1);
        }
    }
    // train every landmark
    struct problem prob;
    struct parameter param;
    prob.l = n;
    prob.n = f;
    prob.x = X;
    prob.bias = -1;
    param.solver_type = L2R_L2LOSS_SVR_DUAL;
    param.C = 1. / n;
    param.p = 0;
    param.eps = 0.0001;

    //#pragma omp parallel for
    for (int i = 0; i < landmark_n; i++) {
        struct problem prob_ = prob;
        prob_.y = Y[2 * i];
        check_parameter(&prob_, &param);
        struct model *model = train(&prob_, &param);
        for (int j = 0; j < f; j++) w(2 * i, j) = get_decfun_coef(model, j + 1, 0);
        freeModel(model);

        prob_.y = Y[2 * i + 1];
        check_parameter(&prob_, &param);
        model = train(&prob_, &param);
        for (int j = 0; j < f; j++) w(2 * i + 1, j) = get_decfun_coef(model, j + 1, 0);
        freeModel(model);
    }
    // free
    for (int i = 0; i < n; i++) free(X[i]);
    for (int i = 0; i < 2 * landmark_n; i++) free(Y[i]);
    free(X);
    free(Y);
}

Mat_<int> BoostCart::GenLBF(const Mat& img, const Mat_<double>& shape) const {
    Mat_<int> lbf(1, K);
    int* ptr = lbf.ptr<int>(0);
    const int base = carts[0].leafNum;
    int offset = 0;
    for (int k = 0; k < K; k++) {
        ptr[k] = offset + carts[k].Forward(img, shape);
        offset += base;
    }
    return lbf;
}

Mat_<double> BoostCart::GenDeltaShape(const Mat_<int>& lbf) const {
    const int landmark_n = w.rows / 2;
    const int m = lbf.cols;
    Mat_<double> delta_shape(1, 2 * landmark_n);
    const double* w_ptr;
    const double* ds_ptr;
    const int* lbf_ptr = lbf.ptr<int>(0);
    for (int i = 0; i < landmark_n; i++) {
        w_ptr = w.ptr<double>(2 * i);
        double y = 0;
        for (int j = 0; j < m; j++) y += w_ptr[lbf_ptr[j]];
        delta_shape(0, 2 * i) = y;

        w_ptr = w.ptr<double>(2 * i + 1);
        y = 0;
        for (int j = 0; j < m; j++) y += w_ptr[lbf_ptr[j]];
        delta_shape(0, 2 * i + 1) = y;
    }
    return delta_shape;
}

} // namespace jda
