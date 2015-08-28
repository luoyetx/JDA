#include <liblinear/linear.h>
#include "jda/common.hpp"
#include "jda/data.hpp"
#include "jda/cart.hpp"

using namespace cv;
using namespace std;

namespace jda {

BoostCart::BoostCart() {}
BoostCart::~BoostCart() {}
BoostCart::BoostCart(const BoostCart& other) {}
BoostCart& BoostCart::operator=(const BoostCart& other) {
    if (this == &other) return *this;
    return *this;
}

void BoostCart::Train(DataSet& pos, DataSet& neg) {
    assert(carts.size() == K);

    const Config& c = Config::GetInstance();
    const int landmark_n = c.landmark_n;
    RNG rng(getTickCount());
    // real boost
    for (int i = 0; i < K; i++) {
        Cart& cart = carts[i];
        pos.UpdateWeights();
        neg.UpdateWeights();
        int landmark_id = i % landmark_n;
        // train cart
        TIMER_BEGIN
            LOG("Train %th Cart", i);
            cart.Train(pos, neg);
            LOG("Done with %th Cart, costs %.4lf s", i, TIMER_NOW);
        TIMER_END
        // update score
        pos.UpdateScores(cart);
        neg.UpdateScores(cart);
        // select th for tp_rate and nf_rate
        double th1 = pos.CalcThresholdByRate(tp_rate);
        double th2 = neg.CalcThresholdByRate(fn_rate);
        // expect th2 < th < th1
        cart.th = (th2 < th1) ? rng.uniform(th2, th1) : th1;
        pos.Remove(cart.th);
        neg.Remove(cart.th);
        // **TODO** more neg if needed
    }
    GlobalRegression(pos);
    // **TODO** update shapes
}

/**
 * Fully Free Model from liblinear
 */
static inline void freeModel(struct model* model) {
    free(model->w);
    free(model->label);
    free(model);
}

void BoostCart::GlobalRegression(DataSet& pos) {
    Config& c = Config::GetInstance();
    const int landmark_n = c.landmark_n;
    const int n = pos.size; // size of dataset
    const int m = carts[0].leafNum * carts.size(); // true size of local binary feature
    const int f = m*(1 << (c.tree_depth - 1)); // full size of local binary feature
    Mat_<int> lbf(n, m);
    vector<int> idx;
    for (int i = 0; i < pos.size; i++) idx[i] = i;
    Mat_<double> shape_residual = pos.CalcShapeResidual(idx);
    // prepare linear regression X, Y
    struct feature_node** X = (struct feature_node**)malloc(n*sizeof(struct feature_node *));
    double** Y = (double**)malloc(2 * landmark_n*sizeof(double *));
    for (int i = 0; i < n; i++) {
        X[i] = (struct feature_node *)malloc((m + 1)*sizeof(struct feature_node));
        for (int j = 0; j < m; j++) {
            X[i][j].index = lbf(i, j) + 1; // index starts from 1
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

} // namespace jda
