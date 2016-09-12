#include <cstdio>
#include <cstdlib>
#include <liblinear/linear.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/data.hpp"
#include "jda/cart.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;

namespace jda {

/*!
 * \brief draw the distribution of scores
 * \note  scores should be in order
 */
static void draw_density_graph(vector<double>& pos_scores, vector<double>& neg_scores, \
                               const int n = 100, const int rows = 20) {
  JDA_Assert(n < 115, "number of bins should be less than 150");
  JDA_Assert(rows < 100, "graph rows should be less than 100");
  double s_max = max(pos_scores[0], neg_scores[0]);
  int pos_size = pos_scores.size();
  int neg_size = neg_scores.size();
  double s_min = min(pos_scores[pos_size - 1], neg_scores[neg_size - 1]);
  vector<int> pos_bin(n, 0);
  vector<int> neg_bin(n, 0);
  double delta = (s_max - s_min) / n + 1e-9;
  double th = s_max - delta;
  // calc bin
  int bin_idx = 0;
  for (int i = 0; i < pos_size; i++) {
    if (pos_scores[i] >= th) {
      pos_bin[bin_idx]++;
    }
    else {
      i--;
      bin_idx++;
      th -= delta;
    }
  }
  th = s_max - delta;
  bin_idx = 0;
  for (int i = 0; i < neg_size; i++) {
    if (neg_scores[i] >= th) {
      neg_bin[bin_idx]++;
    }
    else {
      i--;
      bin_idx++;
      th -= delta;
    }
  }
  // enlargre local detail
  double max_bin_rate = 0;
  double min_bin_rate = 1;
  for (int i = 0; i < n; i++) {
    if (pos_bin[i] > 0) {
      double bin_rate = pos_bin[i] / double(pos_size);
      if (bin_rate > max_bin_rate) max_bin_rate = bin_rate;
      if (bin_rate <  min_bin_rate) min_bin_rate = bin_rate;
    }
    if (neg_bin[i] > 0) {
      double bin_rate = neg_bin[i] / double(neg_size);
      if (bin_rate > max_bin_rate) max_bin_rate = bin_rate;
      if (bin_rate < min_bin_rate) min_bin_rate = bin_rate;
    }
  }
  max_bin_rate += 1e-5;
  min_bin_rate -= 1e-5;
  double range = max_bin_rate - min_bin_rate + 1e-18;
  // draw graph
  int graph[100][120] = { 0 };
  for (int i = 0; i < n; i++) {
    if (pos_bin[i]>0) {
      int density = int((pos_bin[i] / double(pos_size) - min_bin_rate) / range * rows);
      graph[density][i] += 1;
    }
    if (neg_bin[i] > 0) {
      int density = int((neg_bin[i] / double(neg_size) - min_bin_rate) / range * rows);
      graph[density][i] += 2;
    }
  }
  // render graph
  for (int c = 0; c < n + 8; c++) {
    printf("=");
  }
  printf("\n");
  char var[5] = { ' ', '+', 'x', '*' };
  for (int r = rows - 1; r >= 0; r--) {
    printf("%06.2lf%% ", (double(r + 1) / rows * range + min_bin_rate) * 100);
    for (int c = 0; c < n; c++) {
      printf("%c", var[graph[r][c]]);
    }
    printf("\n");
  }
  for (int c = 0; c < n + 8; c++) {
    printf("=");
  }
  printf("\n");
}

BoostCart::BoostCart(int stage) {
  const Config& c = Config::GetInstance();
  this->stage = stage;
  K = c.K;
  carts.reserve(K);
  for (int i = 0; i < K; i++) {
    // distribute the landmarks
    carts.push_back(Cart(stage, i%c.landmark_n));
  }
  const int landmark_n = c.landmark_n;
  const int m = K*(1 << (c.tree_depth - 1)); // K * leafNume
  w = Mat_<double>::zeros(m, 2 * landmark_n);
}
BoostCart::~BoostCart() {
}

void BoostCart::Train(DataSet& pos, DataSet& neg) {
  Config& c = Config::GetInstance();
  JoinCascador& joincascador = *c.joincascador;

  // statistic parameters
  const int pos_original_size = pos.size;
  const int neg_original_size = int(pos_original_size * c.nps[stage]);
  int neg_rejected = 0;

  const int landmark_n = c.landmark_n;
  const int normalization_step = landmark_n*c.score_normalization_steps[stage];
  RNG& rng = c.rng_pool[0];
  //int drop_n = (1. - c.recall[stage])*pos.size / K; // pos drop number per cart
  //if (drop_n <= 1) drop_n = 1;
  int drop_n = c.drops[stage];

  const int start_of_cart = joincascador.current_cart_idx + 1;
  int restarts = 0;
  double best_drop_rate = 0.;
  Cart best_cart = carts[0];

  // Real Boost

  // if neg.size < neg_th, mining starts
  int current_stage_idx = c.joincascador->current_stage_idx;
  int neg_th = int(pos.size*c.nps[current_stage_idx] * c.mining_th[current_stage_idx]);
  for (int k = start_of_cart; k < K; k++) {
    const int kk = k + 1;
    Cart& cart = carts[k];
    if (neg.size < neg_th) {
      neg.MoreNegSamples(pos.size, c.nps[stage]);
      neg_th = int(neg.size * c.mining_th[current_stage_idx]); // update neg_th
    }
    // print out data set status
    pos.QSort(); neg.QSort();
    LOG("Pos max score = %.4lf, min score = %.4lf", pos.scores[0], pos.scores[pos.size - 1]);
    LOG("Neg max score = %.4lf, min score = %.4lf", neg.scores[0], neg.scores[neg.size - 1]);
    // draw scores desity graph
    draw_density_graph(pos.scores, neg.scores);
    // update weights
    DataSet::UpdateWeights(pos, neg);
    LOG("Current Positive DataSet Size is %d", pos.size);
    LOG("Current Negative DataSet Size is %d", neg.size);
    // train cart
    TIMER_BEGIN
      LOG("Train %d th Cart", k + 1);
      cart.Train(pos, neg);
      LOG("Done with %d th Cart, costs %.4lf s", k + 1, TIMER_NOW);
    TIMER_END
    joincascador.current_cart_idx = k;
    // update score and last_score
    pos.UpdateScores(cart);
    neg.UpdateScores(cart);
    if (kk % normalization_step == 0) {
      DataSet::CalcMeanAndStd(pos, neg, cart.mean, cart.std);
      pos.ApplyMeanAndStd(cart.mean, cart.std);
      neg.ApplyMeanAndStd(cart.mean, cart.std);
    }
    else {
      cart.mean = 0.;
      cart.std = 1.;
    }
    // select th for pre-defined recall
    pos.QSort();
    neg.QSort();
    cart.th = pos.CalcThresholdByNumber(drop_n);
    int pos_n = pos.size;
    int neg_n = neg.size;
    int will_removed = neg.PreRemove(cart.th);
    double tmp_drop_rate = double(will_removed) / neg_n;
    int number_of_carts = joincascador.current_stage_idx*joincascador.K + joincascador.current_cart_idx;
    if (c.restart_on && tmp_drop_rate < c.restart_th[joincascador.current_stage_idx] && number_of_carts > 10) {
      restarts++;
      LOG("***** Drop %d, Drop rate neg is %.4lf%%, Restart current Cart *****", will_removed, tmp_drop_rate*100.);
      LOG("***** Restart Time: %d *****", restarts);
      LOG("Current trained Cart below");
      cart.PrintSelf();

      // compare with best cart for now
      if (tmp_drop_rate > best_drop_rate) {
        best_drop_rate = tmp_drop_rate;
        best_cart = cart;
      }
      // select the best cart for this cart
      if (restarts >= c.restart_times) {
        LOG("***** Select a cart which give us %.4lf%% drop rate *****", best_drop_rate*100.);
        cart = best_cart;
        best_drop_rate = 0.;
        pos.ResetScores();
        neg.ResetScores();
        pos.UpdateScores(cart);
        neg.UpdateScores(cart);
        if (kk % normalization_step == 0) {
          DataSet::CalcMeanAndStd(pos, neg, cart.mean, cart.std);
          pos.ApplyMeanAndStd(cart.mean, cart.std);
          neg.ApplyMeanAndStd(cart.mean, cart.std);
        }
        else {
          cart.mean = 0.;
          cart.std = 1.;
        }
        pos.QSort();
        neg.QSort();
        //JDA_Assert(cart.th == pos.CalcThresholdByNumber(1), "restart error");
      }
      else {
        // recover data scores
        pos.ResetScores();
        neg.ResetScores();
        k--;
        continue;
      }
    }

    // update restart parameters
    best_drop_rate = 0.;
    restarts = 0;

    pos.Remove(cart.th);
    neg.Remove(cart.th);

    // print cart info
    cart.PrintSelf();
    if ((kk != K) && (kk%c.snapshot_iter == 0)) { // snapshot model and data
      DataSet::Snapshot(pos, neg);
      c.joincascador->Snapshot();
    }

    int pos_drop = pos_n - pos.size;
    int neg_drop = neg_n - neg.size;
    double pos_drop_rate = double(pos_drop) / double(pos_n)* 100.;
    double neg_drop_rate = double(neg_drop) / double(neg_n)* 100.;
    LOG("Pos drop = %d, Neg drop = %d, drop rate = %.2lf%%", pos_drop, neg_drop, neg_drop_rate);
    neg_rejected += neg_n - neg.size;
  }
  // Global Regression with LBF
  // generate lbf
  const int pos_n = pos.size;
  const int neg_n = neg.size;
  LOG("Generate LBF of DataSet");
  vector<Mat_<int> > pos_lbf(pos_n);
  vector<Mat_<int> > neg_lbf(neg_n);

  #pragma omp parallel for
  for (int i = 0; i < pos_n; i++) {
    pos_lbf[i] = GenLBF(pos.imgs[i], pos.current_shapes[i]);
  }
  #pragma omp parallel for
  for (int i = 0; i < neg_n; i++) {
    neg_lbf[i] = GenLBF(neg.imgs[i], neg.current_shapes[i]);
  }

  // regression, use valid face which has gt_shape
  vector<int> valid_pos_idx;
  vector<Mat_<int> > valid_pos_lbf;
  for (int i = 0; i < pos.size; i++) {
    if (pos.HasGtShape(i)) {
      valid_pos_idx.push_back(i);
      valid_pos_lbf.push_back(pos_lbf[i]);
    }
  }
  Mat_<double> shape_residual_valid = pos.CalcShapeResidual(valid_pos_idx);
  LOG("Start Global Regression");
  GlobalRegression(valid_pos_lbf, shape_residual_valid);
  // update shapes
  #pragma omp parallel for
  for (int i = 0; i < pos_n; i++) {
    pos.current_shapes[i] += GenDeltaShape(pos_lbf[i], pos.stp_mc[i]);
  }
  #pragma omp parallel for
  for (int i = 0; i < neg_n; i++) {
    neg.current_shapes[i] += GenDeltaShape(neg_lbf[i], neg.stp_mc[i]);
  }

  // summary
  LOG("====================");
  LOG("|      Summary     |");
  LOG("====================");
  // regression error
  vector<Mat_<double> > valid_gt_shapes;
  vector<Mat_<double> > valid_current_shapes;
  for (int i = 0; i < pos.size; i++) {
    if (pos.HasGtShape(i)) {
      valid_gt_shapes.push_back(pos.gt_shapes[i]);
      valid_current_shapes.push_back(pos.current_shapes[i]);
    }
  }
  double e = calcMeanError(valid_gt_shapes, valid_current_shapes);
  LOG("Regression Mean Error = %.4lf", e);

  // accept and reject rate
  double accept_rate = 0.;
  double reject_rate = 0.;
  accept_rate = double(pos_n) / double(pos_original_size) * 100.;
  reject_rate = double(neg_rejected) / double(neg_rejected + neg_original_size) * 100.;
  LOG("Accept Rate = %.2lf%%", accept_rate);
  // Done
}

/*!
 * \brief Fully Free Model from liblinear
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
  struct feature_node** X = (struct feature_node**)malloc(n*sizeof(struct feature_node*));
  double** Y = (double**)malloc(2 * landmark_n*sizeof(double*));
  for (int i = 0; i < n; i++) {
    X[i] = (struct feature_node*)malloc((m + 1)*sizeof(struct feature_node));
    for (int j = 0; j < m; j++) {
      X[i][j].index = lbf[i](0, j) + 1; // index starts from 1
      X[i][j].value = 1.;
    }
    X[i][m].index = -1;
    X[i][m].value = -1.;
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

  #pragma omp parallel for
  for (int i = 0; i < landmark_n; i++) {
    struct problem prob_ = prob;
    prob_.y = Y[2 * i];
    check_parameter(&prob_, &param);
    struct model *model = train(&prob_, &param);
    for (int j = 0; j < f; j++) w(j, 2 * i) = get_decfun_coef(model, j + 1, 0);
    freeModel(model);

    prob_.y = Y[2 * i + 1];
    check_parameter(&prob_, &param);
    model = train(&prob_, &param);
    for (int j = 0; j < f; j++) w(j, 2 * i + 1) = get_decfun_coef(model, j + 1, 0);
    freeModel(model);
  }

  // free
  for (int i = 0; i < n; i++) free(X[i]);
  for (int i = 0; i < 2 * landmark_n; i++) free(Y[i]);
  free(X);
  free(Y);
}

Mat_<int> BoostCart::GenLBF(const Mat& img, const Mat_<double>& shape) const {
  const Config& c = Config::GetInstance();
  Mat_<int> lbf(1, K);
  int* ptr = lbf.ptr<int>(0);
  const int base = carts[0].leafNum;
  int offset = 0;
  Mat img_h, img_q;
  cv::resize(img, img_h, Size(c.img_h_size, c.img_h_size));
  cv::resize(img, img_q, Size(c.img_q_size, c.img_q_size));
  STParameter stp_mc = STParameter::Calc(shape, c.joincascador->mean_shape);
  for (int k = 0; k < K; k++) {
    ptr[k] = offset + carts[k].Forward(img, img_h, img_q, shape, stp_mc);
    offset += base;
  }
  return lbf;
}

Mat_<double> BoostCart::GenDeltaShape(const Mat_<int>& lbf, const STParameter& stp_mc) const {
  const int landmark_dim = w.cols;
  const int m = lbf.cols;
  Mat_<double> delta_shape = Mat_<double>::zeros(1, landmark_dim);

  double* ds_ptr = delta_shape.ptr<double>(0);
  const int* lbf_ptr = lbf.ptr<int>(0);
  for (int i = 0; i < m; i++) {
    const double* w_ptr = w.ptr<double>(lbf_ptr[i]);
    for (int j = 0; j < landmark_dim; j += 2) {
      ds_ptr[j] += w_ptr[j];
      ds_ptr[j + 1] += w_ptr[j + 1];
    }
  }

  stp_mc.Apply(delta_shape, delta_shape);
  return delta_shape;
}

} // namespace jda
