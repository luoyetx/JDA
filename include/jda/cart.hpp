#ifndef CART_HPP_
#define CART_HPP_

#include <opencv2/core/core.hpp>

namespace jda {

// forward declaration
class DataSet;
class Feature;
class JoinCascador;

/*!
 * \breif Classification and Regression Random Tree
 *  see more detail on paper in section 4 about `CR_k^t`
 *  We organize the nodes in sequence, the index is started from 1
 *  the structure is shown below
 *          root(idx = i)
 *          |           |
 *  left(idx = 2*i)     right(idx = 2*i + 1)
 */
class Cart {
public:
  /*!
   * \breif default constructor, with parameters from config
   * \param stage         which stage this cart lie in
   * \param landmark_id   which landmark this cart training for regression
   */
  Cart(int stage, int landmark_id);
  ~Cart();

public:
  /*!
   * \breif Generate feature pool, the pool size is determined by Config.feats[stage]
   * \param feature_pool    feature pool
   */
  void GenFeaturePool(std::vector<Feature>& feature_pool);
  /*!
   * \breif Wrapper for `SplitNode`
   */
  void Train(const DataSet& pos, const DataSet& neg);
  /*!
   * \breif Split node with training data
   * \param pos         positive dataset
   * \param neg         negative dataset
   * \param pos_idx     index of used positive dataset
   * \param neg_idx     index of used negative dataset
   * \param node_idx    index of current node in this cart
   */
  void SplitNode(const DataSet& pos, const std::vector<int>& pos_idx, \
                 const DataSet& neg, const std::vector<int>& neg_idx, \
                 int node_idx);
  /*!
   * \breif Classification
   *  split node with classification, minimum Gini
   *  `f = argmax_{f \in F} H_{root} - (H_{left} + H_{right})`
   *
   * \param pos           positive dataset
   * \param neg           negative dataset
   * \param pos_idx       index of used positive dataset
   * \param neg_idx       index of used negative dataset
   * \param pos_feature   pos feature
   * \param neg_feature   neg feature
   * \param feature_id    which feature we should use
   * \param threshold     split threshold
   */
  static void SplitNodeWithClassification(const DataSet& pos, const std::vector<int>& pos_idx, \
                                          const DataSet& neg, const std::vector<int>& neg_idx, \
                                          const cv::Mat_<int>& pos_feature, \
                                          const cv::Mat_<int>& neg_feature, \
                                          int& feature_id, int& threshold);
  /*!
   * \breif Regression
   *  split node with regression, minimize variance of shape_residual
   *  `f = argmax_{f \in F} S_{root} - (S_{left} + S_{right})`
   *
   * \param pos           positive dataset
   * \param neg           negative dataset
   * \param pos_idx       index of used positive dataset
   * \param neg_idx       index of used negative dataset
   * \param pos_feature   pos feature
   * \param neg_feature   neg feature
   * \param feature_id    which feature we should use
   * \param threshold     split threshold
   */
  static void SplitNodeWithRegression(const DataSet& pos, const std::vector<int>& pos_idx, \
                                      const DataSet& neg, const std::vector<int>& neg_idx, \
                                      const cv::Mat_<int>& pos_feature, \
                                      const cv::Mat_<double>& shape_residual, \
                                      int& feature_id, int& threshold);
  /*!
   * \breif Write parameters to a binary file
   * \param   file discriptor of the model file
   */
  void SerializeTo(FILE* fd) const;
  /*!
   * \breif Read parameters from a binary file
   * \param fd    file discriptor of the model file
   */
  void SerializeFrom(FILE* fd);
  /*! \breif Print out the Cart */
  void PrintSelf();

public:
  /*!
   * \breif Forward a data point to leaf node
   * \param img     original region
   * \param img_h   half of original region
   * \param img_q   quarter of original region
   * \param shape   shape
   * \return        leaf node index in this tree, start from 0
   */
  int Forward(const cv::Mat& img, const cv::Mat& img_h, \
              const cv::Mat& img_q, const cv::Mat_<double>& shape) const;

public:
  /*! \breif cascade stage */
  int stage;
  /*! \breif depth of cart */
  int depth;
  /*! \breif numbers of nodes, `nodes_n = 2^depth` */
  int nodes_n;
  /*! \breif number of feature points used in training */
  int featNum;
  /*! \breif radius for sampling feature points */
  double radius;
  /*! \breif number of leaf on cart */
  int leafNum;
  /*! \breif landmark id for regression in this tree */
  int landmark_id;
  /*! \breif threshold, see more on paper about `\theta_k^t` */
  double th;
  /*! \breif features used by this cart, in sequence */
  std::vector<Feature> features;
  /*! \breif thresholds associated with features */
  std::vector<int> thresholds;
  /*! \breif scores to pos/neg, see more on paper in `Algorithm 3` */
  std::vector<double> scores;
};

/*!
 * \breif Boost Classification and Regression Tree
 *  every stage has a BoostCart which combined by boosted Cart
 */
class BoostCart {
public:
  /*!
   * \breif default constructor, with parameters from config
   * \param stage   which stage this boost cart lie in
   */
  BoostCart(int stage);
  ~BoostCart();

public:
  /*!
   * \breif Train boosted Cart
   */
  void Train(DataSet& pos, DataSet& neg);
  /*!
   * \breif Global Regression Training for landmarks
   *  we only use DataSet of pos, X = lbf, Y = shape_residual
   *  see more detail on paper in section 4
   */
  void GlobalRegression(const std::vector<cv::Mat_<int> >& lbf, \
                        const cv::Mat_<double>& shape_residual);

public:
  /*!
   * \breif Generate Local Binary Feature
   * \param img     region
   * \param shape   shape
   * \return        one row of local binary feature
   */
  cv::Mat_<int> GenLBF(const cv::Mat& img, const cv::Mat_<double>& shape) const;
  /*!
   * \breif Generate delta shape with given lbf
   * \param lbf   lbf generated by `GenLBF`
   * \return      one row of delta shape
   */
  cv::Mat_<double> GenDeltaShape(const cv::Mat_<int>& lbf) const;

public:
  /*! \breif number of carts */
  int K;
  /*! \breif which stage this boost cart lies */
  int stage;
  /*! \breif boosted carts */
  std::vector<Cart> carts;
  /*! \breif weight of global regression */
  cv::Mat_<double> w; // stages x LBF_N x 2*landmark_n
};

} // namespace jda

#endif // CART_HPP_
