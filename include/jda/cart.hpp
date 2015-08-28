#ifndef CART_HPP_
#define CART_HPP_

#include <opencv2/core/core.hpp>

namespace jda {

// pre-define
class Feature;
class DataSet;

/**
 * Classification and Regression Random Tree
 *
 * see more detail on paper in section 4 about `CR_k^t`
 */
class Cart {
public:
    Cart();
    ~Cart();
    Cart(const Cart& other);
    Cart& operator=(const Cart& other);

public:
    /**
     * Generate feature pool
     */
    void GenFeaturePool(std::vector<Feature>& feature_pool);
    /**
     * Wrapper for `SplitNode`
     */
    void Train(DataSet& pos, DataSet& neg);
    /**
     * Split node with training data
     */
    void SplitNode(DataSet& pos, std::vector<int>& pos_idx, \
                   DataSet& neg, std::vector<int>& neg_idx, \
                   int node_idx);
    /**
     * Classification
     *
     * split node with classification, minimize binary entropy of pos and neg
     * `f = argmax_{f \in F} H_{root} - (H_{left} + H_{right})`
     */
    void SplitNodeWithClassification(cv::Mat_<int>& pos_feature, \
                                     cv::Mat_<int>& neg_feature, \
                                     int& feature_id, int& threshold);
    /**
     * Regression
     *
     * split node with regression, minimize variance of shape_residual
     * `f = argmax_{f \in F} S_{root} - (S_{left} + S_{right})`
     */
    void SplitNodeWithRegression(cv::Mat_<int>& pos_feature, \
                                 cv::Mat_<double>& shape_residual, \
                                 int& feature_id, int& threshold);

public:
    /**
     * Forward a data point to leaf node
     * :return:     leaf node index in this tree
     */
    int Forward(cv::Mat& img, cv::Mat& shape);

public:
    int depth; // depth of cart
    int nodes_n; // numbers of nodes, `nodes_n = 2^depth`
    int featNum; // number of feature points used in training
    double radius; // radius for sampling feature points
    int leafNum; // number of leaf on cart
    double th; // threshold, see more on paper about `\theta_k^t`
    double p; // probability of internel node to do classification or regression
    int landmark_id; // landmark id for regression in this tree

    std::vector<Feature> features; // features used by this cart, in sequence
    std::vector<int> thresholds; // thresholds associated with features
    std::vector<bool> is_classifications; // classification of internel node
    std::vector<double> scores; // scores to pos/neg, see more on paper in `Algorithm 3`
};

/**
 * Boost Classification and Regression Tree
 *
 * every stage has a BoostCart which combined by boosted Cart
 */
class BoostCart {
public:
    BoostCart();
    ~BoostCart();
    BoostCart(const BoostCart& other);
    BoostCart& operator=(const BoostCart& other);

public:
    /**
     * Train boosted Cart
     */
    void Train(DataSet& pos, DataSet& neg);
    /**
     * Global Regression Training for landmarks
     *
     * we only use DataSet of pos, X = lbf, Y = shape_residual
     * see more detail on paper in section 4
     */
    void GlobalRegression(DataSet& pos);

public:
    int K; // number of carts
    double tp_rate; // true positive rate
    double fn_rate; // false negative rate

    std::vector<Cart> carts; // boosted carts
    cv::Mat_<double> w; // weight of global regression
};

} // namespace jda

#endif // CART_HPP_
