#ifndef DATA_HPP_
#define DATA_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace jda {

// pre-define
class Feature;
class Cart;

/**
 * DataSet Wrapper
 */
class DataSet {
public:
    DataSet();
    ~DataSet();
    DataSet(const DataSet& other);
    DataSet& operator=(const DataSet& other);

public:
    /**
     * Calculate feature values from `feature_pool` with `idx`
     * :return:     every row presents a feature with every colum presents a data point
     *              `feature_{i, j} = f_i(data_j)`
     */ 
    cv::Mat CalcFeatureValues(std::vector<Feature>& feature_pool, \
                              std::vector<int>& idx);
    /**
     * Calcualte shape residual of landmark_id over positive dataset
     * :return:     every data point in each row
     */
    cv::Mat CalcShapeResidual(std::vector<int>& idx, int landmark_id = -1);
    /**
     * Update weights
     *
     * `w_i = e^{-y_i*f_i}`, see more on paper in section 4.2
     */
    void UpdateWeights();
    /**
     * Update scores by cart
     *
     * `f_i = f_i + Cart(x, s)`, see more on paper in `Algorithm 3`
     */
    void UpdateScores(Cart& cart);
    /**
     * Calculate threshold which seperate scores in two part
     *
     * `sum(scores < th) / N = rate`
     */
    double CalcThresholdByRate(double rate);
    /**
     * Adjust DataSet by removing scores < th
     */
    void Remove(double th);
    /**
     * Quick Sort by scores
     */
    void QSort();
    void _QSort_(int left, int right);

public:
    std::vector<cv::Mat> imgs; // face/none-face images
    std::vector<cv::Mat_<double>> gt_shapes; // ground-truth shapes for face
    std::vector<cv::Mat_<double>> current_shapes; // current shapes
    std::vector<double> scores; // scores, see more about `f_i` on paper
    std::vector<double> weights; // weights, see more about `w_i` on paper
    bool is_pos; // is positive dataset
    bool is_sorted; // is sorted by scores
    int size; // size of dataset
};

} // namespace jda

#endif // DATA_HPP_
