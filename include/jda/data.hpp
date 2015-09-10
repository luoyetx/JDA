#ifndef DATA_HPP_
#define DATA_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace jda {

// pre-define
class Feature;
class Cart;
class JoinCascador;

/**
 * Negative Training Sample Generator
 *
 * hard negative training sample will be needed if less negative alives
 */
class NegGenerator {
public:
    NegGenerator();
    ~NegGenerator();
    NegGenerator(const NegGenerator& other);
    NegGenerator& operator=(const NegGenerator& other);

public:
    /**
     * Generate more negative samples
     * :input join_cascdor: JoinCascador in training
     * :input size:         how many samples we need
     * :input stage:        we will run join_cascador to stage
     * :output imgs:        negative samples
     * :output scores:      scores of negative samples
     * :return:             real size
     *
     * We will generate negative training samples from origin images, all generated samples
     * should be hard enough to get through all stages of Join Cascador in current training
     * state, it may be every hard to generate enough hard negative samples, we may fail with
     * real size smaller than `int size`. We will give back all negative training samples with
     * their scores and current shapes for further training.
     * Online or Offline ?
     */
    int Generate(JoinCascador& joincascador, int size, int stage, \
                 std::vector<cv::Mat>& imgs, std::vector<double>& scores, \
                 std::vector<cv::Mat_<double> >& shapes);
    /**
     * Set file list from path
     */
    void SetOriginList(const std::string& path);

public:
    std::vector<std::string> list; // negative file list
    std::vector<std::string> current;
    int list_idx;
    int current_idx;
    cv::Mat_<double> mean_shape; // mean shape of pos dataset for prediction
};


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
     * Load Postive DataSet
     * :input positive:     a text file path
     *
     * All positive samples are listed in this text file with each line represents a sample.
     * We assume all positive samples are processed and generated before our program runs,
     * this including resize the training samples, grayscale and data augmentation
     */
    void LoadPositiveDataSet(const std::string& positive);
    /**
     * Load Negative DataSet
     * :input negative:     a text file path
     *
     * We generate negative samples like positive samples before the program runs. Each line
     * of the text file hold another text file which holds the real negative sample path in
     * the filesystem, in this way, we can easily add more negative sample groups without
     * touching other groups
     */
    void LoadNegativeDataSet(const std::string& negative);
    /**
     * Wrapper for `LoadPositiveDataSet` and `LoadNegative DataSet`
     *
     * Since positive dataset and negative dataset may share some information between
     * each other, we need to load them all together
     */
    static void LoadDataSet(DataSet& pos, DataSet& neg);
    /**
     * Calculate feature values from `feature_pool` with `idx`
     * :return:     every row presents a feature with every colum presents a data point
     *              `feature_{i, j} = f_i(data_j)`
     */
    cv::Mat_<int> CalcFeatureValues(std::vector<Feature>& feature_pool, \
                                    std::vector<int>& idx);
    /**
     * Calcualte shape residual of landmark_id over positive dataset
     * :return:     every data point in each row
     */
    cv::Mat_<double> CalcShapeResidual(std::vector<int>& idx);
    cv::Mat_<double> CalcShapeResidual(std::vector<int>& idx, int landmark_id);
    /**
     * Calculate Mean Shape over gt_shapes
     */
    cv::Mat_<double> CalcMeanShape();
    /**
     * Random Shapes, a random perturbations on mean_shape
     * :input mean_shape:       mean shape of positive samples
     * :output shapes:          this vector should already malloc memory for shapes
     */
    static void RandomShapes(cv::Mat_<double>& mean_shape, std::vector<cv::Mat_<double> >& shapes);
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
     * More Negative Samples if needed (only neg dataset needs)
     * :input stage:    which stage it is
     * :input size:     positive dataset size, reference for generating
     */
    void MoreNegSamples(int stage, int size);
    /**
     * Set Join Cascador (only neg dataset needs)
     */
    void set_joincascador(JoinCascador* joincascador) { this->joincascador = joincascador; }
    /**
     * Quick Sort by scores
     */
    void QSort();
    void _QSort_(int left, int right);

public:
    NegGenerator neg_generator; // generator for more negtive samples
    std::vector<cv::Mat> imgs; // face/none-face images
    // all shapes follows (x_1, y_1, x_2, y_2, ... , x_n, y_n)
    std::vector<cv::Mat_<double> > gt_shapes; // ground-truth shapes for face
    std::vector<cv::Mat_<double> > current_shapes; // current shapes
    std::vector<double> scores; // scores, see more about `f_i` on paper
    std::vector<double> weights; // weights, see more about `w_i` on paper
    bool is_pos; // is positive dataset
    bool is_sorted; // is sorted by scores
    int size; // size of dataset

private:
    JoinCascador* joincascador; // join cascador for hard negative sample mining
};

} // namespace jda

#endif // DATA_HPP_
