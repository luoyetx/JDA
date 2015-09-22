#ifndef CASCADOR_HPP_
#define CASCADOR_HPP_

#include <vector>

namespace jda {

// pre-define
class BoostCart;

/**
 * JoinCascador for face classification and landmark regression
 */
class JoinCascador {
public:
    JoinCascador();
    ~JoinCascador();
    JoinCascador(const JoinCascador& other);
    JoinCascador& operator=(const JoinCascador& other);
    void Initialize(int T);

public:
    /**
     * Train JoinCascador
     *
     * See Full Algorithm on paper `Algorithm 3`
     */
    void Train(DataSet& pos, DataSet& neg);
    /**
     * Write parameters to a binary file
     */
    void SerializeTo(FILE* fd);
    /**
     * Read parameters from a binary file
     */
    void SerializeFrom(FILE* fd);

public:
    /**
     * Validate a region whether a face or not
     * :input region:   region
     * :output score:   classification score of this region
     * :output shape:   shape on this region
     * :return:         whether a face or not
     *
     * In training state, we use this function for hard negative mining based on
     * the training status. In testing state, we just go through all carts to get
     * a face score for this region. The training status is based on `current_stage_idx`
     * and `current_cart_idx`.
     */
    bool Validate(cv::Mat& region, double& score, cv::Mat_<double>& shape);

public:
    int T; // number of stages
    cv::Mat_<double> mean_shape; // mean shape of positive training data

    std::vector<BoostCart> btcarts;

    // training status
    int current_stage_idx;
    int current_cart_idx;
};

} // namespace jda

#endif // CASCADOR_HPP_
