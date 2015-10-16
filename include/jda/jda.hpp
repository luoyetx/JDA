#ifndef JDA_HPP_
#define JDA_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace jda {

/**
 * Well Optimized JDA
 */
class jdaCascador {
public:
    jdaCascador();
    ~jdaCascador();

public:
    /**
     * Load model from pre-trained model file
     * :input fd:   file discriptor of the model file
     */
    void SerializeFrom(FILE* fd);

    /**
     * Detect faces in a gray image
     * :input img:          gray image
     * :output rects:       face locations
     * :output scores:      score of faces
     * :output shapes:      shape of faces
     * :return:             number of faces
     */
    int Detect(cv::Mat& img, std::vector<cv::Rect>& rects, std::vector<double>& scores, \
               std::vector<cv::Mat_<double> >& shapes) const;

private:
    // pre-define
    struct jdaCart;

    int T; // number of stages
    int K; // number of carts per-stage
    int landmark_n; // number of landmarks
    int depth; // depth of cart
    int node_n; // number of nodes in a cart, `2^depth`
    int leaf_n; // number of leaves in a cart, `2^(depth-1)`
    cv::Mat_<double> mean_shape; // mean shape

    std::vector<jdaCart> carts; // all carts
    std::vector<cv::Mat_<double> > ws; // all regression weights
};

} // namespace jda

#endif // JDA_HPP_
