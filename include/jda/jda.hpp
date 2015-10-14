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
    /**
     * Data Structure of a Cart
     */
    struct jdaCart {
        // **TODO** define data
    };

public:
    /**
     * Load model from pre-trained model file
     * :input fd:   file discriptor of the model file
     */
    void SerializeFrom(FILE* fd);

    /**
     * Detect faces in image
     * :input img:          gray image
     * :output rects:       face locations
     * :output scores:      score of faces
     * :output shapes:      shape of faces
     * :return:             number of faces
     */
    int Detect(cv::Mat& img, std::vector<cv::Rect>& rects, \
               std::vector<double>& scores, std::vector<cv::Mat_<double> >& shapes);
public:
    int T; // number of stages
    int K; // number of carts per-stage
    std::vector<jdaCart> carts; // all carts
    std::vector<cv::Mat_<double> > ws; // all regression weights
};

} // namespace jda

#endif // JDA_HPP_
