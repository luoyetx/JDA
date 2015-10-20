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
     * :return:     whether the model is loaded or not
     */
    bool SerializeFrom(FILE* fd);
    /**
     * Detect faces in a gray image
     * :input img:          gray image
     * :output rects:       face locations
     * :output scores:      score of faces
     * :output shapes:      shape of faces
     * :return:             number of faces
     *
     * **NOTICE** the interface may change later
     */
    int Detect(cv::Mat& img, std::vector<cv::Rect>& rects, std::vector<double>& scores, \
               std::vector<cv::Mat_<double> >& shapes) const;

private:
    // pre-define
    struct jdaCart;
    cv::Mat_<double> mean_shape; // mean shape

    std::vector<jdaCart> carts; // all carts
    std::vector<cv::Mat_<double> > ws; // all regression weights
};

} // namespace jda

#endif // JDA_HPP_
