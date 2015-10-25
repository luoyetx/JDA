#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <cassert>
#include <opencv2/core/core.hpp>

/**
 * Timer for evaluation
 *
 * usage:
 *      TIMER_BEGIN
 *          ....
 *          TIMER_NOW # get delta time from TIMER_BEGIN
 *          ....
 *      TIMER_END
 * 
 * The Timer can be cascaded
 *
 *      TIMER_BEGIN # TIMER-1
 *          ....
 *          TIMER_BEGIN # TIMER-2
 *              ....
 *              TIMER_NOW # delta time from TIMER-2
 *              ....
 *          TIMER_END # End of TIMER-2
 *          ....
 *          TIMER_NOW # delta time from TIMER-1
 *          ....
 *      TIMER_END # End of TIMER-1
 */
#define TIMER_BEGIN { double __time__ = cv::getTickCount();
#define TIMER_NOW   ((static_cast<double>(cv::getTickCount()) - __time__) / cv::getTickFrequency())
#define TIMER_END   }

#define JDA_Assert(expr, msg) assert((expr) && (msg))

namespace jda {

/**
 * Feature used by Cart
 *
 * see more detail on paper in section 4.2
 */
class Feature {
public:
    static const int ORIGIN = 0;
    static const int HALF = 1;
    static const int QUARTER = 2;

public:
    /**
     * Calculate feature value
     * :input o:        original image
     * :input h:        half of original image
     * :input q:        quarter of original image
     * :input s:        shape of origin image
     * :return:         feature value
     *
     * We have three scaled image and one shape of original image, the shape of half size
     * and quarter size will be calculated in this function for feature value
     */
    int CalcFeatureValue(const cv::Mat& o, const cv::Mat& h, const cv::Mat& q, \
                         const cv::Mat_<double>& s) const;

public:
    int scale;
    int landmark_id1, landmark_id2;
    double offset1_x, offset1_y; // relative offset range in [0, 1]
    double offset2_x, offset2_y;

    static inline Feature Default() {
        Feature f;
        f.scale = ORIGIN;
        f.landmark_id1 = f.landmark_id2 = 0;
        f.offset1_x = f.offset1_y = f.offset2_x = f.offset2_y = 0;
        return f;
    }
};

/**
 * Configure of JDA
 */
class Config {
public:
    static inline Config& GetInstance() {
        static Config c;
        return c;
    }

public:
    // parameters of `Config`, see initialization in `common.cpp::Config()`
    int T; // stages
    int K; // number of boost carts in each stage
    int landmark_n; // number of landmarks
    int tree_depth; // depth of cart
    int img_o_width, img_o_height; // size of all training data
    int img_h_width, img_h_height;
    int img_q_width, img_q_height;
    int shift_size; // maximum random shift size on mean shape range [0, shift_size]
    std::vector<double> nps; // N(negative) / N(postive)
    std::vector<double> radius; // sample radius of feature points in each stages
    std::vector<int> feats; // feature numbers used by carts in each stages
    std::vector<double> probs; // probability of classification in each stages
    std::vector<double> accept_rates;

    double scale_factor; // hard negative mining parameters
    int x_step, y_step;

    std::string train_pos_txt; // a text file for train positive dataset
    std::string train_neg_txt; // a text file for train negative dataset
    std::string test_pos_txt; // a text file for test positive dataset
    std::string test_neg_txt; // a text file for test negative dataset
    std::string detection_txt; // a text file for detection

    double esp;

private:
    Config();
    ~Config() {}
    Config(const Config& other);
    Config& operator=(const Config& other);
};

/**
 * Printf with timestamp
 */
void LOG(const char* fmt, ...);

/**
 * Terminate the program with a message
 * **NOTICE** the message shouldn't be too long
 */
void dieWithMsg(const char* fmt, ...);

/**
 * Calculate Mean Error between gt_shapes and current_shapes
 */
double calcMeanError(const std::vector<cv::Mat_<double> >& gt_shapes, \
                     const std::vector<cv::Mat_<double> >& current_shapes);

/**
 * Check the point (x, y) in Image, modify if needed
 * :input w:    width of image
 * :input h:    height of image
 * :output x:   x of point
 * :output y:   y of point
 */
inline void checkBoundaryOfImage(int w, int h, int& x, int& y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= w) x = w - 1;
    if (y >= h) y = w - 1;
}

/**
 * Draw shape in the image with optional bounding box
 * :input img:      image
 * :input shape:    absolute shape binding to the image
 * :input bbox:     bounding box of a face binding to the image
 * :return:         image with landmarks
 */
cv::Mat drawShape(const cv::Mat& img, const cv::Mat_<double>& shape);
cv::Mat drawShape(const cv::Mat& img, const cv::Mat_<double>& shape, const cv::Rect& bbox);
/**
 * Show image with shape
 * :input img:      image to show
 */
void showImage(const cv::Mat& img);

} // namespace jda

#endif // COMMON_HPP_
