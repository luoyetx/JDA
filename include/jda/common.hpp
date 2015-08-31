#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <opencv2/core/core.hpp>

/**
 * Timer for evaluation
 *
 * usage:
 *      TIMER_BEGIN
 *          ....
 *          TIMER_NOW // get delta time from TIMER_BEGIN
 *          ....
 *      TIMER_END
 */
#define TIMER_BEGIN { double __time__ = getTickCount();
#define TIMER_NOW   ((getTickCount() - __time__) / getTickFrequency())
#define TIMER_END   }

namespace jda {

/**
* Feature used by Cart
*
* see more detail on paper in section 4.2
*/
class Feature {
public:
    enum Scale {
        ORIGIN,
        HALF,
        QUARTER
    };

public:
    Scale scale;
    int landmark_id1, landmark_id2;
    double offset1_x, offset1_y;
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
    int T;
    int K;
    int landmark_n;
    int tree_depth;
    std::vector<double> radius;
    std::vector<int> feats;
    std::vector<double> probs;

private:
    Config();
    ~Config() {}
    Config(const Config& other);
    Config& operator=(const Config& other);
};

/**
 * printf with timestamp
 */
void LOG(const char* fmt, ...);

/**
 * Calculate Variance of vector
 */
double calcVariance(const cv::Mat_<double>& vec);
double calcVariance(const std::vector<double>& vec);

/**
* Check the point (x, y) in Image, modify if needed
*/
inline void checkBoundaryOfImage(int w, int h, double& x, double& y) {
    if (x < 0) x = 0.0;
    if (y < 0) y = 0.0;
    if (x > w) x = double(w);
    if (y > h) y = double(h);
}

} // namespace jda

#endif // COMMON_HPP_
