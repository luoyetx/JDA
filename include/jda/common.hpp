#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <cassert>
#include <opencv2/core/core.hpp>

/*!
 * \breif Timer for evaluation
 *
 * usage:
 *  TIMER_BEGIN
 *    ....
 *    TIMER_NOW # get delta time from TIMER_BEGIN
 *    ....
 *  TIMER_END
 *
 *  The Timer can be cascaded
 *
 *  TIMER_BEGIN # TIMER-1
 *    ....
 *    TIMER_BEGIN # TIMER-2
 *      ....
 *      TIMER_NOW # delta time from TIMER-2
 *      ....
 *    TIMER_END # End of TIMER-2
 *    ....
 *    TIMER_NOW # delta time from TIMER-1
 *    ....
 *  TIMER_END # End of TIMER-1
 */
#define TIMER_BEGIN { double __time__ = cv::getTickCount();
#define TIMER_NOW   ((double(cv::getTickCount()) - __time__) / cv::getTickFrequency())
#define TIMER_END   }

#define JDA_Assert(expr, msg) do { if (!(expr)) jda::dieWithMsg(msg); } while(0)

namespace jda {

// forward declaration
class JoinCascador;

/*!
 * \breif Feature used by Cart
 *  see more detail on paper in section 4.2
 */
class Feature {
public:
  static const int ORIGIN = 0;
  static const int HALF = 1;
  static const int QUARTER = 2;

public:
  Feature() {
    scale = ORIGIN;
    landmark_id1 = landmark_id2 = 0;
    offset1_x = offset1_y = 0.;
    offset2_x = offset2_y = 0.;
  }
  /*!
   * \breif Calculate feature value
   *  We have three scaled image and one shape of original image, the shape of half size
   *  and quarter size will be calculated in this function for feature value
   *
   * \param o   original image
   * \param h   half of original image
   * \param q   quarter of original image
   * \param s   shape of origin image
   * \return    feature value
   */
  int CalcFeatureValue(const cv::Mat& o, const cv::Mat& h, const cv::Mat& q, \
                       const cv::Mat_<double>& s) const;

public:
  /*! \breif scale */
  int scale;
  /*! \breif landmark ids */
  int landmark_id1, landmark_id2;
  /*! \breif relative offset range in [0, 1] */
  double offset1_x, offset1_y, offset2_x, offset2_y;
};

/*!
 * \breif Configure of JDA
 */
class Config {
public:
  static inline Config& GetInstance() {
    static Config c;
    return c;
  }

public:
  // parameters of `Config`, see initialization in `common.cpp::Config()`
  /*! \breif stages */
  int T;
  /*! \breif number of boost carts in each stage */
  int K;
  /*! \breif number of landmarks */
  int landmark_n;
  /*! \breif depth of cart */
  int tree_depth;
  /*! \breif size of all training data */
  bool multi_scale; // whether use multi scale or not
  int img_o_size;
  int img_h_size;
  int img_q_size;
  /*! \breif maximum random shift size on mean shape range [0, shift_size] */
  double shift_size;
  /*! \breif N(negative) / N(postive) */
  std::vector<double> nps;
  /*! \breif sample radius of feature points in each stages */
  std::vector<double> radius;
  /*! \breif feature numbers used by carts in each stages */
  std::vector<int> feats;
  /*! \breif probability of classification in each stages */
  std::vector<double> probs;
  /*! \breif recall of each stage */
  std::vector<double> recall;
  /*! \breif hard negative mining parameters */
  double scale_factor;
  int x_step, y_step;
  int mining_pool_size;
  /*! \breif a text file for train positive dataset */
  std::string train_pos_txt;
  /*! \breif a text file for train negative dataset */
  std::string train_neg_txt;
  /*! \breif a text file for test positive dataset */
  std::string test_pos_txt;
  /*! \breif a text file for test negative dataset */
  std::string test_neg_txt;
  /*! \breif a text file for detection */
  std::string detection_txt;
  /*! \breif esp */
  double esp;
  /*! \breif global training join casacdor */
  JoinCascador* joincascador;
  /*! \breif model snapshot by JDA, used for resume and test only */
  std::string tmp_model;
  /*! \breif snapshot per iters */
  int snapshot_iter;
  /*!
   * \breif global status, train or test, 0 for train and 1 for test
   *
   * \note    when test the model, you should give `current_stage_idx`
   *          and `current_cart_idx` to point out the model
   */
  int phase;
  /*! \breif detection parameters */
  int fddb_x_step, fdbb_y_step;
  double fddb_scale_factor;
  double fddb_overlap;
  double fddb_minimum_size;
  bool fddb_result;

private:
  Config();
  ~Config() {}
  Config(const Config& other);
  Config& operator=(const Config& other);
};

/*!
 * \breif Printf with timestamp
 */
void LOG(const char* fmt, ...);
/*!
 * \breif Terminate the program with a message
 *
 * \note the message shouldn't be too long
 */
void dieWithMsg(const char* fmt, ...);
/*!
 * \breif Calculate Mean Error between gt_shapes and current_shapes
 */
double calcMeanError(const std::vector<cv::Mat_<double> >& gt_shapes, \
                     const std::vector<cv::Mat_<double> >& current_shapes);
/*!
 * \breif Check the point (x, y) in Image, modify if needed
 * \param w   width of image
 * \param h   height of image
 * \param x   x of point
 * \param y   y of point
 */
inline void checkBoundaryOfImage(int w, int h, int& x, int& y) {
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x >= w) x = w - 1;
  if (y >= h) y = h - 1;
}
/*!
 * \breif Draw shape in the image with optional bounding box
 * \param img     image
 * \param shape   absolute shape binding to the image
 * \param bbox    bounding box of a face binding to the image
 * \return        image with landmarks
 */
cv::Mat drawShape(const cv::Mat& img, const cv::Mat_<double>& shape);
cv::Mat drawShape(const cv::Mat& img, const cv::Mat_<double>& shape, const cv::Rect& bbox);
/*!
 * \breif Show image with shape
 * \param img   image to show
 */
void showImage(const cv::Mat& img);

} // namespace jda

#endif // COMMON_HPP_
