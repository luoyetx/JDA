#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <cassert>
#include <opencv2/core/core.hpp>

// system
#ifdef WIN32
#define NOMINMAX
#include <io.h>
#include <direct.h>
#include <Windows.h>
#define EXISTS(path) (access(path, 0)!=-1)
#define MKDIR(path) mkdir(path)
#define SLEEP(ms) Sleep(ms)
#else
#include <unistd.h>
#include <sys/stat.h>
#define EXISTS(path) (access(path, 0)!=-1)
#define MKDIR(path) mkdir(path, 0775)
#define SLEEP(ms) usleep(ms)
#endif

/*!
 * \brief Timer for evaluation
 *
 * \usage
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

#define JDA_Assert(expr, msg) do { \
  if (!(expr)) \
    jda::dieWithMsg("JDA_Assert failed at LINE: %d, FILE: %s with message \"%s\"", \
                    __LINE__, __FILE__, msg); \
  } while(0)

namespace jda {

// forward declaration
class JoinCascador;
class STParameter;

/*!
 * \brief Feature used by Cart
 *  see more detail on paper in section 4.2
 */
class Feature {
public:
  // scales
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
   * \brief Calculate feature value
   *  We have three scaled image and one shape of original image, the shape of half size
   *  and quarter size will be calculated in this function for feature value
   *
   * \param o   original image
   * \param h   half of original image
   * \param q   quarter of original image
   * \param s   shape of origin image
   * \param stp similarity transform parameter
   * \return    feature value
   */
  int CalcFeatureValue(const cv::Mat& o, const cv::Mat& h, const cv::Mat& q, \
                       const cv::Mat_<double>& s, const STParameter& stp) const;

public:
  /*! \brief scale */
  int scale;
  /*! \brief landmark ids */
  int landmark_id1, landmark_id2;
  /*! \brief relative offset range in [0, 1] */
  double offset1_x, offset1_y, offset2_x, offset2_y;
};

/*!
 * \brief Configure of JDA
 */
class Config {
public:
  static inline Config& GetInstance() {
    static Config c;
    return c;
  }

public:
  // parameters of `Config`, see initialization in `common.cpp::Config()`
  /*! \brief stages */
  int T;
  /*! \brief number of boost carts in each stage */
  int K;
  /*! \brief number of landmarks */
  int landmark_n;
  /*! \brief depth of cart */
  int tree_depth;
  /*! \brief size of all training data */
  bool multi_scale; // whether use multi scale or not
  int img_o_size;
  int img_h_size;
  int img_q_size;
  /*! \brief maximum random shift size on mean shape range [0, shift_size] */
  double shift_size;
  /*! \brief N(negative) / N(postive) */
  std::vector<double> nps;
  /*! \brief sample radius of feature points in each stages */
  std::vector<double> radius;
  /*! \brief feature numbers used by carts in each stages */
  std::vector<int> feats;
  /*! \brief probability of classification in each stages */
  std::vector<double> probs;
  /*! \brief recall of each stage */
  std::vector<double> recall;
  /*! \brief drop number */
  std::vector<int> drops;
  /*! \brief score normalization step, step = normalization_step*landmark_n */
  std::vector<int> score_normalization_steps;
  /*! \brief whether to use similarity transform */
  bool with_similarity_transform;
  /*! \brief hard negative mining parameters */
  double mining_factor;
  int mining_min_size;
  double mining_step_ratio;
  std::vector<double> mining_th;
  /*! \brief a text file for train positive dataset */
  std::string face_txt;
  /*! \brief a text file for train negative dataset */
  std::vector<std::string> bg_txts;
  bool use_hard;
  /*! \brief a text file for face detection test */
  std::string test_txt;
  /*! \brief esp */
  double esp;
  /*! \brief global training join casacdor */
  JoinCascador* joincascador;
  /*! \brief snapshot per iters */
  int snapshot_iter;
  /*! \brief resume model and data */
  std::string resume_model;
  std::string resume_data;
  /*! \brief detection parameters */
  std::string fddb_dir;
  int fddb_step;
  double fddb_scale_factor;
  double fddb_overlap;
  double fddb_minimum_size;
  bool fddb_result;
  bool fddb_nms;
  bool fddb_draw_score;
  bool fddb_draw_shape;
  int fddb_detect_method;
  /*! \brief restart of a cart */
  bool restart_on;
  int restart_times;
  int restart_stage;
  std::vector<double> restart_th;
  /*! \brief online augment parameters */
  bool face_augment_on;
  int landmark_offset;
  std::vector<std::vector<int> > symmetric_landmarks;
  /*! \brief pupils for calculating regreesin error*/
  std::vector<int> left_pupils;
  std::vector<int> right_pupils;
  /*! \brief random generator pool */
  std::vector<cv::RNG> rng_pool;
  /*! \brief thread number */
  int thread_n;

private:
  Config();
  ~Config();
  Config(const Config& other);
  Config& operator=(const Config& other);
};

/*!
 * \brief Printf with timestamp
 */
void LOG(const char* fmt, ...);
/*!
 * \brief Terminate the program with a message
 *
 * \note the message shouldn't be too long
 */
void dieWithMsg(const char* fmt, ...);
/*!
 * \brief Calculate Mean Error between gt_shapes and current_shapes
 */
double calcMeanError(const std::vector<cv::Mat_<double> >& gt_shapes, \
                     const std::vector<cv::Mat_<double> >& current_shapes);
/*!
 * \brief Check the point (x, y) in Image, modify if needed
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
 * \brief Draw shape in the image with optional bounding box
 * \param img     image
 * \param shape   absolute shape binding to the image
 * \param bbox    bounding box of a face binding to the image
 * \return        image with landmarks
 */
cv::Mat drawShape(const cv::Mat& img, const cv::Mat_<double>& shape);
cv::Mat drawShape(const cv::Mat& img, const cv::Mat_<double>& shape, const cv::Rect& bbox);
/*!
 * \brief Show image with shape
 * \param img   image to show
 */
void showImage(const cv::Mat& img);

} // namespace jda

#endif // COMMON_HPP_
