#ifndef JDA_HPP_
#define JDA_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace jda {

/*!
 * \breif Well Optimized JDA
 */
class jdaCascador {
public:
  jdaCascador();
  ~jdaCascador();

public:
  /*!
   * \breif Load model from pre-trained model file
   * \param fd    file discriptor of the model file
   * \return      whether the model is loaded or not
   */
  bool SerializeFrom(FILE* fd);
  /**
   * \breif Detect faces in a gray image
   *
   * \note the interface may change later
   *
   * \param img       gray image
   * \param rects     face locations
   * \param scores    score of faces
   * \param shapes    shape of faces
   * \return          number of faces
   */
  int Detect(cv::Mat& img, std::vector<cv::Rect>& rects, std::vector<double>& scores, \
             std::vector<cv::Mat_<double> >& shapes) const;

private:
  // pre-define
  struct jdaCart;
  /*! \breif mean shape */
  cv::Mat_<double> mean_shape;
  /*! \breif all carts */
  std::vector<jdaCart> carts;
  /*! \breif all regression weights */
  std::vector<cv::Mat_<double> > ws;
};

} // namespace jda

#endif // JDA_HPP_
