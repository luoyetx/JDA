#pragma once

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <jda/data.hpp>
#include <jda/cascador.hpp>

namespace jda {

/*!
 * \breif post filter after join cascador
 */
class PostFilter {
public:
  PostFilter();

public:
  /*!
   * \breif get sift given by patch and shape
   * \param img     patch
   * \param shape   shape
   * \return        one row feature
   */
  cv::Mat_<double> SiftFeature(const cv::Mat& img, const cv::Mat_<double>& shape) const;
  /*! \breif get lbp given by patch and shape */
  cv::Mat_<double> LbpFeature(const cv::Mat& img, const cv::Mat_<double>& shape) const;
  /*!
   * \breif train post filter
   * \param imgs_p    pos images
   * \param shapes_p  pos shapes
   * \param imgs_n    neg images
   * \param shapes_n  neg shapes
   */
  void Train(const std::vector<cv::Mat>& imgs_p, const std::vector<cv::Mat_<double> >& shapes_p, \
             const std::vector<cv::Mat>& imgs_n, const std::vector<cv::Mat_<double> >& shapes_n);
  /*!
   * \breif filter
   * \param imgs    patches
   * \param shapes  shapes
   * \return        set of result, 1 for face and 0 for non-face
   */
  std::vector<int> Filter(const std::vector<cv::Mat>& imgs, const std::vector<cv::Mat_<double> >& shapes) const;

public:
  JoinCascador cascador_;
  std::vector<double> w; // linear svm weights
  cv::SIFT sift;
};

} // namespace jda
