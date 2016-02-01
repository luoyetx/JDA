#ifndef DATA_HPP_
#define DATA_HPP_

#include <vector>
#include <opencv2/core/core.hpp>

namespace jda {

// forward declaration
class Cart;
class DataSet;
class Feature;
class JoinCascador;

/*!
 * \breif Negative Training Sample Generator
 *  hard negative training sample will be needed if less negative alives
 */
class NegGenerator {
public:
  NegGenerator();
  ~NegGenerator();

public:
  /*!
   * \breif Generate more negative samples
   *  We will generate negative training samples from origin images, all generated samples
   *  should be hard enough to get through all stages of Join Cascador in current training
   *  state, it may be very hard to generate enough hard negative samples, we may fail with
   *  real size smaller than `int size`. We will give back all negative training samples with
   *  their scores and current shapes for further training.
   *
   * \note OpenMP supported hard negative mining, we may have `real size` > `size`
   *
   * \param join_cascador   JoinCascador in training
   * \param size            how many samples we need
   * \param imgs            negative samples
   * \param scores          scores of negative samples
   * \param shapes          shapes of samples, for training
   * \return                real size
   */
  int Generate(const JoinCascador& joincascador, int size, \
               std::vector<cv::Mat>& imgs, std::vector<double>& scores, \
               std::vector<cv::Mat_<double> >& shapes);

  /*!
   * \breif Load nagetive image file list from path
   * \param path    background image file list
   */
  void Load(const std::string& path);

private:
  /*!
   * \breif Generate negative samples online for hard negative mining
   * \return    negative sample
   */
  cv::Mat NextImage();
  /*!
   * \breif Next State, update parameters for next image
   */
  void NextState();
  /*!
   * \breif We will use this function to load more background images online
   */
  void SaveTheWorld();

 public:
  /*! \breif mean shape of pos dataset for init_shape */
  cv::Mat_<double> mean_shape;

private:
  typedef enum {
    ORIGIN = 0,
    ORIGIN_R,
    ORIGIN_RR,
    ORIGIN_RRR,
    ORIGIN_FLIP,
    ORIGIN_FLIP_R,
    ORIGIN_FLIP_RR,
    ORIGIN_FLIP_RRR,
  } TransformType;
  /*! \breif index of image current used */
  int current_idx;
  /*! \breif negative file list */
  std::vector<std::string> list;
  int x, y;
  cv::Mat img;
  TransformType transform_type;
};

/*!
 * \breif DataSet Wrapper
 */
class DataSet {
public:
  DataSet();
  ~DataSet();

public:
  /*!
   * \breif Load Postive DataSet
   *  All positive samples are listed in this text file with each line represents a sample.
   *  We assume all positive samples are processed and generated before our program runs,
   *  this including resize the training samples, grayscale and data augmentation
   *
   * \param positive    a text file path
   */
  void LoadPositiveDataSet(const std::string& positive);
  /*!
   * \breif Load Negative DataSet
   *  We generate negative samples like positive samples before the program runs. Each line
   *  of the text file hold another text file which holds the real negative sample path in
   *  the filesystem, in this way, we can easily add more negative sample groups without
   *  touching other groups
   *
   * \param negative    a text file path
   */
  void LoadNegativeDataSet(const std::string& negative);
  /*!
   * \breif Wrapper for `LoadPositiveDataSet` and `LoadNegative DataSet`
   *  Since positive dataset and negative dataset may share some information between
   *  each other, we need to load them all together
   */
  static void LoadDataSet(DataSet& pos, DataSet& neg);
  /*!
   * \breif Calculate feature values from `feature_pool` with `idx`
   *
   * \param feature_pool    features
   * \param idx             index of dataset to calculate feature value
   * \return                every row presents a feature with every colum presents a data point
   *                        `feature_{i, j} = f_i(data_j)`
   */
  cv::Mat_<int> CalcFeatureValues(const std::vector<Feature>& feature_pool, \
                                  const std::vector<int>& idx) const;
  /*!
   * \breif Calcualte shape residual of landmark_id over positive dataset
   *  If a landmark id is given, we only generate the shape residual of that landmark
   * \param idx           index of positive dataset
   * \param landmark_id   landmark id to calculate shape residual
   * \return              every data point in each row
   */
  cv::Mat_<double> CalcShapeResidual(const std::vector<int>& idx) const;
  cv::Mat_<double> CalcShapeResidual(const std::vector<int>& idx, int8_t landmark_id) const;
  /*!
   * \biref Calculate Mean Shape over gt_shapes
   * \return    mean_shape of gt_shapes in positive dataset
   */
  cv::Mat_<double> CalcMeanShape() const;
  /*!
   * \breif Random Shapes, a random perturbations on mean_shape
   * \param mean_shape    mean shape of positive samples
   * \param shape         random shape
   * \param shapes        this vector should already malloc memory for shapes
   */
  static void RandomShape(const cv::Mat_<double>& mean_shape, cv::Mat_<double>& shape);
  static void RandomShapes(const cv::Mat_<double>& mean_shape, std::vector<cv::Mat_<double> >& shapes);
  /*!
   * \breif Update weights
   *  `w_i = e^{-y_i*f_i}`, see more on paper in section 4.2
   */
  void UpdateWeights();
  static void UpdateWeights(DataSet& pos, DataSet& neg);
  /*!
   * \breif Update scores by cart
   *  `f_i = f_i + Cart(x, s)`, see more on paper in `Algorithm 3`
   */
  void UpdateScores(const Cart& cart);
  /*!
   * \breif Calculate threshold which seperate scores in two part
   *  `sum(scores < th) / N = rate`
   */
  double CalcThresholdByRate(double rate);
  /*!
   * \breif Adjust DataSet by removing scores < th
   */
  void Remove(double th);
  /*!
   * \breif More Negative Samples if needed (only neg dataset needs)
   * \param pos_size    positive dataset size, reference for generating
   * \param rate        N(negative) / N(positive)
   */
  void MoreNegSamples(int pos_size, double rate);
  /*!
   * \breif Quick Sort by scores descending
   */
  void QSort();
  void _QSort_(int left, int right);

public:
  /*! \breif generator for more negative samples */
  NegGenerator neg_generator;
  /*! \breif face/none-face images */
  std::vector<cv::Mat> imgs;
  std::vector<cv::Mat> imgs_half;
  std::vector<cv::Mat> imgs_quarter;
  // all shapes follows (x_1, y_1, x_2, y_2, ... , x_n, y_n)
  /*! \breif ground-truth shapes for face */
  std::vector<cv::Mat_<double> > gt_shapes;
  /*! \breif current shapes */
  std::vector<cv::Mat_<double> > current_shapes;
  /*! \breif scores, see more about `f_i` on paper */
  std::vector<double> scores;
  /*! \breif weights, see more about `w_i` on paper */
  std::vector<double> weights;
  /*! \breif is positive dataset */
  bool is_pos;
  /*! \breif is sorted by scores */
  bool is_sorted;
  /*! \breif size of dataset */
  int size;
};

} // namespace jda

#endif // DATA_HPP_
