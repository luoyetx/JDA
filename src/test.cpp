#include <ctime>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/data.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

/*! \breif Test over your own data */
void test() {
  Config& c = Config::GetInstance();
  c.shift_size = 0.; // no shift

  JoinCascador joincascador;
  FILE* fd = fopen("../model/jda.model", "rb");
  JDA_Assert(fd, "Can not open model file");
  joincascador.SerializeFrom(fd);
  fclose(fd);

  JDA_Assert(EXISTS(c.test_txt.c_str()), "No test.txt!");
  if (!EXISTS("../data/test_result")) {
    MKDIR("../data/test_result");
  }

  FILE* fin = fopen(c.test_txt.c_str(), "r");
  char path[300];
  int counter = 0;
  while (fscanf(fin, "%[^\n]\n", path) > 0) {
    Mat img = imread(path);
    if (!img.data) {
      LOG("Can not open %s, Skip it", path);
      continue;
    }
    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);
    vector<double> scores;
    vector<Rect> rects;
    vector<Mat_<double> > shapes;
    DetectionStatisic statisic;
    joincascador.Detect(gray, rects, scores, shapes, statisic);

    const int n = rects.size();
    LOG("%s get %d faces", path, n);

    for (int j = 0; j < n; j++) {
      const Rect& r = rects[j];
      double score = scores[j];
      const Mat_<double> shape = shapes[j];
      cv::rectangle(img, r, Scalar(0, 0, 255), 3);
      for (int k = 0; k < c.landmark_n; k++) {
        cv::circle(img, Point(shape(0, 2 * k), shape(0, 2 * k + 1)), 3, Scalar(0, 255, 0), -1);
      }
    }
    char buff[300];
    if (c.fddb_result) {
      counter++;
      sprintf(buff, "../data/test_result/%04d.jpg", counter);
      cv::imwrite(buff, img);
    }
  }

  fclose(fin);
}

/*!
 * \breif Test JoinCascador Face Detection over FDDB
 */
void fddb() {
  Config& c = Config::GetInstance();
  c.shift_size = 0; // no shift

  JoinCascador joincascador;
  FILE* fd = fopen("../model/jda.model", "rb");
  JDA_Assert(fd, "Can not open model file");
  joincascador.SerializeFrom(fd);
  fclose(fd);

  const char* fddb_dir = c.fddb_dir.c_str();

  JDA_Assert(EXISTS(fddb_dir), "No fddb data!");

  // print out detection result
  time_t t = time(NULL);
  char buff[300];
  string format = c.fddb_dir + string("/result/%Y%m%d-%H%M%S");
  strftime(buff, sizeof(buff), format.c_str(), localtime(&t));
  if (c.fddb_result) {
    MKDIR(buff);
  }
  string result_prefix(buff);

  string prefix = c.fddb_dir + string("/images/");
  vector<DetectionStatisic> statisic(11);
  // full test
  #pragma omp parallel for
  for (int i = 1; i <= 10; i++) {
    char fddb[300];
    char fddb_out[300];
    char fddb_answer[300];

    LOG("Testing FDDB-fold-%02d.txt", i);
    sprintf(fddb, "%s/FDDB-folds/FDDB-fold-%02d.txt", fddb_dir, i);
    sprintf(fddb_out, "%s/result/fold-%02d-out.txt", fddb_dir, i);
    sprintf(fddb_answer, "%s/FDDB-folds/FDDB-fold-%02d-ellipseList.txt", fddb_dir, i);

    FILE* fin = fopen(fddb, "r");
    JDA_Assert(fin, "Can not open fddb");
    FILE* fanswer = fopen(fddb_answer, "r");
    JDA_Assert(fanswer, "Can not open fddb_answer");
#ifdef WIN32
    FILE* fout = fopen(fddb_out, "wb"); // replace \r\n on Windows platform
#else
    FILE* fout = fopen(fddb_out, "w");
#endif // WIN32
    JDA_Assert(fin, "Can not open fddb_out");

    char buff[300];
    char _buff[30];
    char path[300];
    int counter = 0;
    while (fscanf(fin, "%s", path) > 0) {
      string full_path = prefix + string(path) + string(".jpg");
      Mat img = imread(full_path);
      if (!img.data) {
        LOG("Can not open %s, Skip it", full_path.c_str());
        continue;
      }
      Mat gray;
      cvtColor(img, gray, CV_BGR2GRAY);
      vector<double> scores;
      vector<Rect> rects;
      vector<Mat_<double> > shapes;
      DetectionStatisic statisic_;

      double fps = 0.;
      TIMER_BEGIN
        joincascador.Detect(gray, rects, scores, shapes, statisic_);
        fps = 1. / TIMER_NOW;
      TIMER_END

      statisic[i].patch_n += statisic_.patch_n;
      statisic[i].face_patch_n += statisic_.face_patch_n;
      statisic[i].nonface_patch_n += statisic_.nonface_patch_n;
      statisic[i].cart_gothrough_n += statisic_.cart_gothrough_n;

      const int n = rects.size();

      fprintf(fout, "%s\n%d\n", path, n);
      LOG("Patch_n = %d, Non-Face Patch_n = %d, Face Patch_n = %d, "
          "Average Cart_N to Reject = %.4lf, FPS = %.4lf", \
          statisic_.patch_n, statisic_.nonface_patch_n, statisic_.face_patch_n, \
          statisic_.average_cart_n, fps);

      for (int j = 0; j < n; j++) {
        const Rect& r = rects[j];
        double score = scores[j];
        const Mat_<double> shape = shapes[j];
        fprintf(fout, "%d %d %d %d %lf\n", r.x, r.y, r.width, r.height, score);
      }

      if (c.fddb_result) {
        counter++;
        sprintf(buff, "%s/%02d_%03d_%03d_%02d.jpg", result_prefix.c_str(), i, counter, statisic_.face_patch_n, n);

        // get answer
        int face_n = 0;
        fscanf(fanswer, "%s", path);
        fscanf(fanswer, "%d", &face_n);
        for (int k = 0; k < face_n; k++) {
          double major_axis_radius, minor_axis_radius, angle, center_x, center_y, score;
          fscanf(fanswer, "%lf %lf %lf %lf %lf %lf", &major_axis_radius, &minor_axis_radius, \
                                                     &angle, &center_x, &center_y, &score);
          // draw answer
          angle = angle / 3.1415926*180.;
          cv::ellipse(img, Point2d(center_x, center_y), Size(major_axis_radius, minor_axis_radius), \
                      angle, 0., 360., Scalar(255, 0, 0), 2);
        }

        // draw result
        for (int j = 0; j < n; j++) {
          const Rect& r = rects[j];
          double score = scores[j];
          const Mat_<double> shape = shapes[j];
          cv::rectangle(img, r, Scalar(0, 0, 255), 3);
          // draw score
          if (c.fddb_draw_score) {
            sprintf(_buff, "%.4lf", score);
            cv::putText(img, _buff, cv::Point(r.x, r.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 2);
          }
          // draw shape
          if (c.fddb_draw_shape) {
            for (int k = 0; k < c.landmark_n; k++) {
              cv::circle(img, Point(shape(0, 2 * k), shape(0, 2 * k + 1)), 3, Scalar(0, 255, 0), -1);
            }
          }
        }

        //string fname(path);
        //std::replace(fname.begin(), fname.end(), '/', '_');
        //fname = result_prefix + "/" + fname + ".jpg";
        //cv::imwrite(fname, img);
        cv::imwrite(buff, img);
      }
    }

    statisic[i].average_cart_n = double(statisic[i].cart_gothrough_n) / statisic[i].nonface_patch_n;
    LOG("Summary of Test-%02d", i);
    LOG("Patch_n = %d, Non-Face Patch_n = %d, Face Patch_n = %d, Average Cart_N to Reject = %.4lf", \
        statisic[i].patch_n, statisic[i].nonface_patch_n, statisic[i].face_patch_n, statisic[i].average_cart_n);

    fclose(fin);
    fclose(fout);
    fclose(fanswer);
  }

  DetectionStatisic statisic_final;
  for (int i = 1; i < 11; i++) {
    statisic_final.patch_n += statisic[i].patch_n;
    statisic_final.face_patch_n += statisic[i].face_patch_n;
    statisic_final.nonface_patch_n += statisic[i].nonface_patch_n;
    statisic_final.cart_gothrough_n += statisic[i].cart_gothrough_n;
  }

  statisic_final.average_cart_n = double(statisic_final.cart_gothrough_n) / statisic_final.nonface_patch_n;
  LOG("Summary of ALL");
  LOG("Patch_n = %d, Non-Face Patch_n = %d, Face Patch_n = %d, Average Cart_N to Reject = %.4lf", \
      statisic_final.patch_n, statisic_final.nonface_patch_n, \
      statisic_final.face_patch_n, statisic_final.average_cart_n);
}
