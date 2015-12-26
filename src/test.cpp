#ifdef WIN32
#include <io.h>
#include <direct.h>
#define EXISTS(path) (access(path, 0)!=-1)
#define MKDIR(path) mkdir(path)
#else
#include <unistd.h>
#include <sys/stat.h>
#define EXISTS(path) (access(path, 0)!=-1)
#define MKDIR(path) mkdir(path, 0775)
#endif

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
  c.shift_size = 0; // no shift
  // set img_o_size, img_h_size and img_q_size by fddb.minimum_size
  const int size = c.fddb_minimum_size;
  c.img_o_size = size;
  c.img_h_size = int(std::sqrt(size*size / 2.));
  c.img_q_size = int(std::sqrt(size*size / 4.));

  JoinCascador joincascador;
  FILE* fd = fopen("../model/jda.model", "rb");
  JDA_Assert(fd, "Can not open model file");
  joincascador.SerializeFrom(fd);
  fclose(fd);

  JDA_Assert(EXISTS("../data/test.txt"), "No test.txt!");
  if (!EXISTS("../data/test_result")) {
    MKDIR("../data/test_result");
  }

  FILE* fin = fopen("../data/test.txt", "r");
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
    joincascador.Detect(gray, rects, scores, shapes);

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
  // set img_o_size, img_h_size and img_q_size by fddb.minimum_size
  const int size = c.fddb_minimum_size;
  c.img_o_size = size;
  c.img_h_size = int(std::sqrt(size*size / 2.));
  c.img_q_size = int(std::sqrt(size*size / 4.));

  JoinCascador joincascador;
  FILE* fd = fopen("../model/jda.model", "rb");
  JDA_Assert(fd, "Can not open model file");
  joincascador.SerializeFrom(fd);
  fclose(fd);

  JDA_Assert(EXISTS("../data/fddb"), "No fddb data!");

  // print out detection result
  time_t t = time(NULL);
  char buff[300];
  strftime(buff, sizeof(buff), "../data/fddb/result/%Y%m%d-%H%M%S", localtime(&t));
  if (c.fddb_result) {
    MKDIR(buff);
  }
  string result_prefix(buff);

  string prefix = "../data/fddb/images/";
  // full test
  #pragma omp parallel for
  for (int i = 1; i <= 10; i++) {
    char fddb[300];
    char fddb_out[300];

    LOG("Testing FDDB-fold-%02d.txt", i);
    sprintf(fddb, "../data/fddb/FDDB-folds/FDDB-fold-%02d.txt", i);
    sprintf(fddb_out, "../data/fddb/result/fold-%02d-out.txt", i);

    FILE* fin = fopen(fddb, "r");
    JDA_Assert(fin, "Can not open fddb");
    FILE* fout = fopen(fddb_out, "w");
    JDA_Assert(fin, "Can not open fddb_out");

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
      joincascador.Detect(gray, rects, scores, shapes);

      const int n = rects.size();
      fprintf(fout, "%s\n%d\n", path, n);
      LOG("%s get %d faces", path, n);

      for (int j = 0; j < n; j++) {
        const Rect& r = rects[j];
        double score = scores[j];
        const Mat_<double> shape = shapes[j];
        fprintf(fout, "%d %d %d %d %lf\n", r.x, r.y, r.width, r.height, score);
        cv::rectangle(img, r, Scalar(0, 0, 255), 3);
        for (int k = 0; k < c.landmark_n; k++) {
          cv::circle(img, Point(shape(0, 2 * k), shape(0, 2 * k + 1)), 3, Scalar(0, 255, 0), -1);
        }
      }
      char buff[300];
      if (c.fddb_result) {
        counter++;
        sprintf(buff, "%s/%02d_%04d.jpg", result_prefix.c_str(), i, counter);
        cv::imwrite(buff, img);
      }
    }

    fclose(fin);
    fclose(fout);
  }
}
