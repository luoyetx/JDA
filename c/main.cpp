#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda.h"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
  void* cascador = jdaCascadorCreateDouble("jda.model");
  jdaCascadorSerializeTo(cascador, "jda_float32.model");
  jdaCascadorRelease(cascador);
  cascador = jdaCascadorCreateFloat("jda_float32.model");

  cv::Mat img = cv::imread("test.jpg");
  Mat gray;
  cvtColor(img, gray, CV_BGR2GRAY);

  const int N = 10;
  jdaResult res[N];
  for (int i = 0; i < N; i++) {
    printf("%02d ", i + 1);
    double t = getTickCount();
    res[i] = jdaDetect(cascador, gray.data, img.cols, img.rows, 1.25, 0.1, 40, -1, -0.5);
    t = (getTickCount() - t) / getTickFrequency();
    printf("time = %.3lfms, fps = %.2lf\n", t * 1000., 1. / t);
  }

  jdaResult result = res[0];
  for (int i = 0; i < result.n; i++) {
    Rect r(result.bboxes[3 * i + 0], result.bboxes[3 * i + 1], \
           result.bboxes[3 * i + 2], result.bboxes[3 * i + 2]);
    float* shape = &result.shapes[2 * result.landmark_n*i];
    rectangle(img, r, Scalar(0, 0, 255), 2);
    char buff[200];
    sprintf(buff, "%.4lf", result.scores[i]);
    cv::putText(img, buff, cv::Point(r.x, r.y), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0), 2);
    for (int j = 0; j < result.landmark_n; j++) {
      circle(img, Point(shape[2 * j], shape[2 * j + 1]), 2, Scalar(0, 255, 0), -1);
    }
  }

  for (int i = 0; i < N; i++) {
    jdaResultRelease(res[i]);
  }

  imshow("res", img);
  waitKey(0);

  imwrite("result.jpg", img);

  jdaCascadorRelease(cascador);
  return 0;
}
