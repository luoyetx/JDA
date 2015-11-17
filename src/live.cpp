#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

void live() {
  VideoCapture cap(0);
  if (!cap.isOpened()) {
    dieWithMsg("Can not open Camera, Please Check it!");
    return;
  }

  JoinCascador joincascador;
  FILE* fd = fopen("../model/jda.model", "rb");
  JDA_Assert(fd, "Can not open model file");
  joincascador.SerializeFrom(fd);
  fclose(fd);

  while (true) {
    Mat frame;
    Mat gray;
    cap >> frame;

    TIMER_BEGIN
      cvtColor(frame, gray, CV_BGR2GRAY);
      vector<Rect> rects;
      vector<double> scores;
      vector<Mat_<double> > shapes;
      int n = joincascador.Detect(gray, rects, scores, shapes);
      for (int i = 0; i < n; i++) {
        frame = jda::drawShape(frame, shapes[i], rects[i]);
      }
      LOG("%.2lf fps", 1. / TIMER_NOW);
    TIMER_END

    cv::imshow("live", frame);
    int key = cv::waitKey(30);
    if (key == 27) {
      break;
    }
  }

  LOG("Bye");
}
