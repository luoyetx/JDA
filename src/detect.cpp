#include <cstdio>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <jda/common.hpp>
#include <jda/cascador.hpp>

using namespace cv;
using namespace std;
using namespace jda;

/**
 * Detect Face over a given image set
 */
void detect() {
    const Config& c = Config::GetInstance();
    JoinCascador joincascdor;
    FILE* fd = fopen("../model/jda.model", "rb");
    JDA_Assert(fd, "Can not open `../model/jda.model`");
    joincascdor.SerializeFrom(fd);
    fclose(fd);

    fd = fopen(c.detection_txt.c_str(), "r");
    JDA_Assert(fd, "Can not open detection text file");
    char buff[256];
    while (fscanf(fd, "%s", buff) > 0) {
        Mat img, gray;
        vector<Rect> rects;
        vector<double> scores;
        vector<Mat_<double> > shapes;
        LOG("Detecting %s", buff);
        img = cv::imread(buff);
        if (!img.data) {
            LOG("Can not open %s, Skip it!", buff);
            continue;
        }
        cvtColor(img, gray, CV_BGR2GRAY);
        int n = joincascdor.Detect(gray, rects, scores, shapes);
        for (int i = 0; i < n; i++) {
            img = jda::drawShape(img, shapes[i], rects[i]);
        }
        jda::showImage(img);
    }
    LOG("Bye");
}
