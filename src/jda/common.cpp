#include <ctime>
#include <cstdio>
#include <cstdarg>
#include "jda/common.hpp"

using namespace cv;
using namespace std;

namespace jda {

void LOG(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char msg[256];
    vsprintf(msg, fmt, args);
    va_end(args);

    char buff[256];
    time_t t = time(NULL);
    strftime(buff, sizeof(buff), "[%x - %X]", localtime(&t));
    printf("%s %s\n", buff, msg);
}

double calcVariance(const Mat_<double>& vec) {
    double m1 = cv::mean(vec)[0];
    double m2 = cv::mean(vec.mul(vec))[0];
    double variance = m2 - m1*m1;
    return variance;
}
double calcVariance(const vector<double>& vec) {
    if (vec.size() == 0) return 0.;
    Mat_<double> vec_(vec);
    double m1 = cv::mean(vec_)[0];
    double m2 = cv::mean(vec_.mul(vec_))[0];
    double variance = m2 - m1*m1;
    return variance;
}

Config::Config() {
    T = 5;
    K = 1080;
    landmark_n = 5;
    tp_rate = 0.99;
    fn_rate = 0.3;
    shift_size = 10;
    np_ratio = 1.1;
    img_height = img_height = 80;
    int feats[5] = { 500, 500, 500, 300, 300 };
    double radius[5] = { 0.4, 0.3, 0.2, 0.15, 0.1 };
    double probs[5] = { 0.9, 0.8, 0.7, 0.6, 0.5 };
    this->feats.clear();
    this->radius.clear();
    this->probs.clear();
    for (int i = 0; i < T; i++) {
        this->feats.push_back(feats[i]);
        this->radius.push_back(radius[i]);
        this->probs.push_back(radius[i]);
    }
    train_txt = "../data/train.txt";
    test_txt = "../data/test.txt";
    nega_txt = "../data/nega.txt";
}

} // namespace jda
