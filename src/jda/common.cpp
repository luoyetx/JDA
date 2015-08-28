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
}

} // namespace jda
