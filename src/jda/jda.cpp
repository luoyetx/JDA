#include <cstdio>
#include "jda/jda.hpp"

using namespace cv;
using namespace std;

namespace jda {

void jdaCascador::SerializeFrom(FILE* fd) {
    // **TODO** read parameters from fd
}

int jdaCascador::Detect(Mat& img, vector<Rect>& rects, \
                        vector<double>& scores, vector<Mat_<double> >& shapes) {
    // **TODO**
    return 0;
}

} // namespace jda
