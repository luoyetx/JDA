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

#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "jda/data.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

/*!
 * \breif Test JoinCascador Face Detection over FDDB
 */
void fddb() {
  const Config& c = Config::GetInstance();
  JoinCascador joincascador;
  FILE* fd = fopen("../model/jda.model", "rb");
  joincascador.SerializeFrom(fd);
  // make sure you set these two variables in `config.json` correctly
  joincascador.current_stage_idx = c.current_stage_idx;
  joincascador.current_cart_idx = c.current_cart_idx;
  fclose(fd);

  JDA_Assert(EXISTS("../data/fddb"), "No fddb data!");

  string prefix = "../data/fddb/images/";
  string fddb = "../data/fddb/FDDB-folds/FDDB-fold-01.txt";
  string fddb_out = "../data/fddb/result/fold-01-out.txt";

  FILE* fin = fopen(fddb.c_str(), "r");
  JDA_Assert(fin, "Can not open FDDB-fold-01.txt");
  FILE* fout = fopen(fddb_out.c_str(), "w");
  JDA_Assert(fin, "Can not open fold-01-out.txt");

  char path[300];
  while (fscanf(fd, "%s", path) > 0) {
    string full_path = prefix + string(path) + string(".jpg");
    Mat img = imread(full_path, CV_LOAD_IMAGE_GRAYSCALE);
    if (!img.data) {
      LOG("Can not open %s, Skip it", full_path);
    }
    vector<double> scores;
    vector<Rect> rects;
    vector<Mat_<double> > shapes;
    joincascador.Detect(img, rects, scores, shapes);

    const int n = rects.size();
    fprintf(fout, "%s\n%d\n", path, n);
    LOG("%s get %d faces", path, n);

    for (int i = 0; i < n; i++) {
      Rect& r = rects[i];
      double s = scores[i];
      fprintf(fout, "%d %d %d %d %lf\n", r.x, r.y, r.width, r.height, s);
    }
  }

  fclose(fin);
  fclose(fout);
}
