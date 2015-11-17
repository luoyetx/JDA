#include <cstdio>
#include <opencv2/core/core.hpp>
#include "jda/data.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

/*!
 * \breif Test JoinCascador Model over Test DataSet
 */
void test() {
  const Config& c = Config::GetInstance();
  JoinCascador joincascador;
  JoinCascador faker;
  FILE* fd = fopen("../model/jda.model", "rb");
  JDA_Assert(fd, "Can not open `../model/jda.model`");
  joincascador.SerializeFrom(fd);
  fclose(fd);

  DataSet pos, neg;
  pos.LoadPositiveDataSet(c.test_pos_txt);
  neg.LoadNegativeDataSet(c.test_neg_txt);
  faker.current_stage_idx = -1;
  faker.current_cart_idx = -1;
  neg.MoreNegSamples(pos.size, 2.);

  LOG("Test JoinCascador");
  LOG("We have %d Positive Samples and %d Negative Samples", pos.size, neg.size);

  int accept = 0;
  for (int i = 0; i < pos.size; i++) {
    bool is_face = joincascador.Validate(pos.imgs[i], pos.scores[i], pos.current_shapes[i]);
    if (is_face) {
      accept++;
    }
  }
  double tp = static_cast<double>(accept) / static_cast<double>(pos.size) * 100.;
  LOG("True Positive Rate = %.2lf%%", tp);
  double e = calcMeanError(pos.gt_shapes, pos.current_shapes);
  LOG("Shape Mean Error = %.4lf", e);

  int reject = 0;
  for (int i = 0; i < neg.size; i++) {
    bool is_face = joincascador.Validate(neg.imgs[i], neg.scores[i], neg.current_shapes[i]);
    if (!is_face) {
      reject++;
    }
  }
  double fp = (1 - static_cast<double>(reject) / static_cast<double>(neg.size)) * 100.;
  LOG("False Positive Rate = %.2lf%%", fp);
  LOG("Done");
}

/*!
 * \breif Test JoinCascador Face Detection over FDDB
 */
void fddb() {
  // **TODO** Run FDDB
  dieWithMsg("Not completed Yet!");
}
