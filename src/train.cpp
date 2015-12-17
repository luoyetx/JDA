#include <cstdio>
#include <opencv2/core/core.hpp>
#include "jda/data.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

/*!
 * \breif Train JoinCascador
 */
void train() {
  Config& c = Config::GetInstance();

  DataSet pos, neg;
  LOG("Load Positive And Negative DataSet");
  DataSet::LoadDataSet(pos, neg);

  JoinCascador joincascador;
  c.joincascador = &joincascador; // set global joincascador
  joincascador.mean_shape = pos.CalcMeanShape();
  LOG("Start training JoinCascador");
  joincascador.Train(pos, neg);
  LOG("End of JoinCascador Training");

  LOG("Saving Model");
  FILE* fd = fopen("../model/jda.model", "wb");
  JDA_Assert(fd, "Can not open the file to save the model");
  joincascador.SerializeTo(fd);
  fclose(fd);
}

/*!
 * \breif Resume Training Status of JoinCascador
 * \note may not work now
 */
void resume() {
  Config& c = Config::GetInstance();

  FILE* fd = fopen(c.tmp_model.c_str(), "rb");
  JDA_Assert(fd, "Can not open model file");

  JoinCascador joincascador;
  c.joincascador = &joincascador; // set global joincascador
  LOG("Loading Model Parameters from model file");
  joincascador.Resume(fd);
  fclose(fd);

  joincascador.current_stage_idx = c.current_stage_idx;
  joincascador.current_cart_idx = c.current_cart_idx;

  DataSet pos, neg;
  LOG("Load Positive And Negative DataSet");
  DataSet::LoadDataSet(pos, neg);
  joincascador.mean_shape = pos.CalcMeanShape();

  LOG("Forward Positive DataSet");
  DataSet pos_remain;
  const int pos_size = pos.size;
  pos_remain.imgs.reserve(pos_size);
  pos_remain.imgs_half.reserve(pos_size);
  pos_remain.imgs_quarter.reserve(pos_size);
  pos_remain.gt_shapes.reserve(pos_size);
  pos_remain.current_shapes.reserve(pos_size);
  pos_remain.scores.reserve(pos_size);
  pos_remain.weights.reserve(pos_size);
  // remove tf data points, update score and shape
  for (int i = 0; i < pos_size; i++) {
    bool is_face = joincascador.Validate(pos.imgs[i], pos.scores[i], pos.current_shapes[i]);
    if (is_face) {
      pos_remain.imgs.push_back(pos.imgs[i]);
      pos_remain.imgs_half.push_back(pos.imgs_half[i]);
      pos_remain.imgs_quarter.push_back(pos.imgs_quarter[i]);
      pos_remain.gt_shapes.push_back(pos.gt_shapes[i]);
      pos_remain.current_shapes.push_back(pos.current_shapes[i]);
      pos_remain.scores.push_back(pos.scores[i]);
      pos_remain.weights.push_back(pos.weights[i]);
    }
  }
  pos_remain.is_pos = true;
  pos_remain.is_sorted = false;
  pos_remain.size = pos_remain.imgs.size();

  LOG("Start Resume Training Status from %dth stage", joincascador.current_stage_idx);
  joincascador.Train(pos_remain, neg);
  LOG("End of JoinCascador Training");

  LOG("Saving Model");
  fd = fopen("../model/jda.model", "wb");
  JDA_Assert(fd, "Can not open the file to save the model");
  joincascador.SerializeTo(fd);
  fclose(fd);
}
