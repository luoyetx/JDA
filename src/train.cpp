#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "jda/data.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

/*!
 * \brief Train JoinCascador
 */
void train() {
  Config& c = Config::GetInstance();

  // can we load training data from a binary file
  bool flag = false;

  JoinCascador joincascador;
  joincascador.current_stage_idx = 0;
  joincascador.current_cart_idx = -1;
  c.joincascador = &joincascador; // set global joincascador

  DataSet pos, neg;
  char data_file[] = "../data/jda_train_data.data";
  if (EXISTS(data_file)) {
    LOG("Load Positive And Negative DataSet from %s", data_file);
    DataSet::Resume(data_file, pos, neg);
  }
  else {
    LOG("Load Positive And Negative DataSet");
    DataSet::LoadDataSet(pos, neg);
    DataSet::Snapshot(pos, neg);
  }

  joincascador.mean_shape = pos.mean_shape;
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
 * \brief Resume Training Status of JoinCascador
 * \note may not work now
 */
void resume() {
  Config& c = Config::GetInstance();

  FILE* fd = fopen(c.resume_model.c_str(), "rb");
  JDA_Assert(fd, "Can not open model file");

  JoinCascador joincascador;
  c.joincascador = &joincascador; // set global joincascador
  LOG("Loading Model Parameters from model file");
  joincascador.Resume(fd);
  fclose(fd);

  DataSet pos, neg;
  LOG("Load Positive And Negative DataSet from %s", c.resume_data.c_str());
  DataSet::Resume(c.resume_data.c_str(), pos, neg);

  LOG("Start Resume Training Status from %dth stage", joincascador.current_stage_idx);
  joincascador.Train(pos, neg);
  LOG("End of JoinCascador Training");

  LOG("Saving Model");
  fd = fopen("../model/jda.model", "wb");
  JDA_Assert(fd, "Can not open the file to save the model");
  joincascador.SerializeTo(fd);
  fclose(fd);
}

void dump() {
  DataSet pos, neg;
  char data_file[] = "../data/jda_train_data.data";
  if (EXISTS(data_file)) {
    LOG("Load Positive And Negative DataSet from %s", data_file);
    DataSet::Resume(data_file, pos, neg);
  }
  pos.Dump("../data/dump/pos");
  neg.Dump("../data/dump/neg");
}
