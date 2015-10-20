#include <cstdio>
#include <opencv2/core/core.hpp>
#include "jda/data.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

/**
 * Train JoinCascador
 */
void train() {
    const Config& c = Config::GetInstance();

    DataSet pos, neg;
    LOG("Load Positive And Negative DataSet");
    DataSet::LoadDataSet(pos, neg);

    JoinCascador joincascador;
    joincascador.Initialize(c.T);
    joincascador.mean_shape = pos.CalcMeanShape();
    neg.set_joincascador(&joincascador);
    LOG("Start training JoinCascador");
    joincascador.Train(pos, neg);
    LOG("End of JoinCascador Training");

    LOG("Saving Model");
    FILE* fd = fopen("../model/jda.model", "wb");
    JDA_Assert(fd, "Can not open the file to save the model");
    joincascador.SerializeTo(fd);
    fclose(fd);
}

/**
 * Resume Training Status of JoinCascador
 */
void resume() {
    printf("Attention! The stage you resume from should range in [2, c.T]\n");
    printf("Please input the stage you want to resume from: ");
    int stage;
    char buff[256];
    scanf("%d", &stage);
    printf("The model file should be ../model/jda_tmp_{time}_stage%d.model", stage - 1);
    printf("Please input the model file path: ");
    scanf("%s", buff);
    printf("\n");

    FILE* fd = fopen(buff, "rb");
    JDA_Assert(fd, "Can not open model file");

    JoinCascador joincascador;
    LOG("Loading Model Parameters from model file");
    joincascador.ResumeFrom(stage, fd);
    fclose(fd);

    DataSet pos, neg;
    LOG("Load Positive And Negative DataSet");
    DataSet::LoadDataSet(pos, neg);
    joincascador.mean_shape = pos.CalcMeanShape();
    neg.set_joincascador(&joincascador);

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

    LOG("Start Resume Training Status from %dth stage", stage + 1);
    joincascador.Train(pos_remain, neg);
    LOG("End of JoinCascador Training");

    LOG("Saving Model");
    fd = fopen("../model/jda.model", "wb");
    JDA_Assert(fd, "Can not open the file to save the model");
    joincascador.SerializeTo(fd);
    fclose(fd);
}
