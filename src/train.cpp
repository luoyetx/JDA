#include <cstdio>
#include <opencv2/core/core.hpp>
#include "jda/data.hpp"
#include "jda/common.hpp"
#include "jda/cascador.hpp"

using namespace cv;
using namespace std;
using namespace jda;

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

void resume() {
    // **TODO** resume
}
