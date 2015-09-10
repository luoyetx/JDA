#include <opencv2/core/core.hpp>
#include "jda/jda.hpp"

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
    LOG("Start training JoinCascador");
    joincascador.Train(pos, neg);
    LOG("End of JoinCascador Training");
}
