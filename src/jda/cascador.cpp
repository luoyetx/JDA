#include "jda/jda.hpp"

namespace jda {

JoinCascador::JoinCascador() {}
JoinCascador::~JoinCascador() {}
JoinCascador::JoinCascador(const JoinCascador& other) {}
JoinCascador& JoinCascador::operator=(const JoinCascador& other) {
    if (this == &other) return *this;
    return *this;
}
void JoinCascador::Initialize(int T) {
    this->T = T;
    btcarts.resize(T);
    for (int t = 0; t < T; t++) {
        btcarts[t].Initialize(t);
        btcarts[t].set_joincascador(this);
    }
}

void JoinCascador::Train(DataSet& pos, DataSet& neg) {
    for (int t = 0; t < T; t++) {
        LOG("Train %dth stages", t + 1);
        TIMER_BEGIN
            btcarts[t].Train(pos, neg);
            LOG("End of train %dth stages, costs %.4lf s", t + 1, TIMER_NOW);
        TIMER_END
    }
}

} // namespace jda
