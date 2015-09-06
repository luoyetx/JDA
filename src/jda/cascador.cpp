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
    boost_carts.resize(T);
    for (int i = 0; i < T; i++) {
        boost_carts[i].set_joincascador(this);
    }
}

// **TODO** implement train
void JoinCascador::Train() {

}

} // namespace jda
