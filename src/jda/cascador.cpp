#include "jda/cascador.hpp"

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
}

// **TODO** implement train
void JoinCascador::Train() {

}

} // namespace jda
