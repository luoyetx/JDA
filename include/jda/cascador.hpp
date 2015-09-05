#ifndef CASCADOR_HPP_
#define CASCADOR_HPP_

#include <vector>
#include "jda/cart.hpp"

namespace jda {

/**
 * JoinCascador for face classification and landmark regression
 */
class JoinCascador {
public:
    JoinCascador();
    ~JoinCascador();
    JoinCascador(const JoinCascador& other);
    JoinCascador& operator=(const JoinCascador& other);
    void Initialize(int T);

public:
    /**
     * Train JoinCascador
     *
     * See Full Algorithm on paper `Algorithm 3`
     */
    void Train();

public:
    int T; // number of stages

    std::vector<BoostCart> boost_carts;
};

} // namespace jda

#endif // CASCADOR_HPP_
