#ifndef CASCADOR_HPP_
#define CASCADOR_HPP_

#include <vector>

namespace jda {

// pre-define
class BoostCart;

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
    void Train(DataSet& pos, DataSet& neg);

public:
    int T; // number of stages

    std::vector<BoostCart> btcarts;
};

} // namespace jda

#endif // CASCADOR_HPP_
