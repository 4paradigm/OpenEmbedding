#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_INITIALIZER_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_INITIALIZER_H

#include "DataType.h"
#include "Factory.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class T>
class EmbeddingInitializer: public Configurable {
public:
    using weight_type = T;
    virtual std::string category() = 0;
    virtual void train_init(T* weights, size_t embedding_dim) = 0;
};

template<class T>
class EmbeddingConstantInitializer: public EmbeddingInitializer<T> {
public:
    std::string category()override { return "constant"; }

    void train_init(T* weights, size_t embedding_dim) override {
        for (size_t i = 0; i < embedding_dim; ++i) {
            weights[i] = value;
        }
    }

private:
    CONFIGURE_PROPERTY(T, value, 0.0);
};


template<class T>
class EmbeddingUniformInitializer: public EmbeddingInitializer<T> {
public:
    std::string category()override { return "uniform"; }

    void load_config(const core::Configure& config) override {
        EmbeddingInitializer<T>::load_config(config);
        device = std::make_unique<std::random_device>();
        engine = std::make_unique<std::default_random_engine>((*device)());
        distribution = std::make_unique<std::uniform_real_distribution<T>>(minval, maxval);
    }

    void train_init(T* weights, size_t embedding_dim) override {
        for (size_t i = 0; i < embedding_dim; ++i) {
            weights[i] = (*distribution)(*engine);
        }
    }

private:
    CONFIGURE_PROPERTY(T, minval, 0.0);
    CONFIGURE_PROPERTY(T, maxval, 1.0);
    std::unique_ptr<std::random_device> device;
    std::unique_ptr<std::default_random_engine> engine;
    std::unique_ptr<std::uniform_real_distribution<T>> distribution;
};

template<class T>
class EmbeddingNormalInitializer: public EmbeddingInitializer<T> {
public:
    std::string category()override { return "normal"; }

    void load_config(const core::Configure& config) override {
        EmbeddingInitializer<T>::load_config(config);
        device = std::make_unique<std::random_device>();
        engine = std::make_unique<std::default_random_engine>((*device)());
        distribution = std::make_unique<std::normal_distribution<T>>(mean, stddev);
    }

    void train_init(T* weights, size_t embedding_dim) override {
        for (size_t i = 0; i < embedding_dim; ++i) {
            weights[i] = (*distribution)(*engine);
            if (truncated > 0.1) {
                while ((weights[i] - mean) / stddev > truncated) {
                    weights[i] = (*distribution)(*engine);
                }
            }
        }
    }

private:
    CONFIGURE_PROPERTY(T, mean, 0.0);
    CONFIGURE_PROPERTY(T, stddev, 1.0);
    CONFIGURE_PROPERTY(T, truncated, 0.0);
    std::unique_ptr<std::random_device> device;
    std::unique_ptr<std::default_random_engine> engine;
    std::unique_ptr<std::normal_distribution<T>> distribution;
};

}
}
}

#endif
