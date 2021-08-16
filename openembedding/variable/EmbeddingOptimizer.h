#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_OPTMIZER_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_OPTMIZER_H

#include "DataType.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class T>
class OptimizerStateView {
public:
    OptimizerStateView(T* buffer, size_t embedding_dim)
        : _buffer(buffer), _n(embedding_dim) {}

    size_t embedding_dim()const {
        return _n;
    }

    T* operator[](size_t i)const {
        return _buffer + i * _n;
    }

private:
    T* _buffer = nullptr;
    size_t _n = 0;
};

template<class T>
class EmbeddingOptimizer: public Configurable {
public:
    using weight_type = T;
    virtual std::string category() = 0;
    virtual size_t state_dim(size_t embedding_dim) = 0;
    virtual void train_init(OptimizerStateView<T> state_view) = 0;
    virtual void update(T* weights, OptimizerStateView<T> state_view, uint64_t count, const T* gradients) = 0;
};


template<class T>
class EmbeddingAdadeltaOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "adadelta"; }

    size_t state_dim(size_t embedding_dim) {
        return embedding_dim * 2;
    }

    void train_init(OptimizerStateView<T> state_view)override {
        for (size_t i = 0; i < state_view.embedding_dim(); ++i) {
            state_view[0][i] = 0;
            state_view[1][i] = 0;
        }
    }

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients)override {
        for (size_t i = 0; i < state_view.embedding_dim(); i++) {
            T& weight = weights[i];
            T& accum = state_view[0][i];
            T& accum_update = state_view[1][i];
            T grad = gradients[i];

            accum = accum * rho + grad * grad * (1 - rho);
            T update = grad * sqrt(accum_update + epsilon) / sqrt(accum + epsilon);
            accum_update = accum_update * rho + update * update * (1 - rho);
            weight -= learning_rate * update;
        }
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0.001);
    CONFIGURE_PROPERTY(T, rho, 0.95);
    CONFIGURE_PROPERTY(T, epsilon, 1e-7);
};


template<class T>
class EmbeddingAdagradOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "adagrad"; }

    size_t state_dim(size_t embedding_dim)override {
        return embedding_dim;
    }

    void train_init(OptimizerStateView<T> state_view)override {
        for (size_t i = 0; i < state_view.embedding_dim(); ++i) {
            state_view[0][i] = initial_accumulator_value;
        }
    }

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients)override {
        for (size_t i = 0; i < state_view.embedding_dim(); i++) {
            T& weight = weights[i];
            T& accum = state_view[0][i];
            T grad = gradients[i];
            accum += grad * grad;
            weight -= learning_rate * grad / sqrt(accum); //sqrt(accum + epsilon)
        }
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0.001);
    CONFIGURE_PROPERTY(T, initial_accumulator_value, 0.1);
    CONFIGURE_PROPERTY(T, epsilon, 1e-7);
};


template<class T>
class EmbeddingAdamOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "adam"; }

    size_t state_dim(size_t embedding_dim)override {
        return embedding_dim * 2 + 2;
    }

    void train_init(OptimizerStateView<T> state_view)override {
        for (size_t i = 0; i < state_view.embedding_dim(); ++i) {
            state_view[0][i] = 0.0;
            state_view[1][i] = 0.0;
        }
        state_view[2][0] = 1.0;
        state_view[2][1] = 1.0;
    }

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients)override {
        T& beta_1_t = state_view[2][0];
        T& beta_2_t = state_view[2][1];
        beta_1_t *= beta_1;
        beta_2_t *= beta_2;
        for (size_t i = 0; i < state_view.embedding_dim(); i++) {
            T& weight = weights[i];
            T& m_t = state_view[0][i];
            T& v_t = state_view[1][i];
            T grad = gradients[i];
            T lr_t = learning_rate * std::sqrt(1 - beta_2_t) / (1 - beta_1_t);
            m_t = m_t * beta_1 + grad * (1 - beta_1);
            v_t = v_t * beta_2 + grad * grad * (1 - beta_2); 
            weight -= lr_t * m_t / (std::sqrt(v_t) + epsilon);
        }
    }
    
    CONFIGURE_PROPERTY(T, learning_rate, 0.001);
    CONFIGURE_PROPERTY(T, beta_1, 0.9);
    CONFIGURE_PROPERTY(T, beta_2, 0.999);
    CONFIGURE_PROPERTY(T, epsilon, 1e-7);
};


template<class T>
class EmbeddingAdamaxOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "adamax"; }

    size_t state_dim(size_t embedding_dim)override {
        return embedding_dim * 2 + 1;
    }

    void train_init(OptimizerStateView<T> state_view)override {
        for (size_t i = 0; i < state_view.embedding_dim(); ++i) {
            state_view[0][i] = 0.0;
            state_view[1][i] = 0.0;
        }
        state_view[2][0] = 1.0;
    }

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients)override {
        T& beta_1_t = state_view[2][0];
        beta_1_t *= beta_1;
        for (size_t i = 0; i < state_view.embedding_dim(); i++) {
            T& weight = weights[i];
            T& m_t = state_view[0][i];
            T& v_t = state_view[1][i];
            T grad = gradients[i];
            T lr_t = learning_rate / (1 - beta_1_t);
            m_t = m_t * beta_1 + grad * (1 - beta_1);
            v_t = std::max(v_t * beta_2, std::abs(grad));
            weight -= lr_t * m_t / (v_t + epsilon);
        }
    }
    
    CONFIGURE_PROPERTY(T, learning_rate, 0.001);
    CONFIGURE_PROPERTY(T, beta_1, 0.9);
    CONFIGURE_PROPERTY(T, beta_2, 0.999);
    CONFIGURE_PROPERTY(T, epsilon, 1e-7);
};


template<class T>
class EmbeddingFtrlOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "ftrl"; }

    T invpow(T a) {
        return learning_rate_power == -0.5 ? std::sqrt(a) : std::pow(a, -learning_rate_power);
    }

    size_t state_dim(size_t embedding_dim)override {
        return embedding_dim * 2;
    }

    void train_init(OptimizerStateView<T> state_view)override {
        for (size_t i = 0; i < state_view.embedding_dim(); ++i) {
            state_view[0][i] = initial_accumulator_value;
            state_view[1][i] = 0.0;
        }
    }

    // Pay attention to the signs of grad and z.
    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients)override {
        for (size_t i = 0; i < state_view.embedding_dim(); i++) {
            T& weight = weights[i];
            T& accum = state_view[0][i];
            T& linear = state_view[1][i];
            T grad = gradients[i] + 2 * l2_shrinkage_regularization_strength * weight;
            T sigma = (invpow(accum + grad * grad) - invpow(accum)) / learning_rate;
            accum += grad * grad;
            linear += grad - sigma * weight;
            
            T quadratic = invpow(accum) / learning_rate + 2 * l2_regularization_strength;
            T l1_reg_adjust = std::max(std::min(linear, l1_regularization_strength), -l1_regularization_strength);
            weight = (l1_reg_adjust - linear) / quadratic;
        }
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0.001);
    CONFIGURE_PROPERTY(T, initial_accumulator_value, 0.1); // beta is approximate
    CONFIGURE_PROPERTY(T, l1_regularization_strength, 0.0);
    CONFIGURE_PROPERTY(T, l2_regularization_strength, 0.0);
    CONFIGURE_PROPERTY(T, l2_shrinkage_regularization_strength, 0.0);
    CONFIGURE_PROPERTY(T, learning_rate_power, -0.5); // from tensorflow
};


template<class T>
class EmbeddingRMSpropOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "rmsprop"; }

    size_t state_dim(size_t embedding_dim)override {
        return embedding_dim * 2;
    }

    void train_init(OptimizerStateView<T> state_view)override {
        for (size_t i = 0; i < state_view.embedding_dim(); ++i) {
            state_view[0][i] = 0.0;
            state_view[1][i] = 0.0;
        }
    }

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients)override {
        for (size_t i = 0; i < state_view.embedding_dim(); i++) {
            T& weight = weights[i];
            T& accum = state_view[0][i];
            T& moment = state_view[1][i];
            T grad = gradients[i];

            accum = accum * rho + grad * grad * (1 - rho);
            moment = moment * momentum + learning_rate * grad / sqrt(accum + epsilon);
            weight -= moment;
        }
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0.001);
    CONFIGURE_PROPERTY(T, rho, 0.9);
    CONFIGURE_PROPERTY(T, momentum, 0.0);
    CONFIGURE_PROPERTY(T, epsilon, 1e-7);
};


template<class T>
class EmbeddingSGDOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "sgd"; }

    size_t state_dim(size_t embedding_dim)override {
        return embedding_dim;
    }

    void train_init(OptimizerStateView<T> state_view)override {
        for (size_t i = 0; i < state_view.embedding_dim(); ++i) {
            state_view[0][i] = 0.0;
        }
    }

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients)override {
        for (size_t i = 0; i < state_view.embedding_dim(); i++) {
            T& weight = weights[i];
            T& moment = state_view[0][i];
            T grad = gradients[i];
            moment = moment * momentum + learning_rate * grad;
            if (nesterov) {
                weight -= moment * momentum + learning_rate * grad;
            } else {
                weight -= moment;
            }
        }
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0.01);
    CONFIGURE_PROPERTY(T, momentum, 0.0);
    CONFIGURE_PROPERTY(bool, nesterov, false);
};



/// TODO: AdamAmsgrad
/// TODO: RMSpropCentered
/// TODO: Nadam


}
}
}

#endif
