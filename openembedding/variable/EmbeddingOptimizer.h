#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_OPTMIZER_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_OPTMIZER_H

#include "DataType.h"
#include "Factory.h"
#include "eigen3/Eigen/Core"
namespace paradigm4 {
namespace pico {
namespace embedding {

template<class T>
using EigenView = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;

template<class T>
using ConstEigenView = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

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
};


// must be stateless
template<class T>
class EmbeddingDefaultOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "default"; }

    size_t state_dim(size_t)override {
        return 0;
    }

    void train_init(OptimizerStateView<T>)override {
        return;
    }

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients) {
        size_t dim = state_view.embedding_dim();
        ConstEigenView<T> grad(gradients, dim);
        EigenView<T> weight(weights, dim);

        if (learning_rate != 0) {
            weight -= learning_rate * grad;
        }
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0);
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

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients) {
        size_t dim = state_view.embedding_dim();
        ConstEigenView<T> grad(gradients, dim);
        EigenView<T> weight(weights, dim);
        _temp.resize(dim);
        
        EigenView<T> accum(state_view[0], dim);
        EigenView<T> accum_update(state_view[1], dim);
        EigenView<T> update(_temp.data(), dim);

        accum = accum * rho + grad * grad * (1 - rho);
        update = grad * (accum_update + epsilon).sqrt() / (accum + epsilon).sqrt();
        accum_update = accum_update * rho + update * update * (1 - rho);
        weight -= learning_rate * update;
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0.001);
    CONFIGURE_PROPERTY(T, rho, 0.95);
    CONFIGURE_PROPERTY(T, epsilon, 1e-7);

private:
    core::vector<T> _temp;
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

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients) {
        size_t dim = state_view.embedding_dim();
        ConstEigenView<T> grad(gradients, dim);
        EigenView<T> weight(weights, dim);

        EigenView<T> accum(state_view[0], dim);
        accum += grad * grad;
        weight -= learning_rate * grad / (accum.sqrt() + epsilon);
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

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients) {
        size_t dim = state_view.embedding_dim();
        ConstEigenView<T> grad(gradients, dim);
        EigenView<T> weight(weights, dim);

        EigenView<T> m_t(state_view[0], dim);
        EigenView<T> v_t(state_view[1], dim);

        T& beta_1_t = state_view[2][0];
        T& beta_2_t = state_view[2][1];
        beta_1_t *= beta_1;
        beta_2_t *= beta_2;
        T lr_t = learning_rate * std::sqrt(1 - beta_2_t) / (1 - beta_1_t);
        m_t = m_t * beta_1 + grad * (1 - beta_1);
        v_t = v_t * beta_2 + grad * grad * (1 - beta_2); 
        weight -= lr_t * m_t / (v_t.sqrt() + epsilon);
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

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients) {
        size_t dim = state_view.embedding_dim();
        ConstEigenView<T> grad(gradients, dim);
        EigenView<T> weight(weights, dim);
        
        EigenView<T> m_t(state_view[0], dim);
        EigenView<T> v_t(state_view[1], dim);
        T& beta_1_t = state_view[2][0];
        beta_1_t *= beta_1;
        T lr_t = learning_rate / (1 - beta_1_t);
        m_t = m_t * beta_1 + grad * (1 - beta_1);
        v_t = grad.abs().max(v_t * beta_2);
        weight -= lr_t * m_t / (v_t + epsilon);
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
    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients) {
        size_t dim = state_view.embedding_dim();
        ConstEigenView<T> grad(gradients, dim);
        EigenView<T> weight(weights, dim);
        _temp1.resize(dim);
        _temp2.resize(dim);
        _temp3.resize(dim);
        
        EigenView<T> accum(state_view[0], dim);
        EigenView<T> linear(state_view[1], dim);
        EigenView<T> g(_temp1.data(), dim);
        EigenView<T> sigma(_temp2.data(), dim);
        EigenView<T> accum_new(_temp3.data(), dim);
        
        T adjusted_l2_regularization_strength = l2_regularization_strength + beta / learning_rate / 2;
        g = grad + 2 * l2_shrinkage_regularization_strength * weight;
        accum_new = accum + grad * grad; // or accum + g * g ?
        if (learning_rate_power == -0.5) {
            sigma = (accum_new.sqrt() - accum.sqrt()) / learning_rate;
            linear += g - sigma * weight;
            accum = accum_new;

            auto quadratic = accum.sqrt() / learning_rate + 2 * adjusted_l2_regularization_strength;
            auto l1_reg_adjust = linear.min(l1_regularization_strength).max(-l1_regularization_strength);
            weight = (l1_reg_adjust - linear) / quadratic;
        } else {
            T p = -learning_rate_power;
            sigma = (accum_new.pow(p) - accum.pow(p)) / learning_rate;
            linear += g - sigma * weight;
            accum = accum_new;
            
            auto quadratic = accum.pow(p) / learning_rate + 2 * adjusted_l2_regularization_strength;
            auto l1_reg_adjust = linear.min(l1_regularization_strength).max(-l1_regularization_strength);
            weight = (l1_reg_adjust - linear) / quadratic;
        }
        
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0.001);
    CONFIGURE_PROPERTY(T, initial_accumulator_value, 0.1); // beta is approximate
    CONFIGURE_PROPERTY(T, l1_regularization_strength, 0.0);
    CONFIGURE_PROPERTY(T, l2_regularization_strength, 0.0);
    CONFIGURE_PROPERTY(T, l2_shrinkage_regularization_strength, 0.0);
    CONFIGURE_PROPERTY(T, learning_rate_power, -0.5); // from tensorflow
    CONFIGURE_PROPERTY(T, beta, 0);
private:
    core::vector<T> _temp1, _temp2, _temp3;
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

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients) {
        size_t dim = state_view.embedding_dim();
        ConstEigenView<T> grad(gradients, dim);
        EigenView<T> weight(weights, dim);
        
        EigenView<T> accum(state_view[0], dim);
        EigenView<T> moment(state_view[1], dim);
        accum = accum * rho + grad * grad * (1 - rho);
        moment = moment * momentum + learning_rate * grad / (accum + epsilon).sqrt();
        weight -= moment;
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

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t, const T* gradients) {
        size_t dim = state_view.embedding_dim();
        ConstEigenView<T> grad(gradients, dim);
        EigenView<T> weight(weights, dim);

        EigenView<T> moment(state_view[0], dim);
        moment = moment * momentum + learning_rate * grad;
        if (nesterov) {
            weight -= moment * momentum + learning_rate * grad;
        } else {
            weight -= moment;
        }
    }

    CONFIGURE_PROPERTY(T, learning_rate, 0.01);
    CONFIGURE_PROPERTY(T, momentum, 0.0);
    CONFIGURE_PROPERTY(bool, nesterov, false);
};

// for ut
template<class T>
class EmbeddingTestOptimizer: public EmbeddingOptimizer<T> {
public:
    std::string category()override { return "test"; }

    size_t state_dim(size_t)override {
        return 2;
    }

    void train_init(OptimizerStateView<T> state_view)override {
        for (size_t i = 0; i < state_view.embedding_dim(); ++i) {
            state_view[0][0] = init;
        }
    }

    void update(T* weights, OptimizerStateView<T> state_view, uint64_t count, const T* gradients) {
        state_view[0][0] = flip - state_view[0][0];
        for (size_t i = 0; i < state_view.embedding_dim(); i++) {
            weights[i] += learning_rate * gradients[i] / count + state_view[0][0]; 
        }
    }
    CONFIGURE_PROPERTY(T, learning_rate, 0.1);
    CONFIGURE_PROPERTY(T, flip, 10000);
    CONFIGURE_PROPERTY(T, init, 0);
};


/// TODO: AdamAmsgrad
/// TODO: RMSpropCentered
/// TODO: Nadam


}
}
}

#endif
