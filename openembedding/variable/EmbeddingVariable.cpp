#include "Meta.h"
#include "EmbeddingOptimizerVariable.h"
#include "EmbeddingVariable.h"

#ifdef USE_DCPMM
#include "PersistentEmbeddingOptimizerVariable.h"
#endif

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class T>
class EmbeddingVariable: public EmbeddingVariableBase {
    using key_type = uint64_t;
    using Entity = EmbeddingOptimizerVariableInterface<key_type, T>;
public:
    EmbeddingVariable(size_t embedding_dim) {
        _entity = std::make_unique<EmbeddingOptimizerVariable<
            EmbeddingArrayTable<key_type, T>, EmbeddingDefaultOptimizer<T>>>(embedding_dim, -1);
    }

    void load_config(const core::Configure& config) {
        std::string table = _entity->embedding_table()->category();
        std::string optimizer = _entity->embedding_optimizer()->category();
        std::string initializer = _entity->embedding_initializer()->category();
        LOAD_CONFIG(config, table);
        LOAD_CONFIG(config, optimizer);
        LOAD_CONFIG(config, initializer);
        if (table != _entity->embedding_table()->category() ||
            optimizer != _entity->embedding_optimizer()->category()) {
            auto& factory = Factory<Entity, size_t, key_type>::singleton();
            std::unique_ptr<Entity> variable1 = factory.create(
                  table + "." + optimizer, _entity->embedding_dim(), -1);
            SCHECK(variable1);
            if (num_indices()) {
                SLOG(WARNING) << "Changing table or optimizer category. This operation may be expensive.";
                if (optimizer != _entity->embedding_optimizer()->category()) {
                    SLOG(WARNING) << "Optimizer category modified, the optimizer states will be reset.";
                }
            }
            if (table == _entity->embedding_table()->category()) {
                core::Configure config_from;
                _entity->embedding_table()->dump_config(config_from);
                variable1->embedding_table()->load_config(config_from);
            }
            if (optimizer == _entity->embedding_optimizer()->category()) {
                core::Configure config_from;
                _entity->embedding_optimizer()->dump_config(config_from);
                variable1->embedding_optimizer()->load_config(config_from);
            }
            variable1->embedding_table()->load_config(config[table]);
            variable1->embedding_optimizer()->load_config(config[optimizer]);
            // May init some optimizer state using new optimizer config.
            variable1->copy_from(std::move(*_entity), server_block_num_items());
            _entity = std::move(variable1);
        } else {
            _entity->embedding_table()->load_config(config[table]);
            _entity->embedding_optimizer()->load_config(config[optimizer]);
        }
        
        if (initializer != _entity->embedding_initializer()->category()) {
            _entity->embedding_initializer() =
                  Factory<EmbeddingInitializer<T>>::singleton().create(initializer);
            SCHECK(_entity->embedding_initializer());
        }
        _entity->embedding_initializer()->load_config(config[initializer]);
    }

    void dump_config(core::Configure& config) override {
        std::string table = _entity->embedding_table()->category();
        std::string optimizer = _entity->embedding_optimizer()->category();
        std::string initializer = _entity->embedding_initializer()->category();
        core::Configure table_config, optimizer_config, initializer_config;
        _entity->embedding_table()->dump_config(table_config);
        _entity->embedding_optimizer()->dump_config(optimizer_config);
        _entity->embedding_initializer()->dump_config(initializer_config);
        SAVE_CONFIG(config, table);
        SAVE_CONFIG(config, optimizer);
        SAVE_CONFIG(config, initializer);
        config.node()[table] = table_config.node();
        config.node()[optimizer] = optimizer_config.node();
        config.node()[initializer] = initializer_config.node();
    }

    void clear_weights() override {
        core::Configure config;
        dump_config(config);
        _entity = std::make_unique<EmbeddingOptimizerVariable<
            EmbeddingArrayTable<key_type, T>, EmbeddingDefaultOptimizer<T>>>(_entity->embedding_dim(), -1);
        load_config(config);
    }

    size_t server_block_num_items() override {
        size_t item_line_size = _entity->embedding_dim() * sizeof(T) + state_line_size();
        return 63 * 1024 / item_line_size + 1;
    }

    void get_weights(const key_type* indices, size_t n,
          char* weights, char* states) override {
        _entity->get_weights(indices, n,
              reinterpret_cast<T*>(weights),
              reinterpret_cast<T*>(states));
    };

    void set_weights(const key_type* indices, size_t n,
          const char* weights, const char* states) override {
        _entity->set_weights(indices, n,
              reinterpret_cast<const T*>(weights),
              reinterpret_cast<const T*>(states));
    };

    void pull_weights(const key_type* indices, size_t n,
          char* weights, VariableAsyncTask& async_task) override {
        _entity->pull_weights(indices, n,
              reinterpret_cast<T*>(weights), async_task);
        if (async_task) {
            async_task.hold_entity(_entity);
        }
    }

    void push_gradients(const key_type* indices, size_t n,
          const char* gradients, const uint64_t* counts, VariableAsyncTask& async_task) override {
        _entity->push_gradients(indices, n,
              reinterpret_cast<const T*>(gradients), counts, async_task);
        if (async_task) {
            async_task.hold_entity(_entity);
        }   
    }

    void update_weights() override {
        SCHECK(_readers.empty()) << "Should not update weights while reading.";
        _entity->update_weights();
    }

    size_t state_line_size() override {
        return _entity->embedding_optimizer()->state_dim(_entity->embedding_dim()) * sizeof(T);
    }

    size_t num_indices() override {
        return _entity->embedding_table()->num_items();
    }

    int create_reader() override {
        core::lock_guard<core::RWSpinLock> lock(_reader_lock);
        int reader_id = _next_reader_id++;
        _readers[reader_id] = _entity->create_key_reader();
        return reader_id;
    }

    size_t read_indices(int reader_id, key_type* indices, size_t n) override {
        SCHECK(_readers.count(reader_id));
        return _readers.at(reader_id)->read_keys(indices, n);
    }

    uint64_t get_reader_cursor(int reader_id) override {
        SCHECK(_readers.count(reader_id));
        return _readers.at(reader_id)->cursor();
    }

    void delete_reader(int reader_id) override {
        core::lock_guard<core::RWSpinLock> lock(_reader_lock);
        _readers.erase(reader_id);
        if (_readers.empty()) {
            _next_reader_id = 0;
        }
    }

private:
    std::shared_ptr<Entity> _entity;

    core::RWSpinLock _reader_lock;
    std::unordered_map<int, std::unique_ptr<EmbeddingVariableKeyReader<key_type>>> _readers;
    int _next_reader_id = 0;
};



template<class Optimizer>
void register_array_optimizer() {
    using key_type = uint64_t;
    using T = typename Optimizer::weight_type;
    using Table = EmbeddingArrayTable<key_type, T>;
    using Entity = EmbeddingOptimizerVariableInterface<key_type, T>;
    using Implementation = EmbeddingOptimizerVariable<Table, Optimizer>;
    auto& factory = Factory<Entity, size_t, key_type>::singleton();
    factory.template register_creator<Implementation>("array." + Optimizer().category());
}

template<class Optimizer>
void register_hash_optimizer() {
    using key_type = uint64_t;
    using T = typename Optimizer::weight_type;
    using Table = EmbeddingHashTable<key_type, T>;
    using Entity = EmbeddingOptimizerVariableInterface<key_type, T>;
    using Implementation = EmbeddingOptimizerVariable<Table, Optimizer>;
    auto& factory = Factory<Entity, size_t, key_type>::singleton();
    factory.template register_creator<Implementation>("hash." + Optimizer().category());
}

#ifdef USE_DCPMM

template<class Optimizer>
void register_pmem_optimizer() {
    using key_type = uint64_t;
    using T = typename Optimizer::weight_type;
    using Table = PersistentEmbeddingTable<key_type, T>;
    using Entity = EmbeddingOptimizerVariableInterface<key_type, T>;
    using Implementation = PersistentEmbeddingOptimizerVariable<Table, Optimizer>;
    auto& factory = Factory<Entity, size_t, key_type>::singleton();
    factory.template register_creator<Implementation>("mixpmem." + Optimizer().category());    
}

#endif

template<class Optimizer>
void register_optimizer() {
    register_array_optimizer<Optimizer>();
    register_hash_optimizer<Optimizer>();

#ifdef USE_DCPMM
    register_pmem_optimizer<Optimizer>();
#endif
}

template<class Initializer>
void register_initializer() {
    using T = typename Initializer::weight_type;
    Factory<EmbeddingInitializer<T>>::singleton()
          .template register_creator<Initializer>(Initializer().category());
}

template<class T>
void register_for_datatype() {
    register_optimizer<EmbeddingAdadeltaOptimizer<T>>();
    register_optimizer<EmbeddingAdagradOptimizer<T>>();
    register_optimizer<EmbeddingAdamOptimizer<T>>();
    register_optimizer<EmbeddingAdamaxOptimizer<T>>();
    register_optimizer<EmbeddingFtrlOptimizer<T>>();
    register_optimizer<EmbeddingRMSpropOptimizer<T>>();
    register_optimizer<EmbeddingSGDOptimizer<T>>();
    register_optimizer<EmbeddingDefaultOptimizer<T>>();
    register_optimizer<EmbeddingTestOptimizer<T>>();

    register_initializer<EmbeddingConstantInitializer<T>>();
    register_initializer<EmbeddingUniformInitializer<T>>();
    register_initializer<EmbeddingNormalInitializer<T>>();
}

class EmbeddingVariableCreator {
public:
    static EmbeddingVariableCreator& singleton() {
        static EmbeddingVariableCreator creator;
        return creator;
    }

    template<class T>
    void operator()(TypeCase<T>, size_t embedding_dim, std::unique_ptr<EmbeddingVariableBase>& variable) {
        variable = std::make_unique<EmbeddingVariable<T>>(embedding_dim);
    }

    static std::unique_ptr<EmbeddingVariableBase> create(DataType datatype, size_t embedding_dim) {
        std::unique_ptr<EmbeddingVariableBase> variable;
        datatype.invoke(EmbeddingVariableCreator(), embedding_dim, variable);
        SCHECK(variable) << "unknown datatype: " << datatype.to_string();
        return variable;
    };

private:
    EmbeddingVariableCreator() {
        register_for_datatype<float>();
        register_for_datatype<double>();
    }
};

std::unique_ptr<EmbeddingVariableBase> EmbeddingVariableBase::create(DataType datatype, size_t embedding_dim) {
    std::unique_ptr<EmbeddingVariableBase> variable;
    datatype.invoke(EmbeddingVariableCreator::singleton(), embedding_dim, variable);
    SCHECK(variable) << "unknown datatype: " << datatype.to_string();
    return variable;
}


}
}
}