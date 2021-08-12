#ifndef PARADIGM4_HYPEREMBEDDING_FACTORY_H
#define PARADIGM4_HYPEREMBEDDING_FACTORY_H

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <pico-core/Configure.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class T>
void LOAD_CONFIG_load_config(const core::Configure& config, const std::string& key, T& value) {
    if (config.has(key)) {
        value = config.get<T>(key, value);
    }
}

template<class T>
void SAVE_CONFIG_save_config(core::Configure& config, const std::string& key, const T& value) {    
    config.node()[key] = value;
}


#define LOAD_CONFIG(config, x) do { \
    LOAD_CONFIG_load_config((config), #x, (x)); \
} while(0)

#define SAVE_CONFIG(config, x) do { \
    SAVE_CONFIG_save_config((config), #x, (x)); \
} while(0)


class Configurable {
public:
    Configurable() {}
    ~Configurable() {}
    Configurable(const Configurable&) = delete;
    Configurable& operator=(Configurable) = delete;

    virtual void dump_config(core::Configure& config)const {
        for (auto& dumper: _inner_dumpers) {
            dumper(config);
        }
    }

    virtual void load_config(const core::Configure& config) {
        for (auto& loader: _inner_loaders) {
            loader(config);
        }
        core::Configure self;
        dump_config(self);

        bool has_default = false;
        core::Configure defaults;
        for (auto pair: self.node()) {
            std::string key = pair.first.as<std::string>();
            if (!config.has(key)) {
                has_default = true;
                defaults.node()[key] = self.node()[key];
            }
        }
        if (has_default) {
            SLOG(INFO) << "using default configure: \n" << defaults.dump();
        }

        bool has_unknown = false;
        core::Configure unknowns;
        for (auto pair: config.node()) {
            std::string key = pair.first.as<std::string>();
            if (!self.has(key)) {
                has_unknown = true;
                unknowns.node()[key] = config.node()[key];
            }
        }
        if (has_unknown) {
            SLOG(WARNING) << "unknown configure: \n" << unknowns.dump();
        }
    }

protected:
    std::vector<std::function<void(core::Configure&)> > _inner_dumpers;
    std::vector<std::function<void(const core::Configure&)> > _inner_loaders;
};

template<class T>
struct CONFIGURE_PROPERTY_LOADER {
    CONFIGURE_PROPERTY_LOADER(const char* key, T* p): key(key), p(p) {}
    void operator()(const core::Configure& config) {
        LOAD_CONFIG_load_config(config, key, *p);
    }
    const char* key;
    T* p;
};

template<class T>
struct CONFIGURE_PROPERTY_DUMPER {
    CONFIGURE_PROPERTY_DUMPER(const char* key, const T* p): key(key), p(p) {}
    void operator()(core::Configure& config) {
        SAVE_CONFIG_save_config(config, key, *p);
    }
    const char* key;
    const T* p;
};


#define CONFIGURE_PROPERTY(type, name, default_value)\
    public:\
        type name = (default_value);\
    private:\
        bool name##_loader_dummy = (this->_inner_loaders.push_back(\
              CONFIGURE_PROPERTY_LOADER<type>(#name, &this->name)), true);\
        bool name##_dumper_dummy = (this->_inner_dumpers.push_back(\
              CONFIGURE_PROPERTY_DUMPER<type>(#name, &this->name)), true);\


template<class Base>
class Factory {
public:
    typedef std::function<std::unique_ptr<Base>(const core::Configure&)> creator_type;
    virtual ~Factory() {}

    template<class Impl>
    bool register_creator(const std::string& category) {
        return _creators.emplace(category, Factory<Base>::creator<Impl>).second;
    }

    std::unique_ptr<Base> create(const std::string& category, const core::Configure& config)const {
        if (_creators.count(category)) {
            return _creators.at(category)(config);
        } else {
            return nullptr;
        }
    }

    static Factory<Base>& singleton() {
        static Factory<Base> factory;
        return factory;
    }
private:
    template<class Impl>
    static std::unique_ptr<Base> creator(const core::Configure& config) {
        std::unique_ptr<Base> obj = std::make_unique<Impl>();
        if (obj) {
            obj->load_config(config);
        }
        return obj;
    }
    std::map<std::string, creator_type> _creators;
};

}
}
}

#endif