#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_META_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_META_H

#include <pico-core/Archive.h>
#include <pico-core/PicoJsonNode.h>
#include <pico-ps/model/Model.h>
#include "DataType.h"

namespace paradigm4 {
namespace pico {
namespace embedding {


#define RETURN_WARNING_STATUS(exp...) do {\
    ps::Status status = (exp);\
    SLOG(WARNING) << status.ToString();\
    return status;\
} while(0)\

#define CHECK_STATUS_RETURN(exp...) do {\
    ps::Status status = (exp);\
    if (!status.ok()) {\
        return status;\
    }\
} while(0)\


// The information required by the client when pull/push.
struct EmbeddingVariableMeta {
    DataType datatype;
    uint64_t embedding_dim = 0;
    uint64_t vocabulary_size = 0;

    size_t line_size()const {
        return datatype.size() * embedding_dim;
    }

    friend bool operator==(const EmbeddingVariableMeta& a, const EmbeddingVariableMeta& b) {
        return a.datatype == b.datatype &&
              a.embedding_dim == b.embedding_dim &&
              a.vocabulary_size == b.vocabulary_size;
    }

    bool use_hash_table()const {
        return vocabulary_size >= (1ull << 63);
    }

    bool from_json_node(const core::PicoJsonNode& json) {
        std::string dtype;
        if (!json.at("datatype").try_as(dtype)) {
            return false;
        }
        datatype = DataType(dtype);
        if (datatype.dtype == DataType::UNKNOWN) {
            return false;
        }
        if (!json.at("embedding_dim").try_as(embedding_dim)) {
            return false;
        }
        if (!json.at("vocabulary_size").try_as(vocabulary_size)) {
            return false;
        }
        return true;
    }

    core::PicoJsonNode to_json_node()const {
        core::PicoJsonNode json;
        json.add("datatype", datatype.to_string());
        json.add("embedding_dim", embedding_dim);
        json.add("vocabulary_size", vocabulary_size);
        return json;
    }

    PICO_SERIALIZATION(datatype, embedding_dim, vocabulary_size);
};

struct ModelVariableMeta {
    EmbeddingVariableMeta meta;
    std::string storage_name;

    friend bool operator==(const ModelVariableMeta& a, const ModelVariableMeta& b) {
        return a.meta == b.meta && a.storage_name == b.storage_name;
    }

    bool from_json_node(const core::PicoJsonNode& json) {
        if (!meta.from_json_node(json)) {
            return false;
        }
        if (!json.at("storage_name").try_as(storage_name)) {
            return false;
        }
        return true;
    }

    core::PicoJsonNode to_json_node()const {
        core::PicoJsonNode json = meta.to_json_node();
        json.add("storage_name", storage_name);
        return json;
    }

    PICO_SERIALIZATION(meta, storage_name);
};

/// TODO: version
struct ModelOfflineMeta {
    std::string model_sign;
    std::vector<ModelVariableMeta> variables;

    static std::string version() {
        return "0.2";
    }

    bool from_json_node(const core::PicoJsonNode& json) {
        variables.clear();
        if (!json.at("model_sign").try_as(model_sign)) {
            return false;
        }
        for (auto& json_item: json.at("variables")) {
            ModelVariableMeta variable;
            if (!variable.from_json_node(json_item)) {
                return false;
            }
            variables.push_back(variable);
        }
        std::string format_version = "unknown";
        json.at("version").try_as(format_version);
        SCHECK(format_version == ModelOfflineMeta::version())
              << "OpenEmbedding model format version is " << format_version
              << ", current versoin is " << ModelOfflineMeta::version() << ".";
        return true;
    }

    core::PicoJsonNode to_json_node()const {
        core::PicoJsonNode json;
        json.add("model_sign", model_sign);
        core::PicoJsonNode vars = core::PicoJsonNode::array();
        for (auto& variable: variables) {
            vars.push_back(variable.to_json_node());
        }
        json.add("variables", vars);
        json.add("version", ModelOfflineMeta::version());
        return json;
    }
    PICO_SERIALIZATION(model_sign, variables);
};

struct ModelMeta {
    std::string model_sign;
    std::string model_uri;
    ps::ModelStatus model_status = ps::ModelStatus::CREATING;
    std::string model_error;
    std::vector<ModelVariableMeta> variables;
    std::map<std::string, int32_t> storages;

    static bool parse(const std::string& str, ps::ModelStatus& model_status) {
        for (int i = 0; i < 5; ++i) {
            if (ps::ModelStatusStr[i] == str) {
                model_status = static_cast<ps::ModelStatus>(i);
                return true;
            }
        }
        return false;
    }

    static std::string to_string(ps::ModelStatus model_status) {
        return ps::ModelStatusStr[static_cast<int>(model_status)];
    }

    bool from_json_node(const core::PicoJsonNode& json) {
        ModelOfflineMeta model_meta;
        if (!model_meta.from_json_node(json)) {
            return false;
        }
        model_sign = model_meta.model_sign;
        variables = model_meta.variables;
        if (!json.at("model_uri").try_as(model_uri)) {
            return false;
        }
        std::string str;
        if (!json.at("model_status").try_as(str)) {
            return false;
        }
        if (!parse(str, model_status)) {
            return false;
        }
        if (!json.at("model_error").try_as(model_error)) {
            return false;
        }
        PicoJsonNode sts = json.at("storages");
        for (auto it = sts.begin(); it != sts.end(); ++it) {
            int32_t storage_id = -1;
            if (!it.value().try_as(storage_id)) {
                return false;
            }
            storages[it.key()] = storage_id;
        } 
        return true;
    }

    core::PicoJsonNode to_json_node()const {
        ModelOfflineMeta model_meta;
        model_meta.model_sign = model_sign;
        model_meta.variables = variables;
        core::PicoJsonNode json = model_meta.to_json_node();
        json.add("model_uri", model_uri);
        json.add("model_status", to_string(model_status));
        json.add("model_error", model_error);
        core::PicoJsonNode sts;
        for (auto& pair: storages) {
            sts.add(pair.first, pair.second);
        }
        json.add("storages", sts);
        return json;
    }

};

}
}
}

#endif
