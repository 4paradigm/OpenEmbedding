#include "EmbeddingLoadOperator.h"

#include <pico-ps/operator/LoadOperator.h>
#include "EmbeddingInitOperator.h"
#include "EmbeddingShardFile.h"
#include "EmbeddingStorage.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

struct DataItems: EmbeddingInitItems {
    core::vector<uint64_t> hold_indices;
    core::vector<char> hold_weights;
    core::vector<char> hold_states;
};

struct FileStream {
    FileReader reader;
    EmbeddingShardDataMeta shard;
    int state = 0; // 0-weight, 1-state_view
    uint64_t offset = 0; // key id
    bool open(const core::URIConfig& uri) {
        return reader.open(uri);
    }
};

struct DataStream {
    size_t i = 0;
    std::vector<FileStream> files;
    DataStream(const core::URIConfig& uri) {
        if (uri.storage_type() == FileSystemType::HDFS) {
            FileStream file;
            SCHECK(file.open(uri));
            files.push_back(file);
        } else {
            auto paths = core::FileSystem::get_file_list(uri.uri());
            for (size_t i = 0; i < paths.size(); ++i) {
                FileStream file;
                SCHECK(file.open(paths[i]));
                files.push_back(file);
            }
        }
    }
};

void EmbeddingLoadOperator::apply_load_response(ps::PSResponse& resp) {
    ps::Status st;
    resp >> st;
    SCHECK(resp.archive().is_exhausted()) << resp.archive().readable_length();
    SCHECK(st.ok()) << st.ToString();
}

void EmbeddingLoadOperator::create_stream(const URIConfig& uri, std::shared_ptr<void>& stream) {
    stream = std::make_shared<DataStream>(uri);
}

size_t EmbeddingLoadOperator::generate_push_items(std::shared_ptr<void>& stream_in,
        core::vector<std::unique_ptr<ps::PushItems>>& push_items) {
    if (stream_in == nullptr) {
        return 0;
    }
    DataStream& stream = *static_cast<DataStream*>(stream_in.get());
    while (stream.i < stream.files.size()) {
        FileStream& file = stream.files[stream.i];
        if (file.state == 0) {
            if (file.reader.read(file.shard)) {
                if (file.shard.num_items > 0) {
                    file.state = 1;
                }
                std::unique_ptr<DataItems> items = std::make_unique<DataItems>();
                items->variable_id = file.shard.variable_id;
                items->meta = file.shard.meta;
                items->variable_config = file.shard.config;
                push_items.push_back(std::move(items));
                return 1;
            } else {
                ++stream.i;
            }
        } else {
            size_t n = 0;
            file.reader.read(n);
            std::unique_ptr<DataItems> items = std::make_unique<DataItems>();
            items->hold_indices.resize(n);
            items->hold_weights.resize(n * file.shard.meta.line_size());
            items->hold_states.resize(n * file.shard.state_line_size);
            file.reader.read(items->hold_indices.data(), items->hold_indices.size());
            file.reader.read(items->hold_weights.data(), items->hold_weights.size());
            file.reader.read(items->hold_states.data(), items->hold_states.size());
            for (uint64_t& key: items->hold_indices) {
                key = file.shard.get_index(key);
            }
            items->variable_id = file.shard.variable_id;
            items->meta = file.shard.meta;
            items->n = items->hold_indices.size();
            items->indices = items->hold_indices.data();
            items->weights = items->hold_weights.data();
            items->states = items->hold_states.data();
            items->state_line_size = file.shard.state_line_size;
            items->variable_config = file.shard.config;
            file.offset += n;
            push_items.push_back(std::move(items));
            if (file.offset >= file.shard.num_items) {
                file.offset = 0;
                file.state = 0;
            }
            return 1;
        }
    }
    return 0;
}

void EmbeddingLoadOperator::restore(const URIConfig& uri, ps::RuntimeInfo& rt, ps::Storage* storage) {
    // for persist only
    auto& st = *static_cast<EmbeddingStorage*>(storage);
    core::shared_lock_guard<EmbeddingStorage> l(st);
    DataStream stream(uri);
    while (stream.i < stream.files.size()) {
        FileStream& file = stream.files[stream.i];
        while (file.reader.read(file.shard)) {
            SCHECK(file.shard.shard_num == rt.global_shard_num());
            SCHECK(rt.local_shards().count(file.shard.shard_id));
            SCHECK(file.shard.num_items == 0);

            auto& shard = *(st.get(file.shard.shard_id));
            core::lock_guard<ps::ShardData> sl(shard);
            auto& ht = *boost::any_cast<EmbeddingShard>(&shard.data);
            EmbeddingVariableBase& variable = ht.get(file.shard.variable_id, file.shard.meta);
            EmbeddingVariableContext variable_context;
            variable_context.variable_id = file.shard.variable_id;
            variable.set_variable_context(variable_context);

            core::Configure variable_config;
            variable_config.load(file.shard.config);
            variable.load_config(variable_config);
        }
        ++stream.i;
    }
}

}
}
}
