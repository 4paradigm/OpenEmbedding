#include "EmbeddingLoadOperator.h"

#include <pico-ps/operator/LoadOperator.h>
#include "EmbeddingInitOperator.h"
#include "EmbeddingShardFile.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

struct DataItems: EmbeddingInitItems {
    std::vector<uint64_t> hold_indices;
    std::vector<char> hold_weights;
};

struct FileStream {
    FileReader reader;
    EmbeddingShardDataMeta shard;
    int state = 2; // 0-weight, 1-state_view
    uint64_t index = 0; // key id
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
        if (file.state == 2) {
            file.shard.indices.clear(); // !!!!
            file.state = file.reader.read(file.shard) ? 0 : -1;
        } else if (file.state >= 0) {
            size_t block_size = 0;
            size_t line_size = file.state ? file.shard.state_line_size : file.shard.meta.line_size();
            size_t n = line_size ? file.shard.num_indices() : 0;
            std::unique_ptr<DataItems> items = std::make_unique<DataItems>();
            while (file.index < n && block_size <= _block_size) {
                items->hold_indices.push_back(file.shard.get_index(file.index));      
                block_size += line_size;
                ++file.index;
            }
            items->hold_weights.resize(block_size);
            file.reader.read(items->hold_weights.data(), items->hold_weights.size());
            items->variable_id = file.shard.variable_id;
            items->meta = file.shard.meta;
            items->n = items->hold_indices.size();
            items->indices = items->hold_indices.data();
            items->weights = items->hold_weights.data();
            items->state_line_size = file.state ? file.shard.state_line_size : 0;
            items->variable_config = file.shard.config;
            if (items->n) {
                push_items.push_back(std::move(items));
                return 1;
            }
            ++file.state;
            file.index = 0;
        } else {
            ++stream.i;
            file.state = 2;
        }
    }
    return 0;
}

}
}
}
