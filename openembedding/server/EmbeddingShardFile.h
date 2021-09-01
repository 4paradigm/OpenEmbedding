
#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_SHRAD_FILE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_SHRAD_FILE_H

#include <pico-core/FileSystem.h>
#include <pico-core/ShellUtility.h>
#include "Meta.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

struct EmbeddingShardDataMeta {
    uint32_t variable_id = 0;
    EmbeddingVariableMeta meta;
    std::string config;
    int32_t shard_id = 0;
    int32_t shard_num = 0;
    uint64_t state_line_size = 0;
    uint64_t num_items = 0;
    PICO_SERIALIZATION(variable_id, meta, config, shard_id, shard_num, state_line_size, num_items);

    uint64_t get_index(uint64_t index)const {
        return index * shard_num + shard_id;
    }
};

class FileReader {
public:
    bool open(const core::URIConfig& uri) {
        std::string hadoop_bin;
        uri.config().get_val(core::URI_HADOOP_BIN, hadoop_bin);
        _file = core::ShellUtility::open_read(uri.name(), "", hadoop_bin);
        _archive.reset(_file);
        return _file;
    }

    template<class T>
    bool read(T& value) {
        return core::pico_deserialize(_archive, value);
    }

    template<class T>
    typename std::enable_if<std::is_pod<T>::value, bool>::type
    read(T* buffer, size_t n) {
        return _archive.read_raw_uncheck(buffer, n * sizeof(T));
    }

private:
    core::shared_ptr<FILE> _file;
    core::BinaryFileArchive _archive;
};

class FileWriter {
public:
    bool open(const core::URIConfig& uri) {
        std::string hadoop_bin;
        uri.config().get_val(core::URI_HADOOP_BIN, hadoop_bin);
        _file = core::ShellUtility::open_write(uri.name(), "", hadoop_bin);
        _archive.reset(_file);
        return _file;
    }

    template<class T>
    void write(const T& value) {
        SCHECK(core::pico_serialize(_archive, value));
    }

    template<class T>
    typename std::enable_if<std::is_pod<T>::value>::type
    write(const T* buffer, size_t n) {
        SCHECK(_archive.write_raw_uncheck(buffer, n * sizeof(T)));
    }

private:
    core::shared_ptr<FILE> _file;
    core::BinaryFileArchive _archive;
};


}
}
}

#endif