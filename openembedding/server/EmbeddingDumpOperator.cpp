#include "EmbeddingDumpOperator.h"

#include <pico-ps/operator/DumpOperator.h>
#include "EmbeddingVariable.h"
#include "EmbeddingShardFile.h"
#include "EmbeddingStorage.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

void EmbeddingDumpOperator::apply_request(ps::RuntimeInfo& rt,
        ps::PSRequest& req,
        ps::Storage* storage,
        ps::PSResponse& resp_ret) {
    ps::DumpArgs dump_args;
    req >> dump_args;
    int32_t file_id;
    req >> file_id;
    std::vector<int32_t> shard_ids;
    req >> shard_ids;
    SCHECK(req.archive().is_exhausted());
    ps::PSResponse resp(req);
    //core::FileSystem::mkdir_p(dump_args.uri());

    core::URIConfig uri(dump_args.uri());
    std::string file = format_string("/model_%d_%d", rt.node_id(), file_id);
    FileWriter writer;
    if (!writer.open(uri + file)) {
        if (uri.storage_type() != core::FileSystemType::HDFS) {
            core::FileSystem::mkdir_p(uri);
        }
        SCHECK(writer.open(uri + file));
    }
    bool include_optimizer = true;
    uri.config().get_val("include_optimizer", include_optimizer);
    
    auto& st = *(static_cast<EmbeddingStorage*>(storage));
    core::shared_lock_guard<EmbeddingStorage> l(st);
    for (int32_t shard_id: shard_ids) {
        SCHECK(rt.local_shards().count(shard_id) != 0) 
                << "Bad Request: invalid shard_id = " << shard_id;
        auto& shard = *(st.get(shard_id));
        core::lock_guard<ps::ShardData> sl(shard);
        EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&shard.data);
        for (uint32_t variable_id: ht.variable_ids()) {
            EmbeddingVariableBase& variable = ht[variable_id];
            
            EmbeddingShardDataMeta shard_meta;
            core::Configure config;
            variable.dump_config(config);
            shard_meta.variable_id = variable_id;
            shard_meta.meta = ht.meta(variable_id);
            if (!include_optimizer) {
                config.node()["optimizer"] = "";
            }
            shard_meta.config = config.dump();
            shard_meta.shard_id = shard_id;
            shard_meta.shard_num = rt.global_shard_num();
            shard_meta.state_line_size = include_optimizer ? variable.state_line_size() : 0;
            shard_meta.indices.resize(variable.num_indices());
            EmbeddingVariableIndexReader& reader = variable.get_reader(-1);
            reader.read(shard_meta.indices.data(), variable.num_indices());
            variable.release_reader(reader.reader_id());
            std::vector<char> buffer(shard_meta.meta.line_size());
            size_t n = shard_meta.num_indices();
            writer.write(shard_meta);

            int32_t global_shard_num = rt.global_shard_num();
            for (size_t i = 0; i < n; ++i) {
                uint64_t index = shard_meta.get_index(i) / global_shard_num;
                variable.read_only_get_weights(&index, 1, buffer.data());
                writer.write(buffer.data(), buffer.size());
            }

            if (include_optimizer) {
                buffer.resize(variable.state_line_size());
                for (size_t i = 0; i < n; ++i) {
                    uint64_t index = shard_meta.get_index(i) / global_shard_num;
                    variable.get_states(&index, 1, buffer.data());
                    writer.write(buffer.data(), buffer.size());
                }
            }
        }
    }
    resp << ps::Status();
    resp_ret = std::move(resp);
}

}
}
}
