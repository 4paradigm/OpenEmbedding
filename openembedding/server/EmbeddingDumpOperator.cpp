#include "EmbeddingDumpOperator.h"

#include <pico-ps/operator/DumpOperator.h>
#include "EmbeddingVariable.h"
#include "EmbeddingShardFile.h"
#include "EmbeddingStorage.h"
#include "Factory.h"

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

    bool persist_checkpoint = false;
    uri.config().get_val("persist_checkpoint", persist_checkpoint);
    if (!include_optimizer && persist_checkpoint) {
        SLOG(WARNING) << "persist checkpoint not support without optimizer.";
        persist_checkpoint = false;
    }
    
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
            shard_meta.variable_id = variable_id;
            shard_meta.meta = ht.meta(variable_id);

            core::Configure config;
            bool variable_persist = false;
            if (persist_checkpoint) {
                variable_persist = variable.dump_persist(config);
                if (!variable_persist) {
                    variable.dump_config(config);
                    SLOG(WARNING) << "variable is not persistent, "
                                  << "fall back to normal dump " << config.dump();
                }
                if (!include_optimizer) {
                    config.node().remove("optimizer");
                }
            } else {
                variable.dump_config(config);
            }
            shard_meta.config = config.dump();
            shard_meta.shard_id = shard_id;
            shard_meta.shard_num = rt.global_shard_num();
            shard_meta.state_line_size = include_optimizer ? variable.state_line_size() : 0;
            shard_meta.num_items = variable_persist ? 0 : variable.num_indices();
            writer.write(shard_meta);

            int reader_id = variable.create_reader();
            size_t n = 0;
            core::vector<uint64_t> indices(variable.server_block_num_items());
            while ( (n = variable.read_indices(reader_id, indices.data(), indices.size())) ) {
                writer.write(n);
                indices.resize(n);
                core::vector<char> weights(indices.size() * shard_meta.meta.line_size());
                core::vector<char> states(indices.size() * shard_meta.state_line_size);
                variable.get_weights(indices.data(), n, weights.data(), states.data());
                writer.write(indices.data(), indices.size());
                writer.write(weights.data(), weights.size());
                writer.write(states.data(), states.size());
            }
            variable.delete_reader(reader_id);
        }
    }
    resp << ps::Status();
    resp_ret = std::move(resp);
}

}
}
}
