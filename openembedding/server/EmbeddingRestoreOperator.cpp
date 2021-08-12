#include "EmbeddingRestoreOperator.h"

#include <pico-ps/operator/RestoreOperator.h>
#include "EmbeddingVariable.h"
#include "EmbeddingShardFile.h"
#include "EmbeddingStorage.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

void EmbeddingRestoreOperator::generate_coordinated_restore_request(
      ps::CoordinatedRestoreRequestItem* req_item, std::vector<ps::PSRequest>& req) {
    auto& item = *static_cast<ps::KVShardCoordinatedRestoreRequestItem*>(req_item);
    req.emplace_back(item.node_id);
    req.back() << item.storage_id << item.iterator_id << item.offset << item.batch_size << item.shard_id;
}

void EmbeddingRestoreOperator::apply_coordinated_restore_request(
      ps::PSRequest& req, ps::Storage* storage, ps::PSResponse& resp) {
    int32_t storage_id;
    int32_t iterator_id;
    size_t offset;
    size_t batch_size;
    int32_t shard_id;
    req >> storage_id >> iterator_id >> offset >> batch_size >> shard_id;
    auto& st = *static_cast<EmbeddingStorage*>(storage);
    core::shared_lock_guard<EmbeddingStorage> l(st);
    if (!st.exist_shard(shard_id)) {
        resp.rpc_response().set_error_code(RpcErrorCodeType::ELOGICERROR);
        resp << ps::Status::InvalidID("Invalid shard id");
        return;
    }

    auto &shard = *(st.get(shard_id));
    core::lock_guard<ps::ShardData> ls(shard);
    EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&shard.data);

    std::vector<uint32_t> variable_ids = ht.variable_ids();
    std::sort(variable_ids.begin(), variable_ids.end());
    size_t vid = 0;
    uint64_t index = offset;
    while (vid < variable_ids.size() && index >= ht[variable_ids[vid]].num_indices()) {
        index -= ht[variable_ids[vid]].num_indices();
        ++vid;
    }

    resp = ps::PSResponse(req);
    bool finished = vid == variable_ids.size();
    resp << storage_id << shard_id;
    if (!finished) {
        uint32_t variable_id = variable_ids[vid];
        EmbeddingVariableMeta meta = ht.meta(variable_id);
        EmbeddingVariableBase& variable = ht[variable_id];
        EmbeddingVariableIndexReader& reader = variable.get_reader(iterator_id);
        std::vector<uint64_t> indices(_block_size / meta.line_size() + 1);
        indices.resize(reader.read(indices.data(), indices.size()));
        if (reader.cursor() == variable.num_indices()) {
            variable.release_reader(reader.reader_id());
            iterator_id = -1;
        } else {
            iterator_id = reader.reader_id();
        }
        offset += indices.size();
        resp << false << iterator_id << offset;

        uint64_t shard_vocabulary = variable.vocabulary_size();
        core::Configure config;
        variable.dump_config(config);
        resp << variable_id << meta << shard_vocabulary << config.dump() << indices;

        BinaryArchive ar;
        ar.prepare_write(indices.size() * meta.line_size());
        variable.read_only_get_weights(indices.data(), indices.size(), ar.end());
        ar.advance_end(indices.size() * meta.line_size());
        ps_serialize(resp.lazy(), _compress_info, std::move(ar));
    } else {
        resp << true << iterator_id << offset;
    }
}

void EmbeddingRestoreOperator::apply_coordinated_restore_response(ps::PSResponse& resp, ps::Storage* storage, ps::CoordinatedRestoreResponseItem* resp_item) {
    int32_t storage_id;
    int32_t shard_id;
    resp >> storage_id >> shard_id;
    resp >> resp_item->finished >> resp_item->iterator_id >> resp_item->next_offset;
    if (!resp_item->finished) {
        uint32_t variable_id;
        EmbeddingVariableMeta meta;
        uint64_t shard_vocabulary;
        std::string config_str;
        std::vector<uint64_t> indices;
        resp >> variable_id >> meta >> shard_vocabulary >> config_str >> indices;

        BinaryArchive ar;
        core::Configure config;
        config.load(config_str);
        ps_deserialize(resp.lazy(), _compress_info, ar);
        auto& st = *static_cast<EmbeddingStorage*>(storage);
        core::shared_lock_guard<EmbeddingStorage> l(st);
        std::string shards;
        for (auto& shard_id: st.shard_list()) {
            shards += std::to_string(shard_id) + " ";
        }
        st.write_shard(shard_id, [&](boost::any& any) {
            EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&any);
            auto& variable = ht.get(variable_id, meta);
            variable.vocabulary_resize(shard_vocabulary);
            variable.load_config(config);
            variable.set_weights(indices.data(), indices.size(), ar.cursor());
        });
    }
}

void EmbeddingRestoreOperator::restore(const core::URIConfig& uri, ps::RuntimeInfo& rt, ps::Storage* storage) {
    auto& st = *static_cast<EmbeddingStorage*>(storage);
    core::shared_lock_guard<EmbeddingStorage> l(st);
    FileReader reader;
    SCHECK(reader.open(uri));
    EmbeddingShardDataMeta shard;
    while (reader.read(shard)) {
        for (int32_t shard_id : rt.local_shards()) {
            st.write_shard(shard_id, [&](boost::any& any) {
                EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&any);
                auto& variable = ht.get(shard.variable_id, shard.meta);
                variable.vocabulary_resize(shard.meta.shard_vocabulary_size(shard_id, rt.global_shard_num()));
                core::Configure config;
                config.load(shard.config);
                variable.load_config(config);
            });
        }
        std::vector<char> value(shard.meta.line_size());
        uint64_t n = shard.num_indices();
        for (size_t i = 0; i < n; ++i) {
            reader.read(value.data(), value.size());
            uint64_t key = shard.get_index(i);
            int32_t shard_id = key % rt.global_shard_num();
            if (rt.local_shards().count(shard_id) > 0) {
                uint64_t index = key / rt.global_shard_num();
                st.write_shard(shard_id, [&](boost::any& any) {
                    EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&any);
                    auto& variable = ht.get(shard.variable_id, shard.meta);
                    variable.set_weights(&index, 1, value.data());
                });
            }
        }
        if (shard.state_line_size) {
            std::vector<char> buffer(shard.state_line_size);
            for (uint64_t i = 0; i < n; ++i) {
                reader.read(buffer.data(), buffer.size());
            }
        }
    }
}


}
}
}
