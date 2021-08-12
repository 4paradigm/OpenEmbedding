#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_RPC_VIEW_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_RPC_VIEW_H

#include <pico-core/LazyArchive.h>
#include <pico-ps/common/message.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class T>
struct RpcView {
    static_assert(std::is_trivially_copyable<T>::value, "");

    RpcView() {}

    // 没有所有权
    RpcView(ps::RpcVector<T>& vector) {
        data = vector.data();
        size = vector.size();
    }

    RpcView(RpcView&& other) {
        *this = std::move(other);
    }

    RpcView& operator=(RpcView&& other) {
        data = other.data;
        size = other.size;
        holder = std::move(other.holder);
        other.data = nullptr;
        other.size = 0;
        return *this;
    }

    // for src_rank == dest_rank
    // receive后有所有权
    void receive() {
        if (!holder.deleter.owner) {
            holder = data_block_t(size * sizeof(T));
            memcpy(holder.data, data, holder.length);
            data = reinterpret_cast<T*>(holder.data);
        }
    }

    void receive(BinaryArchive&& ar) {
        SCHECK(ar.length() % sizeof(T) == 0);
        holder = data_block_t(ar.length());
        memcpy(holder.data, ar.buffer(), ar.length());
        data = reinterpret_cast<T*>(holder.data);
        size = ar.length() / sizeof(T);
        ar = BinaryArchive();
    }

    T* data = nullptr;
    size_t size = 0;
    data_block_t holder;
};


template<class T>
bool pico_serialize(core::ArchiveWriter&, core::SharedArchiveWriter& sar, RpcView<T>& view) {
    sar.put_shared_uncheck(view.data, view.size);
    return true;
}

template<class T>
bool pico_deserialize(core::ArchiveReader&, core::SharedArchiveReader& sar, RpcView<T>& view) {
    // receive后有所有权
    if (sar.is_exhausted()) {
        return false;
    }
    sar.get_shared_uncheck(view.data, view.size, view.holder);
    return true;
}


template<class T>
void serialize(core::LazyArchive& lazy, ps::CompressInfo& compress_info, RpcView<T>&& view) {
    if (compress_info._enabled) {
        BinaryArchive msg_ar, compressed_ar(true);
        msg_ar.set_read_buffer(reinterpret_cast<char*>(view.data), view.size * sizeof(T));
        compress_info._compresser.raw_compress(msg_ar, compressed_ar);
        lazy << std::move(compressed_ar);
    } else {
        lazy << std::move(view);
    }
    view = RpcView<T>();
}

template<class T>
void deserialize(core::LazyArchive& lazy, ps::CompressInfo& compress_info, RpcView<T>& view) {
    if (compress_info._enabled) {
        BinaryArchive msg_ar, compressed_ar;
        lazy >> compressed_ar;
        compress_info._compresser.raw_uncompress(compressed_ar, msg_ar);
        view.receive(std::move(msg_ar));
    } else {
        lazy >> view;
        view.receive();
    }
}



}


}
}

#endif