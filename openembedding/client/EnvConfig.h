#ifndef PARADIGM4_HYPEREMBEDDING_EXB_ENV_CONFIG_H
#define PARADIGM4_HYPEREMBEDDING_EXB_ENV_CONFIG_H

#include <pico-core/Configure.h>
#include <pico-core/ConfigureHelper.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

using core::ConfigNode;
using core::ConfigUnit;

#ifdef USE_RDMA
DECLARE_CONFIG(RdmaConfig, ConfigNode) {
    PICO_CONFIGURE_DECLARE(std::string, ib_devname);
    PICO_CONFIGURE_DECLARE(int, gid_index);
    PICO_CONFIGURE_DECLARE(int, ib_port);
    PICO_CONFIGURE_DECLARE(int, traffic_class);
    PICO_CONFIGURE_DECLARE(int, sl);
    PICO_CONFIGURE_DECLARE(int, mtu);
    PICO_CONFIGURE_DECLARE(int, pkey_index);
    PICO_CONFIGURE_DECLARE(int, min_rnr_timer);
    PICO_CONFIGURE_DECLARE(int, retry_cnt);
    PICO_CONFIGURE_DECLARE(int, timeout);
};
#endif

DECLARE_CONFIG(TcpConfig, ConfigNode) {
    PICO_CONFIGURE_DECLARE(int, keepalive_time);
    PICO_CONFIGURE_DECLARE(int, keepalive_intvl);
    PICO_CONFIGURE_DECLARE(int, keepalive_probes);
    PICO_CONFIGURE_DECLARE(int, connect_timeout);
};

DECLARE_CONFIG(RpcConfig, ConfigNode) {
    PICO_CONFIGURE_DECLARE(std::string, bind_ip);
    PICO_CONFIGURE_DECLARE(size_t, io_thread_num);
    PICO_CONFIGURE_DECLARE(std::string, protocol);
#ifdef USE_RDMA
    PICO_CONFIGURE_DECLARE(RdmaConfig, rdma);
#endif
    PICO_CONFIGURE_DECLARE(TcpConfig, tcp);
};

DECLARE_CONFIG(MasterConfig, ConfigNode) {
    PICO_CONFIGURE_DECLARE(std::string, endpoint);
    PICO_CONFIGURE_DECLARE(std::string, type);
    PICO_CONFIGURE_DECLARE(std::string, root_path);
    PICO_CONFIGURE_DECLARE(size_t, recv_timeout);
    PICO_CONFIGURE_DECLARE(size_t, cache_timeout);
};

DECLARE_CONFIG(ServerConfig, ConfigNode) {
    PICO_CONFIGURE_DECLARE(std::string, pmem_pool_root_path);
    PICO_CONFIGURE_DECLARE(size_t, cache_size);
    PICO_CONFIGURE_DECLARE(std::string, message_compress);
    PICO_CONFIGURE_DECLARE(int, server_concurrency);
    PICO_CONFIGURE_DECLARE(int, recv_timeout);
    PICO_CONFIGURE_DECLARE(int, report_interval);
    PICO_CONFIGURE_DECLARE(bool, update_early_return);
};

class EnvConfig: public ConfigNode {
    // client server shared
    // default shard_num = server_concurrency * server_num
//    PICO_CONFIGURE_DECLARE(size_t, max_request_merge_num); // pull push
public:
    PICO_CONFIGURE_DECLARE(RpcConfig, rpc);
    PICO_CONFIGURE_DECLARE(MasterConfig, master);
    PICO_CONFIGURE_DECLARE(ServerConfig, server);
public:

    void load_yaml(const core::Configure& configure, const std::string& master_endpoint = "", const std::string& rpc_bind_ip = "") {
        SCHECK(load_config(configure));
        if (!master_endpoint.empty()) {
            master.endpoint = master_endpoint;
        }
        if (!rpc_bind_ip.empty()) {
            rpc.bind_ip = rpc_bind_ip;
        }
    }
};


}
}
}

#endif
