#include "EnvConfig.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

using namespace core;

PICO_CONFIGURE_DEFINE(ServerConfig,
        message_compress,
        std::string,
        "",
        "The algorithm to compress request/response in pull/push operator,"
        " emptry string \"\" means not using compress",
        true,
        EnumChecker<std::string>({"", "snappy", "lz4", "zlib"}));


PICO_CONFIGURE_DEFINE(ServerConfig,
        server_concurrency,
        int,
        -1,
        "server concurrency",
        true,
        NotEqualChecker<int>(0));



PICO_CONFIGURE_DEFINE(ServerConfig,
        recv_timeout,
        int,
        -1,
        "timeout (ms) of client requests",
        true,
        GreaterEqualChecker<int>(-1));


PICO_CONFIGURE_DEFINE(ServerConfig,
        report_interval,
        int,
        -1,
        "report accumulator interval (s)",
        true,
        GreaterEqualChecker<int>(-1));

PICO_CONFIGURE_DEFINE(ServerConfig,
        update_early_return,
        bool,
        true,
        "client unique pull push keys",
        true,
        DefaultChecker<bool>());

PICO_CONFIGURE_DEFINE(MasterConfig,
        endpoint,
        std::string,
        "",
        "endpoint of tcp or zk master",
        true,
        DefaultChecker<std::string>());

PICO_CONFIGURE_DEFINE(MasterConfig,
        type,
        std::string,
        "tcp",
        "tcp or zk",
        true,
        EnumChecker<std::string>({"tcp", "zk"}));

PICO_CONFIGURE_DEFINE(MasterConfig,
        root_path,
        std::string,
        "/openembedding",
        "master root path",
        true,
        DefaultChecker<std::string>());

PICO_CONFIGURE_DEFINE(MasterConfig,
        recv_timeout,
        size_t,
        300000u,
        "zookeeper recv timeout ms",
        true,
        NotEqualChecker<size_t>(0u));

PICO_CONFIGURE_DEFINE(MasterConfig,
        cache_timeout,
        size_t,
        300u,
        "zookeeper recv timeout ms",
        true,
        NotEqualChecker<size_t>(0u));

/************** rdma ****************/
#ifdef USE_RDMA
PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        ib_devname,
        std::string,
        "",
        "lb_devname",
        true,
        DefaultChecker<std::string>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        gid_index,
        int,
        0,
        "",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        ib_port,
        int,
        0,
        "",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        traffic_class,
        int,
        4,
        "",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        sl,
        int,
        4,
        "",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        mtu,
        int,
        1024,
        "",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        pkey_index,
        int,
        0,
        "",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        min_rnr_timer,
        int,
        12,
        "",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        retry_cnt,
        int,
        7,
        "",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        RdmaConfig,
        timeout,
        int,
        12,
        "",
        true,
        DefaultChecker<int>());
#endif

PICO_CONFIGURE_DEFINE(
        TcpConfig,
        keepalive_time,
        int,
        -1,
        "tcp_keepalive_time, -1:use sys config",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        TcpConfig,
        keepalive_intvl,
        int,
        -1,
        "tcp_keepalive_intvl, -1:use sys config",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        TcpConfig,
        keepalive_probes,
        int,
        -1,
        "tcp_keepalive_probes, -1:use sys config",
        true,
        DefaultChecker<int>());

PICO_CONFIGURE_DEFINE(
        TcpConfig,
        connect_timeout,
        int,
        3600,
        "connect timeout seconds, -1:inf",
        true,
        DefaultChecker<int>());

/************** rpc ****************/
PICO_CONFIGURE_DEFINE(
        RpcConfig,
        bind_ip,
        std::string,
        "",
        "rpc bind ip",
        true,
        DefaultChecker<std::string>());

PICO_CONFIGURE_DEFINE(
        RpcConfig,
        io_thread_num,
        size_t,
        2,
        "rpc io thread num",
        true,
        NotEqualChecker<size_t>(0u));

PICO_CONFIGURE_DEFINE(
        RpcConfig,
        protocol,
        std::string,
        "tcp",
        "network socket protocol",
        true,
#ifdef USE_RDMA
        EnumChecker<std::string>({"tcp", "rdma"})
#else
        EnumChecker<std::string>({"tcp"})
#endif
        );

#ifdef USE_RDMA
PICO_STRUCT_CONFIGURE_DEFINE(
        RpcConfig,
        rdma,
        RdmaConfig,
        "rdma config",
        true,
        DefaultChecker<RdmaConfig>());
#endif 

PICO_STRUCT_CONFIGURE_DEFINE(
        RpcConfig,
        tcp,
        TcpConfig,
        "tcp config",
        true,
        DefaultChecker<TcpConfig>());

PICO_STRUCT_CONFIGURE_DEFINE(EnvConfig,
        master,
        MasterConfig,
        "master",
        true,
        DefaultChecker<MasterConfig>());

PICO_STRUCT_CONFIGURE_DEFINE(EnvConfig,
        rpc,
        RpcConfig,
        "rpc",
        true,
        DefaultChecker<RpcConfig>());

PICO_STRUCT_CONFIGURE_DEFINE(EnvConfig,
        server,
        ServerConfig,
        "parameter server",
        true,
        DefaultChecker<ServerConfig>());


}
}
}
