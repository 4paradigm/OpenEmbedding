#include <pico-core/pico_log.h>
#include <pico-ps/service/Server.h>
#include "Connection.h"
#include "c_api.h"
#include <pico-core/observability/metrics/Metrics.h>

DEFINE_bool(enable_metrics, false, "enable/disable metrics");
DEFINE_string(service_name, "service_name", "service name of this binary");
DEFINE_string(instance_name, "instance_name", "instance name of this binary");
DEFINE_string(metrics_ip, "0.0.0.0", "Binding IP of the metrics exposer");
DEFINE_int32(metrics_port, 8001, "TCP port of the metrics exposer");
DEFINE_string(metrics_url, "/metrics", "URL of the metrics exposer");

DEFINE_bool(restore, true, "is replace one dead node"); // try replace one dead node

DEFINE_string(config, "", "");
DEFINE_string(config_file, "", "");
DEFINE_string(rpc_bind_ip, "", "");
DEFINE_string(master_endpoint, "", "");


using namespace paradigm4::pico;
using namespace paradigm4::pico::ps;

int main(int argc, char* argv[]) {
    // exb_serving(); // Import registered optimizer.
    google::InstallFailureSignalHandler();
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    google::AllowCommandLineReparsing();
    google::ParseCommandLineFlags(&argc, &argv, false);

    paradigm4::pico::core::Memory::singleton().initialize();

    paradigm4::pico::metrics_initialize(FLAGS_metrics_ip, FLAGS_metrics_port, FLAGS_metrics_url,
            FLAGS_service_name, FLAGS_instance_name, FLAGS_enable_metrics);

    paradigm4::pico::embedding::EnvConfig env;
    paradigm4::pico::core::Configure configure;    
    if (FLAGS_config.empty()) {
        configure.load(FLAGS_config);
    } else {
        configure.load_file(FLAGS_config_file);
    }
    env.load_yaml(configure, FLAGS_master_endpoint, FLAGS_rpc_bind_ip);
    paradigm4::pico::embedding::RpcConnection conn(env);
    paradigm4::pico::core::LogReporter::set_id("SERVER", conn.rpc()->global_rank());
    std::unique_ptr<paradigm4::pico::ps::Server> server;
    server = conn.create_server();
    server->initialize();

    if (FLAGS_restore) {
        server->restore_storages(false);
    }

    server->finalize();
    paradigm4::pico::core::Memory::singleton().finalize();
    paradigm4::pico::metrics_finalize();
    return 0;
}
