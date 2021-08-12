#include <gflags/gflags.h>
#include <brpc/restful.h>
#include <pico-core/pico_log.h>

#include "EmbeddingShardFile.h"
#include "Connection.h"
#include "ModelController.h"
#include "controller.pb.h"

// ps
DEFINE_string(config, "", "");
DEFINE_string(config_file, "", "");
DEFINE_string(master_endpoint, "", "");
DEFINE_string(rpc_bind_ip, "", "");


// restful
DEFINE_int32(port, 8010, "TCP Port of this server");
DEFINE_int32(idle_timeout_s, -1, "Connection will be closed if there is no "
             "read/write operations during the last `idle_timeout_s'");
DEFINE_int32(logoff_ms, 2000, "Maximum duration of server's LOGOFF state "
             "(waiting for client to close connection before server stops)");

DEFINE_string(certificate, "", "Certificate file path to enable SSL");
DEFINE_string(private_key, "", "Private key file path to enable SSL");
DEFINE_string(ciphers, "", "Cipher suite used for SSL connections");


using namespace paradigm4::pico;
using namespace paradigm4::pico::embedding;

EnvConfig env;
std::unique_ptr<RpcConnection> connection; 
std::unique_ptr<ModelController> model_controller;

namespace exb {

struct ServiceException {
    int code;
    std::string status;
    ServiceException(int code, const std::string& status): code(code), status(status) {}
};

void check_status_throw(ps::Status status) {
    if (!status.ok()) {
        if (status.IsInvalidID()) {
            throw ServiceException(brpc::HTTP_STATUS_NOT_FOUND, status.ToString());
        } else {
            throw ServiceException(brpc::HTTP_STATUS_FORBIDDEN, status.ToString());
        }
    }
}

class ModelService : public models {
    template<class T>
    bool json_get(core::PicoJsonNode& json, const std::string& key, T& value) {
        if (!json.has(key)) {
            throw ServiceException(brpc::HTTP_STATUS_BAD_REQUEST,
                  "json not has field \"" + key + "\"");
        }
        return json.at(key).try_as(value);
    }

    template<class T>
    T json_get(core::PicoJsonNode& json, const std::string& key) {
        T value;
        if (!json_get(json, key, value)) {
            throw ServiceException(brpc::HTTP_STATUS_BAD_REQUEST,
                  "can not parse json field \"" + key + "\" to " + core::readable_typename<T>());
        }
        return value;
    }

    PicoJsonNode parse_json(const std::string& content) {
        core::PicoJsonNode json;
        if (!json.load(content)) {
            throw ServiceException(brpc::HTTP_STATUS_BAD_REQUEST, 
                  "invalid json: " + content);
        }
        return json;
    }

    ModelOfflineMeta open_model_meta(const core::URIConfig& model_uri) {
        ModelOfflineMeta model_meta;
        FileReader meta_file;
        if (!meta_file.open(model_uri + "/model_meta")) {
            throw ServiceException(brpc::HTTP_STATUS_BAD_REQUEST,
                  "model meta file is not openable");
        }
        if (!meta_file.read(model_meta)) {
            throw ServiceException(brpc::HTTP_STATUS_BAD_REQUEST,
                  "invalid model meta file");
        }
        return model_meta;
    }
 
public:
    ModelService(ModelController* model_controller): _model_controller(model_controller) {}

    void default_method(google::protobuf::RpcController* cntl_base,
          const HttpRequest*, HttpResponse*,
          google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        try {
            core::PicoJsonNode result;
            if (cntl->http_request().method() == brpc::HTTP_METHOD_POST) {
                core::PicoJsonNode json = parse_json(cntl->request_attachment().to_string());
                core::URIConfig model_uri(json_get<std::string>(json, "model_uri"));
                int32_t replica_num = 3;
                int32_t shard_num = -1;
                json_get(json, "replica_num", replica_num);
                json_get(json, "shard_num", shard_num);
                std::string model_sign;
                check_status_throw(_model_controller->create_model(
                    model_uri, model_sign, result, replica_num, shard_num));
                std::stringstream ss;
                ss << cntl->http_request().uri() << "/" << model_sign;
                cntl->http_response().set_status_code(brpc::HTTP_STATUS_CREATED);
                cntl->http_response().AppendHeader("Location", ss.str());

            } else if (cntl->http_request().method() == brpc::HTTP_METHOD_GET) {
                std::string model_sign = cntl->http_request().unresolved_path();
                if (model_sign.empty()) {
                    check_status_throw(_model_controller->show_models(result));
                } else {
                    check_status_throw(_model_controller->show_model(model_sign, result));
                }
                cntl->http_response().set_status_code(brpc::HTTP_STATUS_OK);

            } else if (cntl->http_request().method() == brpc::HTTP_METHOD_DELETE) {
                std::string model_sign = cntl->http_request().unresolved_path();
                check_status_throw(_model_controller->delete_model(model_sign));
                cntl->http_response().set_status_code(brpc::HTTP_STATUS_ACCEPTED);
            } else {
                throw ServiceException(brpc::HTTP_STATUS_NOT_FOUND, "invalid method");
            }
            cntl->http_response().set_content_type("text/json");
            cntl->response_attachment().append(result.dump(4));
        } catch(ServiceException& exception) {
            core::PicoJsonNode result;
            result.add("error", exception.status);
            cntl->http_response().set_status_code(exception.code);
            cntl->http_response().set_content_type("text/json");
            cntl->response_attachment().append(result.dump(4));
        }
    }
private:
    ModelController* _model_controller = nullptr;
};


class NodeService : public nodes {
public:
    NodeService(ModelController* model_controller): _model_controller(model_controller) {}

    int32_t parse_node_id(const std::string& str) {
        int32_t node_id;
        if (!pico_lexical_cast(str, node_id)) {
            throw ServiceException(brpc::HTTP_STATUS_NOT_FOUND,
                  "invalid node id: " + str);
        }
        return node_id;
    }

    void default_method(google::protobuf::RpcController* cntl_base,
          const HttpRequest*, HttpResponse*,
          google::protobuf::Closure* done) {
        brpc::ClosureGuard done_guard(done);
        brpc::Controller* cntl = static_cast<brpc::Controller*>(cntl_base);
        try {
            core::PicoJsonNode result;
            if (cntl->http_request().method() == brpc::HTTP_METHOD_GET) {
                std::string node = cntl->http_request().unresolved_path();
                if (node.empty()) {
                    check_status_throw(_model_controller->show_nodes(result));
                } else {
                    int32_t node_id = parse_node_id(node);
                    check_status_throw(_model_controller->show_node(node_id, result));
                }
                cntl->http_response().set_status_code(brpc::HTTP_STATUS_OK);
            } else if (cntl->http_request().method() == brpc::HTTP_METHOD_DELETE) {
                int32_t node_id = parse_node_id(cntl->http_request().unresolved_path());
                check_status_throw(_model_controller->shutdown_node(node_id));
                cntl->http_response().set_status_code(brpc::HTTP_STATUS_ACCEPTED);
            } else {
                throw ServiceException(brpc::HTTP_STATUS_NOT_FOUND, "invalid method");
            }
            cntl->http_response().set_content_type("text/json");
            cntl->response_attachment().append(result.dump(4));

        } catch(ServiceException& exception) {
            core::PicoJsonNode result;
            result.add("error", exception.status);
            cntl->http_response().set_status_code(exception.code);
            cntl->http_response().set_content_type("text/json");
            cntl->response_attachment().append(result.dump(4));
        }
    }

private:
    ModelController* _model_controller = nullptr;
};

};


int main(int argc, char* argv[]) {
    google::InstallFailureSignalHandler();
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;
    google::AllowCommandLineReparsing();
    google::ParseCommandLineFlags(&argc, &argv, false);

    core::Configure configure;
    if (FLAGS_config.empty()) {
        configure.load(FLAGS_config);
    } else {
        configure.load_file(FLAGS_config_file);
    }
    env.load_yaml(configure, FLAGS_master_endpoint, FLAGS_rpc_bind_ip);
   
    connection = std::make_unique<RpcConnection>(env);
    model_controller = std::make_unique<ModelController>(connection.get());

    brpc::Server server;
    exb::ModelService model_service(model_controller.get());
    exb::NodeService node_service(model_controller.get());

    if (server.AddService(&model_service,
                          brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        SLOG(ERROR) << "Fail to add model_service";
        return -1;
    }

    if (server.AddService(&node_service,
                          brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
        SLOG(ERROR) << "Fail to add node_service";
        return -1;
    }

    // Start the server.
    brpc::ServerOptions options;
    options.idle_timeout_sec = FLAGS_idle_timeout_s;
    // options.mutable_ssl_options()->default_cert.certificate = FLAGS_certificate;
    // options.mutable_ssl_options()->default_cert.private_key = FLAGS_private_key;
    // options.mutable_ssl_options()->ciphers = FLAGS_ciphers;
    if (server.Start(FLAGS_port, &options) != 0) {
        SLOG(ERROR) << "Fail to start HttpServer";
        return -1;
    }

    // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
    server.RunUntilAskedToQuit();

    model_controller.reset();
    connection.reset();
    return 0;
}
