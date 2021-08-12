#include "Communication.h"

namespace paradigm4 {
namespace pico {
namespace embedding {


Communication::Communication(): _comm_size(1) {} // for native connection

Communication::Communication(core::RpcService* rpc, int32_t comm_size, std::string rpc_name) {
    _rpc = rpc;
    _comm_size = comm_size;
    _rpc_name = rpc_name;
    _rpc_server = _rpc->create_server(_rpc_name);
    _serving_th = std::thread(&Communication::serving, this);
    _rpc_client = _rpc->create_client(_rpc_name, comm_size);
    core::RpcServiceInfo info;
    _rpc_client->get_rpc_service_info(info);
    SCHECK(info.servers.size() == static_cast<size_t>(comm_size)) << "error sync num";
    for (core::ServerInfo server: info.servers) {
        SCHECK(server.server_id < comm_size) << "error server id";
    }
    _comm_rank = _rpc_server->id();
    SCHECK(_comm_rank < comm_size) << "error comm rank";
    _dealer = [this]() { return _rpc_client->create_dealer(); };
}

Communication::~Communication() {
    _dealer.clear();
    _rpc_server->terminate();
    _serving_th.join();
    _rpc_client.reset();
    _rpc_server.reset();
    _rpc->deregister_rpc_service(_rpc_name);
}


comm_rank_t Communication::barrier(std::string name) {
    int32_t num = _comm_size;
    if (num == 1) {
        return _comm_rank;
    }

    core::RpcRequest req;
    req.head().sid = std::hash<std::string>()(name) % _comm_size;
    req << BARRIER << name << num << _comm_rank;

    std::shared_ptr<core::Dealer> dealer = _dealer.acquire();
    core::RpcResponse resp = dealer->sync_rpc_call(std::move(req));
    _dealer.release(std::move(dealer));

    comm_rank_t selected;
    resp >> selected;
    return selected;
}

bool Communication::load_model_sign(const std::string& model_sign) {
    core::RpcRequest req;
    req.head().sid = 0;
    req << LOAD_MODEL_SIGN << model_sign;

    std::shared_ptr<core::Dealer> dealer = _dealer.acquire();
    core::RpcResponse resp = dealer->sync_rpc_call(std::move(req));
    _dealer.release(std::move(dealer));

    bool result;
    resp >> result;
    return result;
}

void Communication::inner_boardcast(std::string name, core::BinaryArchive& ar, comm_rank_t from) {
    int32_t num = _comm_size;
    if (num == 1) {
        return;
    }

    core::RpcRequest req;
    req.head().sid = from;
    bool is_main = from == _comm_rank;
    req << BOARD_CAST << name << num << is_main;
    if (is_main) {
        req << ar;
    }

    std::shared_ptr<core::Dealer> dealer = _dealer.acquire();
    core::RpcResponse resp = dealer->sync_rpc_call(std::move(req));
    _dealer.release(std::move(dealer));
    if (!is_main) {
        resp >> ar;
    }
}

void Communication::serving() {
    core::RpcRequest req;
    std::shared_ptr<core::Dealer> dealer = _rpc_server->create_dealer();
    while (dealer->recv_request(req)) {
        uint32_t req_type;
        req >> req_type;
        if (req_type == BOARD_CAST) {
            std::string name;
            req >> name;
            uint32_t num;
            req >> num;
            auto& reqs = _reqs[name];
            reqs.push_back(std::move(req));
            if (reqs.size() >= num) {
                SCHECK(reqs.size() == num) << "error barrier node num!";
                core::BinaryArchive ar;
                for (core::RpcRequest& req1: reqs) {
                    bool is_main;
                    req1 >> is_main;
                    if (is_main) {
                        req1 >> ar;
                    }
                }
                for (core::RpcRequest& req1: reqs) {
                    core::RpcResponse resp(req1);
                    resp << ar;
                    dealer->send_response(std::move(resp));
                }
                _reqs.erase(name);
            }
        } else if (req_type == LOAD_MODEL_SIGN) {
            std::string model_sign;
            req >> model_sign;
            core::RpcResponse resp(req);
            resp << (_model_sign != model_sign);
            _model_sign = model_sign;
            dealer->send_response(std::move(resp));
        } else if (req_type == BARRIER) {
            std::string name;
            req >> name;
            uint32_t num;
            req >> num;
            auto& reqs = _barriers[name];
            reqs.push_back(std::move(req));
            if (reqs.size() >= num) {
                SCHECK(reqs.size() == num) << "error barrier node num: " << reqs.size() << ' ' << num;
                int32_t fast_comm_rank = -1;
                for (core::RpcRequest& req1: reqs) {
                    if (fast_comm_rank == -1) {
                        req1 >> fast_comm_rank;
                    }
                    core::RpcResponse resp(req1);
                    resp << fast_comm_rank;
                    dealer->send_response(std::move(resp));
                }
                _barriers.erase(name);
            }
        }
    }
}


}
}
}
