import os
import ctypes
libcexb_pack = ctypes.cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/libcexb_pack.so')
import openembedding.libexb as libexb

__version__ = libexb.version()

'''
Master

config: yaml格式的配置

master_endpoint: server 或 worker 初始化时需要 master_endpoint
    '': 表示在进程内启动一个 master

    '{ip}:{port}': 外部 master 的 endpoint

bind_ip: 绑定的 ip 地址，对 worker 和 server 均有效
    '': 自动绑定到一个网卡的 ip 地址

    '{ip}': 指定网卡的 ip 地址

num_workers: worker 的数量，不同 worker 的配置应一致

wait_num_servers: server 的数量，不同 worker 的配置应一致
    -1: 每个 worker 进程内部启动一个 server
    n: worker 需要先等待 n 个 server 启动完成
'''
class Flags:
    def __init__(self, config='', master_endpoint='', bind_ip='', num_workers=1, wait_num_servers=-1):
        self.config = config
        self.master_endpoint = master_endpoint
        self.bind_ip = bind_ip
        self.num_workers = num_workers
        self.wait_num_servers = wait_num_servers
flags = Flags()

'''
进程内启动一个Master
'''
class Master:
    def __init__(self):
        self.__master = libexb.Master(flags.bind_ip)

    def __del__(self):
        self.__master.finalize()

    @property
    def endpoint(self):
        '''
        格式 '{ip}:{port}'
        '''
        return self.__master.endpoint

'''
进程内启动一个参数服务器
'''
class Server:
    def __init__(self):
        self.__server = libexb.Server(flags.config, flags.master_endpoint, flags.bind_ip)

    def exit(self):
        '''
        向这个参数服务器发送退出指令
        '''
        return self.__server.exit()
    
    def join(self):
        '''
        等待参数服务器退出，可能通过Server.exit退出或其他方式退出
        '''
        return self.__server.join()