import os
import ctypes
libcexb_pack = ctypes.cdll.LoadLibrary(os.path.dirname(os.path.realpath(__file__)) + '/libcexb_pack.so')
import openembedding.libexb as libexb
from openembedding.libexb import checkpoint_batch_id

__version__ = libexb.version()



'''
Master

config: Configure in yaml format

master_endpoint: Required when the Server or worker is initialized
    '': Start a Master in this process

    '{ip}:{port}': The endpoint of master

bind_ip: Used by worker, Server and Master
    '': automatically bind to the ip address of a network card

    '{ip}': specify the ip address, bind on random port

    '{ip}:{port}': bind on the specified ip and port, only supported by Master

num_workers: should be consistent in different workers

wait_num_servers: should be consistent in different workers
    -1: start a Server in each worker process.
    n: need to wait the number of Servers start.
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
Run a master in this process.
'''
class Master:
    def __init__(self):
        self.__master = libexb.Master(flags.bind_ip)

    def __del__(self):
        self.__master.finalize()

    @property
    def endpoint(self):
        '''
        The format is '{ip}:{port}'.
        '''
        return self.__master.endpoint

'''
Run a parameter server in this process.
'''
class Server:
    def __init__(self):
        self.__server = libexb.Server(flags.config, flags.master_endpoint, flags.bind_ip)

    def exit(self):
        '''
        Send exit request to this server.
        '''
        return self.__server.exit()
    
    def join(self):
        '''
        Waiting for the server to exit.
        '''
        return self.__server.join()