import os 
import sys
import time
import psutil
from threading import Thread
import openembedding as embed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--bind_ip', default='')
parser.add_argument('--server_concurrency', default=28, type=int)

# For paper experiment
parser.add_argument('--pmem', default='')
parser.add_argument('--cache_size', default=1000, type=int)

args = parser.parse_args()
if args.pmem:
    embed.flags.config = ('{"server":{"server_concurrency":%d'
            ',"pmem_pool_root_path":"%s", "cache_size":%d } }') % (
            args.server_concurrency, args.pmem, args.cache_size)
else:
    embed.flags.config = '{"server":{"server_concurrency":%d } }' % (
            args.server_concurrency)


def print_rss():
    print(psutil.Process(os.getpid()).memory_info().rss  / 1024 / 1024 / 1024, 'GB', flush=True)


def start():
    if len(sys.argv) > 1:
        embed.flags.bind_ip = args.bind_ip
    _master = embed.Master()
    print(_master.endpoint)
    embed.flags.bind_ip = embed.flags.bind_ip[:embed.flags.bind_ip.find(':')]
    embed.flags.master_endpoint = _master.endpoint
    _server = embed.Server()
    _server.join()


i = 0
print_rss()
th = Thread(target=start, args=[])
th.start()
while th.is_alive():
    i += 1
    th.join(0.1)
    if i % 100 == 0:
        print_rss()
print_rss()