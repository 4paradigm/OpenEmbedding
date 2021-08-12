import os 
import sys
import time
import psutil
from threading import Thread
import openembedding as embed


def print_rss():
    print(psutil.Process(os.getpid()).memory_info().rss  / 1024 / 1024 / 1024, 'GB', flush=True)


def start():
    if len(sys.argv) > 1:
        embed.flags.bind_ip = sys.argv[1]
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