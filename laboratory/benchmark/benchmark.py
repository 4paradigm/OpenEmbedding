# cuda = 10.1
# torch = 1.7
# tensorflow = 2.2

import os
import sys
import time

def run_remote_server(user, ip, port):
    os.system('echo "bash run_server.sh {}:{}\n sleep 1\n exit\n" | ssh {}@{}'.format(ip, port, user, ip))

def run(py, data, model, embedding_dim, options, np=1, bind_ip=None, master_endpoint=None):
    if data.endswith('csv'):
        extend = 'csv'
    else:
        extend = 'tf'
    name = 'result/{}_{}_{}_{}'.format(py, extend, model, embedding_dim)
    command = 'horovodrun -np {} python3.7 {}.py'.format(np, py)
    command += ' --data {} --model {} --embedding_dim {}'.format(data, model, embedding_dim)
    for option in options:
        name += '_{}'.format(option)
        command += ' --{}'.format(option)
    if master_endpoint:
        name += '_remote'
        command += ' --master_endpoint {}'.format(master_endpoint)
    if bind_ip:
        command += ' --bind_ip {}'.format(bind_ip)
    name += '_' + str(np)
    command += ' 1>{}.out 2>{}.err'.format(name, name)
    print(command)
    os.system(command)
    time.sleep(1)


if len(sys.argv) > 3:
    # remote
    user = sys.argv[1]
    remote_ip = sys.argv[2]
    bind_ip = sys.argv[3]
    port = 61000
    for model in ['WDL', 'DeepFM']:
        for embedding_dim in [9, 64]:
            for options in [['server'], ['server', 'cache'], ['server', 'cache', 'prefetch']]:
                for np in [1, 2, 4, 8]:
                    port += 1
                    time.sleep(60)
                    run_remote_server(user, remote_ip, port)
                    time.sleep(60)
                    run('deepctr_criteo', 'tfrecord', model, embedding_dim, options, np=np,
                          bind_ip=bind_ip, master_endpoint='{}:{}'.format(remote_ip, port))
else:
    #local
    for data in ['tfrecord', 'train.csv']:
        for model in ['WDL', 'DeepFM', 'xDeepFM']:
            for embedding_dim in [9, 64]:
                for options in [[], ['server'], ['server', 'cache'], ['server', 'cache', 'prefetch']]:
                    for np in [1, 2, 4, 8]:
                        run('deepctr_criteo', data, model, embedding_dim, options, np=np)
                    run('deepctr_criteo', data, model, embedding_dim, options + ['cpu'], np=1)

    for model in ['WDL', 'DeepFM']:
        for embedding_dim in [9, 64]:
            for np in [1, 2, 4, 8]:
                run('deepctr_criteo_torch', data, model, embedding_dim, [], np=np)

    for model in ['WDL', 'DeepFM']:
        for embedding_dim in [9, 64]:
            run('deepctr_criteo_torch', data, model, embedding_dim, ['cpu'], np=1)

