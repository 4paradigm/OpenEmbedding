import os
import sys
times = dict()

for name in os.listdir(sys.argv[1]):
    time = 100000000
    for line in open(sys.argv[1] + '/' + name):
        r = line.find('s - loss')
        l = line.find('-')
        if l > r:
            l = line.find(':')
        if l != -1 and r != -1:
            time = min(time, int(line[l+1:r]))
        sp = len(name) - 6
        key, np = name[:sp], name[sp:]
        times.setdefault(key, [0, 0, 0, 0])
        if np == '_1.out':
            times[key][0] = time
        if np == '_2.out':
            times[key][1] = time
        if np == '_4.out':
            times[key][2] = time
        if np == '_8.out':
            times[key][3] = time

for key, value in sorted(times.items()):
    print(key, *value)