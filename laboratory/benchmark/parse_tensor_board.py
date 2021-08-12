import sys
import json

profile = json.loads(open(sys.argv[1]).read())

timeline = dict()
for event in profile['traceEvents']:
    if 'name' in event and 'ts' in event:
        p = event['name'].rfind(':')
        name = event['name'][p + 1:]
        timeline.setdefault(name, list())
        timeline[name].append(event)

for name, events in timeline.items():
    l = min(event['ts'] for event in events)
    r = max(event['ts'] + event['dur'] for event in events)
    s = sum(event['dur'] for event in events)
    c = len(events)
    print(name, int(l / 1000), int(r / 1000), int(s / 1000), c)

