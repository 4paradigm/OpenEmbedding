import os
import json
import pandas
import argparse
parser = argparse.ArgumentParser()
default_data = os.path.dirname(os.path.abspath(__file__)) + '/train100.csv'
parser.add_argument('--data', default=default_data)
parser.add_argument('--rows', type=int, default=None)
parser.add_argument('--hash', action='store_true')
parser.add_argument('--host', default='127.0.0.1:8501')
parser.add_argument('--model', default='criteo')
args = parser.parse_args()


# process data
data = pandas.read_csv(args.data, nrows=args.rows)
feature_names = list()
for name in data.columns:
    if name[0] == 'C':
        if args.hash:
            data[name] = (data[name] + int(name[1:]) * 1000000007) % (2**63)
        else:
            data[name] = data[name] % 65536
        feature_names.append(name)
    elif name[0] == 'I':
        feature_names.append(name)

inputs = dict()
for name in data.columns:
    if name[0] == 'C':
        inputs[name] = [[int(value)] for value in data[name]]
    elif name[0] == 'I':
        inputs[name] = [[float(value)] for value in data[name]]
post = json.dumps({'inputs':inputs})
command = f"curl -d '{post}' {args.host}/v1/models/{args.model}:predict"
print(command)
result = json.load(os.popen(command))
print(json.dumps(result))

if "outputs" not in result or len(result["outputs"]) != data.shape[0]:
    print("get error result!")
    exit(1)
