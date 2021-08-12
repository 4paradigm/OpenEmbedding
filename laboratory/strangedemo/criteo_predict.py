import os
import json
import pandas
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--rows', type=int, required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--host', required=True)
args = parser.parse_args()
data = pandas.read_csv(args.data, nrows=args.rows)

inputs = dict()
for name in data.columns:
    if name[0] == 'C':
        inputs[name] = [[int(value)] for value in data[name]]
    elif name[0] == 'I':
        inputs[name] = [[float(value)] for value in data[name]]
post = json.dumps({'inputs':inputs})
command = f"curl -d '{post}' {args.host}/v1/models/{args.model}:predict"
print(command)
os.system(command)

