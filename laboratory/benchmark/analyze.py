import pandas
data = pandas.read_csv('train_1000w.csv')

cache = dict()

feature = list()
batch_size = 4096
all_whole_unique = 0
all_related_unique = 0
for name in data.columns:
    if name[0] != 'C':
        continue
    cache[name] = set()
    column = data[name]
    whole = 0
    whole_unique = 0
    related = 0
    related_unique = 0
    for i in range(1, 100):
        prev = set()
        for j in range(batch_size):
            cache[name].add(column[(i - 1) * batch_size + j])
            prev.add(column[(i - 1) * batch_size + j])
        rlt = list()
        whl = column[i * batch_size: (i + 1) * batch_size]
        cache_hit = list()
        for key in whl:
            if key in prev:
                rlt.append(key)
            if key in cache[name]:
                cache_hit.append(key)
        whole += len(whl)
        whole_unique += len(set(whl))
        related += len(rlt)
        related_unique += len(set(rlt))
        if i == 64:
            print(name, data[name].max() + 1, len(set(whl)), len(set(cache_hit)))
    feature.append([name, [whole, whole_unique, related, related_unique]])
    all_whole_unique += whole_unique
    all_related_unique += related_unique
    print(name, whole, related)
    print(name, whole_unique, related_unique)

print()
print(all_whole_unique, all_related_unique)
feature = sorted(feature, key=lambda x: x[1][1])
for name, values in feature:
    print(name, values[1] / values[0], values[2] / values[0], values[3] / values[1])
