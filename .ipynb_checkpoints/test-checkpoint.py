import json

path = 'data/PatientBackground/1.json'

with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(data[0])

inner_data_str = data[0].replace("'", "\"")

d1 = json.loads(inner_data_str)

print(type(d1))
print(type(d1[0]))
print(d1[0])