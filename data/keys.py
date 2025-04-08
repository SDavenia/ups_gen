import json
with open('prompting/additional_context.json', 'r') as f:
    data = json.load(f)

print([key for key in data.keys()])
