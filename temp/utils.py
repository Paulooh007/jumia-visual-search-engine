import json


def load_config(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data
