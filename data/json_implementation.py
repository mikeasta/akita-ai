import json
import config

# Database file
file_name = config.FILE_NAME

# Get data
def json_get_data():
    data = {}
    with open(file_name, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
        data = json.loads(data)
    return data

json_get_data()

# Push data
def json_push_data(data):
    with open(file_name, "w") as json_file:
        json.dump(data, json_file)