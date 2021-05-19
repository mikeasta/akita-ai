# Dependencies
import json
import config

# Database file name
file_name = config.DATABASE_FILE_NAME

# Get data
def json_get_data():
    data = {}

    # Getting data
    with open(file_name, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data