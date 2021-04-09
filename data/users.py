import config 
import json
import json_implementation as JI


# Add new user
def add_new_user(user_name):
    # Get JSON Data
    data = JI.json_get_data()

    # User model
    user = {
        "user_name": str(user_name),
        "eaten": []
    }

    # Check, if user already exists
    found = False
    for user_item in data["users"]:
        if (user_item["user_name"] == user_name):
            found = True
            break

    if not found :
        data["users"].append(user)

    JI.json_push_data(json.dumps(data))


# Edit/Create new rating
def edit_user_rating(user_name, feed_id, rating):
    # Get JSON Data
    data = JI.json_get_data()

    # Convert to string
    feed_id = str(feed_id)
    rating  = str(rating)

    # Check if user exists
    for user_item in data["users"]:
        if (user_item["user_name"] == user_name):

            # Check if "eaten" contains special feed
            found = False
            for feed_item in user_item["eaten"]:
                if (feed_item["feed_id"] == feed_id):
                    feed_item["rating"] = rating
                    found = True

            # If not, append new feed dict.
            if not found:
                feed = {
                    "feed_id" : feed_id,
                    "rating"  : rating
                }
                user_item["eaten"].append(feed)
    
    # Save data
    JI.json_push_data(json.dumps(data))


# Delete user
def delete_user(user_name):
    # Get JSON Data
    data = JI.json_get_data()

    # Finding user dict.
    for user in data["users"]:
        if (user["user_name"] == user_name):
            data["users"].remove(user)

    # Save data
    JI.json_push_data(json.dumps(data))
