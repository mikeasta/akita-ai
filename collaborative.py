# Collaborative filtering

# Dependencies
import numpy as np
import json_implementation as JI

# Getting data
ratings = JI.json_get_data()

# New user stats
handling_user_ratings = [1, 3, 0, 0, 0]

# Cosine calculation
def cosine(user_a, user_b):
    vector_a = np.array(user_a)
    vector_b = np.array(user_b)

    length_a = np.linalg.norm(vector_a)
    length_b = np.linalg.norm(vector_b)

    distance = np.dot(vector_a, vector_b) / np.dot(length_a, length_b)
    return distance

# Result array
cosine_values = []
for user_ratings in ratings:
    current_distance = cosine(handling_user_ratings, user_ratings)
    cosine_values.append({current_distance: user_ratings})

# Descending sort
cosine_values.sort(key=lambda item: sorted(list(item.keys())), reverse=True)
 
print(cosine_values)


