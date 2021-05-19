# Collaborative filtering

# Dependencies
import numpy as np
import json_implementation as ji


# Getting data
ratings = ji.json_get_data()["collaborative_filtering_ratings"]


# New user stats
handling_user_ratings = [1, 3, 0, 0, 0]


# Cosine calculation
def cosine(ratings_a, ratings_b):
    length_a = np.linalg.norm(ratings_a)
    length_b = np.linalg.norm(ratings_b)

    distance = np.dot(ratings_a, ratings_b) / np.dot(length_a, length_b)
    return distance


# Result array
cosine_values = []
for user_ratings in ratings:
    current_distance = cosine(handling_user_ratings, user_ratings)
    cosine_values.append({current_distance: user_ratings})

    
# Descending sort
cosine_values.sort(key=lambda item: sorted(list(item.keys())), reverse=True)
print(cosine_values)
