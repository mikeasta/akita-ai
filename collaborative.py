# Collaborative filtering

# Dependencies
import numpy as np
from config import RATING_TO_RECOMMEND, SAME_RATIO
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

    
# Recommendation UX
def recommend(cosine_values, user_ratings):

    # Check, if we need to make a prediction
    if 0 not in user_ratings: return

    # Descending sort
    cosine_values.sort(key=lambda item: sorted(list(item.keys())), reverse=True)
    print(cosine_values)

    # Define, which goods we should recommend or not
    print("\nUser ratings:", user_ratings)

    # Buffer rating
    buffer_user_rating = list.copy(user_ratings)

    # Handle each vector to predict users rating
    for user_index, user_data in enumerate(cosine_values):
        
        # Get it's key & property
        key_distance = list(user_data.keys())[0]
        prop_vector  = user_data[key_distance]
        print("Vector", user_index + 1, ":", key_distance, ", ", prop_vector)

        # Check if difference is too big
        if key_distance < SAME_RATIO: break

        # Handle every coordinate in prop vector
        for i in range(5):
            # Check if data and current prop vector coordinate is already written
            if prop_vector[i] != 0 and buffer_user_rating[i] == 0:
                buffer_user_rating[i] = prop_vector[i]
    
        if 0 not in buffer_user_rating: break

    print("Most nearest prediction:", buffer_user_rating)
    print()
    
    # Say, if we recommend this good to recommend 
    for i in range(5):
        # Check, if rating is already written
        if user_ratings[i] != 0: continue

        recommend_string = ""
        if buffer_user_rating[i] == 0:
            recommend_string = "we can't make a reliable prediction with that good"
        elif buffer_user_rating[i] < RATING_TO_RECOMMEND:
            recommend_string = "we don't recommend this good"
        else: 
            recommend_string = "we recommend this good"

        print("Good", i + 1, ":", recommend_string)


recommend(cosine_values, handling_user_ratings)

