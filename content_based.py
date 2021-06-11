# Content-Based filtering

# Dependencies
import numpy as np
import json_implementation as ji
import string
import spacy
from stop_words import get_stop_words


# Texts
texts = ji.json_get_data()["content_based_texts"]
document_amount = len(texts)


# Initialazing spacy model
nlp = spacy.load("en_core_web_sm")


# Current user text
current_user_text = texts[0]


# Stop words library
stop_words_array = get_stop_words("en")


# Cosine distance
def cosine(vector_a, vector_b):
    length_a = np.linalg.norm(vector_a)
    length_b = np.linalg.norm(vector_b)

    return np.dot(vector_a, vector_b) / np.dot(length_a, length_b)


# Calculate text's words weight
def calc_words_weight(text):

    # Lemmatizing
    word_array = nlp(text)
    text = " ".join([token.lemma_ for token in word_array])

    # Deleting stop words
    word_array = [word.strip(string.punctuation).lower() for word in text.split() if word.strip(string.punctuation).lower() not in stop_words_array]
    word_vocabulary = {}

    for word in word_array:
        word_vocabulary[word] = word_vocabulary[word] + 1 if word in word_vocabulary else 1
        
    unique_words_amount = len(word_vocabulary)
        
    for word in word_vocabulary:
        # IDF value 
        found_times = 0
        for i in range(document_amount):
            if texts[i].lower().find(word) != -1:
                found_times += 1

        idf = np.log10(document_amount / (found_times or 1))

        # TF value
        tf = word_vocabulary[word] / unique_words_amount

        # TF-IDF value
        word_vocabulary[word] = tf * idf
    
    return word_vocabulary
    

# Calculate cosine distance between Vector-Spaces
def calc_cosine_distances(text_to_compare):
    cosine_distances = []

    for i in range(document_amount):
        comparable_text_library = calc_words_weight(text_to_compare)
        current_text_library    = calc_words_weight(texts[i])

        for word in comparable_text_library:
            if word not in current_text_library:
                current_text_library[word] = 0

        for word in current_text_library:
            if word not in comparable_text_library:
                comparable_text_library[word] = 0

        vector_comparable = [comparable_text_library[word] for word in comparable_text_library]
        vector_current    = [current_text_library[word]    for word in comparable_text_library]

        cosine_distances.append(cosine(vector_comparable, vector_current))
    
    return cosine_distances


cosine_distance = calc_cosine_distances(current_user_text)
print(cosine_distance)