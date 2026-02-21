import csv
from utils import *
from config import *
from constants import *
import datetime
from collections import Counter
from anagram.generator import normalize, filter_valid_words, load_words

# load words
words = load_words()


# convert float seconds to duration
def seconds_to_duration(seconds):
    intervals = (
        ('years', 31557600),
        ('days', 86400),
        ('hours', 3600),
        ('minutes', 60),
        ('seconds', 1),
    )

    result = {}
    remainder = int(seconds)

    for name, count in intervals:
        value = remainder // count
        remainder %= count
        result[name] = value

    return result


# find exhaustive anagram search duration for name
def project_search_durations(name):
    name_length = len(normalize(name))
    valid_word_dict = filter_valid_words(name, words, Counter(normalize(name)), len(name))
    #draw_histogram([(key, len(value)) for key, value in valid_word_dict.items()])
    word_ct_dict = {key: len(value) for key, value in valid_word_dict.items()}
    search_times = sorted([(key, 0.1*val*pow(2, name_length-key)) for key, val in word_ct_dict.items()], key=lambda x: x[1])
    return search_times
    

# find projections for each name in db
def calc_db_exhaust_durations():
    search_durations = []
    with open(DB_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = row["input"]
            search_durations = project_search_durations(name)
            print(round(sum([p[1] for p in search_durations]), 2))


# ask user for name and calc exhaust duration
def project_exhaust_duration():
    print("===========================================================================================")
    name = input("--- Enter a name, and I'll tell you how long it would take to find all of its anagrams. ---\n" +
                "===========================================================================================\n")
    search_durations = project_search_durations(name)
    total_search_duration = seconds_to_duration(round(sum([p[1] for p in search_durations]), 0))
    print()
    print(total_search_duration)
    print()
    

#calc_db_exhaust_durations()
#project_exhaust_duration()


