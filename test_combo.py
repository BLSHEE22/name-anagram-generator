import math
import numpy as np
from collections import Counter
from functools import lru_cache
from anagram.generator import load_words

# ========================================
# --- Count combinations ---
# ========================================

def leftover_permutations(counter):
    total = sum(counter.values())
    numerator = math.factorial(total)
    denominator = 1
    for c in counter.values():
        denominator *= math.factorial(c)
    return numerator // denominator


def exact_total_anagrams(name, starting_words):
    name = name.lower().replace(" ", "")
    base_counter = Counter(name)

    total = 0

    for word in starting_words:
        word_counter = Counter(word.lower())
        leftover = base_counter - word_counter
        total += leftover_permutations(leftover)

    return total

# ========================================
# --- Recursive check using dictionary ---
# ========================================

def multiset_to_string(counter):
    """Convert a Counter of letters to a sorted string for memoization."""
    letters = []
    for c, count in counter.items():
        letters.extend([c]*count)
    return ''.join(sorted(letters))


def exact_total_anagrams_problem_b(name, starting_words, word_dict):
    """
    Counts total arrangements where:
        - first N letters form a word (from starting_words)
        - leftover letters form valid words (from word_dict)
    """
    name_counter = Counter(name.lower().replace(" ", ""))
    
    # Prepare a cached recursive function
    @lru_cache(maxsize=None)
    def count_leftovers(leftover_str):
        leftover_counter = Counter(leftover_str)
        if sum(leftover_counter.values()) == 0:
            return 1  # valid combination
        total = 0
        for word in word_dict:
            word_counter = Counter(word)
            # Can we form this word from leftovers?
            if all(leftover_counter[c] >= word_counter[c] for c in word_counter):
                new_leftover = leftover_counter - word_counter
                total += count_leftovers(multiset_to_string(new_leftover))
        return total

    total_anagrams = 0
    for word in starting_words:
        word_counter = Counter(word.lower())
        leftover = name_counter - word_counter
        total_anagrams += count_leftovers(multiset_to_string(leftover))
    
    return total_anagrams

# =================================================
# --- Count mean, max, and total branching load ---
# =================================================

def effective_search_load(name, starting_words):
    """
    Estimate effective search load for a name:
    - Sum of raw factorials of leftover letters for each starting word
    - Captures all permutations explored by naive backtracking search
    Returns:
        total_load: sum over all starting words
        mean_load: average per starting word
        max_load: largest per starting word
    """
    name_counter = Counter(name.replace(" ", "").lower())
    loads = []
    for word in starting_words:
        word_counter = Counter(word.lower())
        leftover = name_counter - word_counter
        total_letters = sum(leftover.values())
        load = math.factorial(total_letters)  # raw permutations
        loads.append(load)
    
    total_load = sum(loads)
    mean_load = np.mean(loads)
    max_load = np.max(loads)
    return total_load, mean_load, max_load



################################################################################################################

dictionary = set(load_words())

starting_words = {
    "Jonathan Abrams": ['marantas', 'trashman', 'maharaja', 'amaranth', 'mastabah', 'boatsman', 'marathon', 'absonant', 'tamaraos', 'sonarman'],
    "Jude Donabedian": ['unbodied', 'abounded', 'unbidden', 'unideaed', 'unjoined', 'aboideau', 'enjoined', 'undenied', 'unbended', 'adjoined'],
    "Anthony Cash": ['chants', 'astony', 'chanty', 'cannas', 'scanty', 'snatch', 'honans', 'sonata', 'octans', 'canyon', 'shanty', 'cytons', 'nachos', 'cannot', 'choana', 'canons', 'annoys', 'sancta', 'stanch', 'yachts', 'nachas', 'shanny', 'ashcan', 'canton', 'sonant', 'cotans', 'cantos'],
    "Connor Hileman": ['chairmen', 'nonmoral', 'chainmen', 'monecian', 'colormen', 'cannelon', 'colinear', 'colorman', 'acromion', 'heroical', 'harmonic', 'nonohmic', 'cinnamon', 'cornmeal', 'ranchmen', 'inchmeal', 'hormonic', 'noncrime', 'monorail', 'coenamor', 'amelcorn', 'acrolein', 'choreman', 'hormonal', 'chlorine', 'heirloom', 'omniarch'],
    "Holly Tente": ['netty', 'telly', 'toney', 'leone', 'thole', 'tolyl', 'hoyle', 'teeny', 'nelly', 'teeth', 'yente', 'hotly', 'tenth', 'lotte', 'hello', 'holey', 'helot', 'tythe', 'honey', 'ethyl', 'hotel', 'tenet', 'holly', 'lento', 'lethe', 'tenty'],
    "Micah White": ['tache', 'amice', 'miche', 'twice', 'white', 'withe', 'hatch', 'cheat', 'watch', 'teach', 'heath', 'hemic', 'mache', 'match', 'tawie', 'which', 'wecht', 'witch', 'theca', 'wheat', 'cheth', 'ethic', 'hitch', 'amici', 'aitch', 'chime'],
    "Emily Cournoyer": ['ceremony', 'recoiler', 'colourer', 'colormen', 'lemurine', 'comelier', 'unicolor', 'relumine', 'uncomely', 'recliner']
}

search_durations = {
    "Jonathan Abrams": 145.5,
    "Jude Donabedian": 101.4,
    "Anthony Cash": 99.6,
    "Connor Hileman": 86.7,
    "Holly Tente": 49.3,
    "Micah White": 49.2,
    "Emily Cournoyer": 42.3
}

# for name, words_list in starting_words.items():
#     print(f"{name}: {exact_total_anagrams(name, words_list, dictionary)}, {search_durations[name]}s")

    # Compute search load features
print(f"{'Name':20} | Search Duration | Total anagrams with with starting word length N")
for name, starters in starting_words.items():
    total = exact_total_anagrams(name, starting_words)
    print(f"{name:20} | {search_durations[name]:14} | {total:10}")
