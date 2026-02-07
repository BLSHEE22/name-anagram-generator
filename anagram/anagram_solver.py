from collections import Counter
from functools import lru_cache
import random

# ==============================
# CONFIGURATION
# ==============================

BEAM_WIDTH = 6
MIN_WORD_LENGTH = 1

# Weighting constants (tune safely)
HOG_PENALTY_SCALE = 0.1
COMMON_WORD_BONUS = 3.0
WORD_COUNT_PENALTY = 0.25
LENGTH_BONUS = 0.15

# Only accept common two-letter words
VALID_TWO_LETTER_WORDS = {
    "of", "to", "in", "on", "at", "by", "up", "if", "is", "it",
    "as", "or", "we", "he", "me", "my", "us", "do", "go", "no",
    "so", "am"
}

# ==============================
# DICTIONARY LOADING
# ==============================

def load_dictionary(path):
    words = []
    with open(path, "r") as f:
        for line in f:
            w = line.strip().lower()

            # Alphabetic only
            if not w.isalpha():
                continue

            # One-letter words: reject
            if len(w) == 1:
                continue

            # Two-letter words: whitelist only
            if len(w) == 2 and w not in VALID_TWO_LETTER_WORDS:
                continue

            words.append(w)

    return words


# ==============================
# SCORING FUNCTIONS
# ==============================

def word_priority(word, remaining_letters):
    """
    Score a word relative to remaining letters.
    """
    score = 0.0

    # Prefer common-length words
    score += len(word) * LENGTH_BONUS

    # Penalize hogging letters
    total_letters = sum(remaining_letters.values())
    usage_ratio = len(word) / total_letters
    score -= (usage_ratio ** 0.5) * HOG_PENALTY_SCALE

    return score


def phrase_score(words):
    """
    Final phrase-level scoring.
    """
    score = 0.0

    score -= len(words) * WORD_COUNT_PENALTY

    # Small reward for longer average word length
    avg_len = sum(len(w) for w in words) / len(words)
    score += avg_len * 0.1

    return score


# ==============================
# SEARCH CONSTRAINTS
# ==============================

def max_word_length(remaining_counter, depth):
    total = sum(remaining_counter.values())

    if depth == 0:
        return int(total * 0.6)
    elif depth == 1:
        return int(total * 0.75)
    else:
        return total


# ==============================
# CORE SEARCH
# ==============================

def counter_fits(word_counter, remaining_counter):
    return all(word_counter[c] <= remaining_counter[c] for c in word_counter)


def subtract_counter(a, b):
    result = a.copy()
    for k in b:
        result[k] -= b[k]
        if result[k] == 0:
            del result[k]
    return result


@lru_cache(maxsize=None)
def search(remaining_tuple, depth):
    remaining = Counter(dict(remaining_tuple))

    if not remaining:
        return [[]]

    max_len = max_word_length(remaining, depth)

    # Generate candidate words
    candidates = []
    for word, wc in WORD_COUNTERS:
        if len(word) > max_len:
            continue
        if counter_fits(wc, remaining):
            candidates.append(word)

    if not candidates:
        return []

    # Sort candidates by priority
    candidates.sort(
        key=lambda w: word_priority(w, remaining),
        reverse=True
    )

    # Beam search
    candidates = candidates[:BEAM_WIDTH]

    results = []

    for word in candidates:
        wc = Counter(word)
        new_remaining = subtract_counter(remaining, wc)

        sub_results = search(tuple(sorted(new_remaining.items())), depth + 1)
        for sr in sub_results:
            results.append([word] + sr)

    return results


# ==============================
# PUBLIC API
# ==============================

def solve_anagram(name, dictionary_path):
    letters = Counter(c for c in name.lower() if c.isalpha())

    words = load_dictionary(dictionary_path)

    global WORD_COUNTERS
    WORD_COUNTERS = [(w, Counter(w)) for w in words]

    raw_results = search(tuple(sorted(letters.items())), 0)

    if not raw_results:
        return []

    # Final scoring & sort
    scored = [
        (" ".join(r), phrase_score(r))
        for r in raw_results
    ]

    scored.sort(key=lambda x: x[1], reverse=True)

    return scored


# ==============================
# CLI TEST
# ==============================

if __name__ == "__main__":
    name = "Carina Gregori Assadourian"
    results = solve_anagram(name, "enable1.txt")

    for phrase, score in results[:50]:
        print(f"{phrase}  [{score:.2f}]")
        # sorted_a = sorted([p.lower() for p in phrase if p != ' '])
        # sorted_name = sorted([n.lower() for n in name if n != ' '])
        # valid_anagram = sorted_a == sorted_name
        # print(f"Valid anagram: {valid_anagram}")
        # if not valid_anagram:
        #     raise Exception
        # print()

