from collections import Counter, deque, defaultdict
import re
import sys
import time
import torch
import random
import itertools
from readchar import readkey
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ===========================
# --- Word lists / scoring ---
# ===========================

# Example lists — you can fill these with your full word lists
VERY_FIT_WORDS = set(["sex", "ass", "anal", "ew", "nerd", "ham", "oil", "hog", "jail"])
FIT_WORDS = set(["hello", "ally", "relay", "jock", "like", "man", "ban", "hick", "van", "elk", "new", "hey", "black"])
HELPER_WORDS = set(["oh", "aim", "love", "run", "the", "and", "cat", "dog", "mr", "ms", "mrs", "dr",
                    "nhl", "ny", "nj", "aa", "ma", "nh"])

# Accepted two-letter words
VALID_TWO_LETTER_WORDS = {
    "of", "to", "in", "on", "at", "by", "up", "if", "is", "it",
    "as", "or", "we", "he", "me", "my", "us", "do", "go", "no",
    "so", "am", "ew", "mr", "ms", "dr", "sr", "oh", "an", "aa", 
    "ny", "nj", "ma", "nh", "mt", "kc", "cc", "nz", "va", "rv", 
    "sa", "la", "vr", "tv", "ad", "hc", "bj", "hi", "ea", "xl", 
    "id", "rc", "ed", "ho", "nc", "be", "ab", "ax", "jr"
}

# Words that need to be all-caps
ALL_CAPS_WORDS = {"aa", "ny", "nj", "nh", "kc", "cc", "nz", "va", 
                  "rv", "sa", "la", "vr", "tv", "hc", "bj", "ea", 
                  "xl", "rc", "nc"
}

def phrase_bonus_score(phrase, baseline_words=5):
    """Assigns heuristic score for CPU prioritization."""
    score = 0.0
    words_in_phrase = phrase.split()
    for w in words_in_phrase:
        if w in VERY_FIT_WORDS:
            score += 3
        elif w in FIT_WORDS:
            score += 2
        elif w in HELPER_WORDS:
            score += 1
    # alliteration bonus
    starting_letters = [w[0] for w in words_in_phrase]
    diff_starting_letters = len(set(starting_letters))
    if diff_starting_letters == 1:
        score += 50
    else:
        score += 1
    return score

def phrase_length_multiplier(phrase, baseline_words=5):
    """Favor fewer words: multiplicative boost"""
    num_words = len(phrase.split())
    return 1.0 + pow(3, max(0, baseline_words - num_words))

def length_score(word, name_length, first_word_length):
    """
    Favor words closest to length [first_word_length].
    """
    #return random.randrange(1, 20)
    return -abs(len(word) - first_word_length)

def max_consecutive_match(word, name):
    """
    Returns the length of the longest consecutive substring of `word` found in `name`.
    Case-insensitive.
    """
    word = word.lower()
    name = name.lower()
    max_match = 0

    for i in range(len(word)):
        for j in range(i + 1, len(word) + 1):
            substr = word[i:j]
            if substr in name:
                max_match = max(max_match, j - i)
            else:
                # longer substrings starting at i won't match
                break
    
    return max_match


# ===========================
# --- Helpers ---
# ===========================

def normalize(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())

def letter_count(text: str) -> Counter:
    return Counter(normalize(text))

def is_subset(counter1, counter2):
    return all(counter1[c] <= counter2.get(c, 0) for c in counter1)

def is_word_allowed(word, previous_words):
    """
    Returns True if the word is allowed in this position.
    - e.g. reject single-letter words like 'T' unless preceded by 'Mr'
    """
    if len(word) == 1:
        #print(f"Leftover single letter: {word}")
        #print(f"Previous words: {previous_words}")
        if previous_words:
            # accept 'a' if not already existing AND consonant-starting word already exists
            if word == 'a' and 'a' not in previous_words and not all(w[0] in ["a", "e", "i", "o", "u"] for w in previous_words):
                #print(f"Accepted 'a' because consonant-starting word exists in {previous_words}.")
                return True
            # accept one letter word if honorific exists unless word already exists
            elif any(w in previous_words for w in ["mr", "sr", "ms", "mrs", "dr"]) and (word not in previous_words):
                #print(f"Accepted '{word}' because honorific exists in {previous_words}.")
                return True    
            elif word == 'i' and 'am' in previous_words:
                #print(f"Accepted 'i' because 'am' exists in {previous_words}.")
                return True
            # otherwise don't accept
            else:
                #print(f"Rejected one-letter word '{word}' because no beginning consonant or honorific exists in {previous_words}.")
                return False
        else:
            #print(f"Rejected one-letter word '{word}' because no words exist in {previous_words}.")
            return False
    elif word == "an":
        if previous_words:
            # accept 'an' if not already existing AND vowel-starting word already exists
            if word not in previous_words and any(w[0] in ["a", "e", "i", "o", "u"] for w in previous_words):
                #print(f"Accepted 'an' because vowel-starting word exists in {previous_words}.")
                return True
            else:
                #print(f"Rejected 'an' because no vowel-starting word exists in {previous_words}.")
                return False
    elif word in previous_words:
        # don't accept a repeat word
        #print(f"Rejected repeat word '{word}' which was already in {previous_words}.")
        return False
    return True

def load_words(path="anagram/enable1.txt", max_len=None):
    words = []
    with open(path) as f:
        for line in f:
            w = line.strip().lower()

            # Alphabetic only
            if not w.isalpha():
                continue

            ## removed one-letter rejection
            # if len(w) == 1 and w not in ['a', 'i']:
            #     continue

            # Two-letter words: whitelist only
            if len(w) == 2 and w not in VALID_TWO_LETTER_WORDS:
                continue

            words.append(w)
    return words

def filter_valid_words(name, words, remaining_counter, first_word_length):
    valid = [w for w in words if is_subset(Counter(w), remaining_counter)]

    # prevents alphabetical bias
    random.shuffle(valid)

    # reward word length
    valid.sort(key=lambda w: length_score(w, len(name), first_word_length) +
                             random.uniform(0, 0.01), reverse=True)

    filtered_group = dict()
    if valid:
        group_by_len = defaultdict(list)

        for s in valid:
            group_by_len[len(s)].append(s)

        #print(f"Unfiltered word group dict: {group_by_len}")
        filtered_group = {key: value for key, value in group_by_len.items() if key <= first_word_length}
        #print(f"Words less than or equal to length {first_word_length}: {filtered_group}")

    return filtered_group

# =========================================
# --- Break function for runaway search ---
# =========================================

def exceeded_time_limit(start, limit=60, debug=False):
    curr_time = time.perf_counter()
    elapsed = curr_time - start
    if debug:
        print(f"Elapsed seconds: {elapsed}")
        print(f"Limit: {limit}")
    return elapsed > limit

# ===========================
# --- Guided anagram generation ---
# ===========================

def generate_anagrams_guided(name, start_time, words=None, max_words=6, limit=200, beam=10, time_limit=60, first_word_length=20):
    name_counter = letter_count(name)
    if words is None:
        words = load_words(max_len=len(normalize(name)))
    
    results = []
    stack = deque()
    stack.append( ([], name_counter) )

    

    while stack and len(results) < limit:

        # TODO
        # if 's' pressed, stop the search
        #global stop_requested
        # if stop_requested:
        #     print("Stopped search at user request.")
        #     break

        current_phrase, remaining = stack.popleft()

        if not any(remaining.values()):
            candidate = " ".join(current_phrase)
            #print(GREEN + f"\nACCEPTED: {current_phrase}\n" + RESET)
            results.append(candidate)
            continue

        if len(current_phrase) >= max_words:
            continue

        valid_word_group_dict = filter_valid_words(name, words, remaining, first_word_length)
        ### DEBUG
        # print(f"First word length: {first_word_length}")
        # print(stack)
        # print(current_phrase)
        # print(remaining)
        # print(valid_word_group_dict)
        valid_word_groups = list(valid_word_group_dict.values())
        for g in valid_word_groups:
            random.shuffle(g)
        top_valid_words = list(itertools.chain.from_iterable(valid_word_groups))[:beam]
        # only search starting word length paths on first pass
        #print(top_valid_words)
        if not stack and not current_phrase:
            top_valid_words = [w for w in top_valid_words if len(w) == first_word_length]
        #print(top_valid_words)
        #print()
        for w in top_valid_words:
            if not is_word_allowed(w, current_phrase):
                continue 
            next_remaining = remaining.copy()
            next_remaining.subtract(Counter(w))
            next_remaining = +next_remaining  # remove zeros/negatives
            # Always append — don't discard sequences too early
            stack.append((current_phrase + [w], next_remaining))
        
        # stop searches that take longer than the time limit
        if exceeded_time_limit(start_time, time_limit):
            stop_requested = True
            break

    return results

# ===========================
# --- Batched T5 scoring ---
# ===========================

def score_anagrams_batch(input_text, phrases, model, tokenizer, batch_size=16):
    scores = []
    for i in range(0, len(phrases), batch_size):
        batch_phrases = phrases[i:i+batch_size]
        inputs = tokenizer([input_text]*len(batch_phrases), return_tensors="pt", padding=True)
        labels = tokenizer(batch_phrases, return_tensors="pt", padding=True).input_ids
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            batch_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1))
            seq_lengths = (shift_labels != -100).sum(dim=1)
            seq_losses = batch_losses.view(shift_labels.size()).sum(dim=1) / seq_lengths
            batch_scores = (-seq_losses).tolist()
            scores.extend(list(zip(batch_phrases, batch_scores)))
    return scores

def finalize_scores(name, anagrams, model, tokenizer, baseline_words=5, batch_size=16, beautify=False):

    # Batched scoring
    scored = score_anagrams_batch(name, anagrams, model, tokenizer, batch_size)

    # Normalize T5 scores to [0,1]
    t5_scores = [s for _, s in scored]
    min_s, max_s = min(t5_scores), max(t5_scores)
    scaled = []
    for p, s in scored:
        norm_s = (s - min_s)/(max_s - min_s) if max_s != min_s else 0.5
        scaled.append((p, norm_s))

    # Apply subtle bonuses and length multiplier
    final_scores = []
    for p, s in scaled:
        #print(BOLD + f"Anagram: {p}" + RESET)
        bonus = phrase_bonus_score(p)
        length_mul = phrase_length_multiplier(p, baseline_words)
        resemblance_penalty = max_consecutive_match(p, name)
        #print(f"P: {p}, S: {s}, Bonus: {bonus}\n" + 
              #f"Length mult: {length_mul}, Resemb. penalty: {resemblance_penalty}\n")
        final_scores.append((p, (s + bonus - (1.5 * resemblance_penalty)) + length_mul))

    final_scores.sort(key=lambda x: x[1], reverse=True)

    # if beuatification stage, make acronyms all-caps
    if beautify:
        final_formatted_scores = []
        for p in final_scores:
            a_split = [q.lower() for q in p[0].split() if q.isalpha()]
            new_a_unjoined = []
            for w in a_split:
                w = w.title()
                if w.lower() in ALL_CAPS_WORDS:
                    #print(f"Making '{w}' all uppercase!")
                    w = w.upper()
                new_a_unjoined.append(w)
            new_a = " ".join(new_a_unjoined)
            final_formatted_scores.append((new_a, p[1]))
        final_scores = final_formatted_scores

    # disqualify anagrams which kept entire words from the name
    split_name = [n.lower() for n in name.split() if n.isalpha()]
    final_scores = [p for p in final_scores if not any(q in split_name for q in p[0].split())]
    return final_scores

# ===========================
# --- Full optimized pipeline ---
# ===========================

def generate_top_anagrams(name, model, tokenizer, time_limit=60, top_n=3, max_words=6, beam=10, limit=200, baseline_words=5, first_word_length=20):
    words = load_words(max_len=len(normalize(name)))

    # start time limit for search
    start = time.perf_counter()

    # run anagram finder
    candidates = generate_anagrams_guided(name, start_time=start, words=words, max_words=max_words,
                                          limit=limit, beam=beam, time_limit=time_limit, first_word_length=first_word_length)

    if not candidates:
        return ["[No valid anagrams found]"]

    # score candidates
    final_scores = finalize_scores(name, candidates, model, tokenizer)
    top_phrases = [p for p, _ in final_scores]

    # print(f"Top phrases: {top_phrases}")

    return top_phrases
