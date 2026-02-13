from collections import Counter, deque
import re
import sys
import time
import torch
import random
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
FIT_WORDS = set(["hello", "ally", "relay", "jock", "man", "ban", "van", "elk", "new", "hey"])
HELPER_WORDS = set(["oh", "aim", "love", "run", "the", "and", "cat", "dog", "mr", "ms", "mrs", "dr"])

# Only accept common two-letter words
VALID_TWO_LETTER_WORDS = {
    "of", "to", "in", "on", "at", "by", "up", "if", "is", "it",
    "as", "or", "we", "he", "me", "my", "us", "do", "go", "no",
    "so", "am", "ew", "mr", "ms", "dr", "oh", "an"
}

def phrase_bonus_score(phrase, baseline_words=10):
    """Assigns heuristic score for CPU prioritization."""
    score = 0.0
    words_in_phrase = phrase.split()
    for w in words_in_phrase:
        if w in VERY_FIT_WORDS:
            score += 0
        elif w in FIT_WORDS:
            score += 0
        elif w in HELPER_WORDS:
            score += 0
    return score

def phrase_length_multiplier(phrase, baseline_words=3):
    """Favor fewer words: multiplicative boost"""
    num_words = len(phrase.split())
    return 1.0 + 0.1 * max(0, baseline_words - num_words)

def length_score(word, name_length, first_word_length):
    """
    Favor words of length [first_word_length]
    """
    # The closer the length to desired, the higher the score
    return -abs(len(word) - first_word_length)
    rand_reward = random.randrange(1, 20) * len(word)
    return rand_reward

def letter_hog_penalty(word, remaining_counter):
    total_letters = sum(remaining_counter.values())
    usage = len(word) / total_letters
    return -usage * 10

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
        #print(f"Leftover letter: {word}")
        #print(f"Remaining words: {previous_words}")
        if previous_words:
            previous_words_lower = [p.lower() for p in previous_words]
            # accept 'a' if not already existing AND consonant-starting word already exists
            if word == 'a' and 'a' not in previous_words_lower and not all(w[0] in ["a", "e", "i", "o", "u"] for w in previous_words_lower):
                return True
            # accept one letter word if honorific exists unless word already exists
            elif any(w in previous_words_lower for w in ["mr", "ms", "mrs", "dr"]) and (word not in previous_words_lower):
                return True    
            # otherwise don't accept
            else:
                return False
        else:
            return False
    elif word == "an":
        if previous_words:
            previous_words_lower = [p.lower() for p in previous_words]
            # accept 'an' if not already existing AND vowel-starting word already exists
            if word not in previous_words_lower and any(w[0] in ["a", "e", "i", "o", "u"] for w in previous_words_lower):
                return True
            else:
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

    # sort by word length, accounting for 'too large' words
    valid.sort(key=lambda w: length_score(w, len(name), first_word_length) +
                              random.uniform(0, 0.01), reverse=True)

    return valid

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

        current_phrase, remaining = stack.pop()

        if not any(remaining.values()):
            candidate = " ".join(current_phrase)
            #print(GREEN + f"\nACCEPTED: {current_phrase}\n" + RESET)
            results.append(candidate)
            continue

        if len(current_phrase) >= max_words:
            continue

        valid_words = filter_valid_words(name, words, remaining, first_word_length)
        M = min(len(valid_words), beam * 2)
        top_valid_words = valid_words[:M]
        valid_words_shuffled = random.sample(top_valid_words, min(beam, len(top_valid_words)))
        for w in valid_words_shuffled:
            #print(f"{w} : {current_phrase}")
            if not is_word_allowed(w, current_phrase):
                continue  # skip invalid single-letter word
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

# ===========================
# --- Full optimized pipeline ---
# ===========================

def generate_top_anagrams(name, model, tokenizer, time_limit=60, top_n=3, max_words=6, beam=10, limit=200, baseline_words=3, first_word_length=20):
    words = load_words(max_len=len(normalize(name)))

    # start time limit for search
    start = time.perf_counter()

    # run anagram finder
    candidates = generate_anagrams_guided(name, start_time=start, words=words, max_words=max_words,
                                          limit=limit, beam=beam, time_limit=time_limit, first_word_length=first_word_length)

    if not candidates:
        return ["[No valid anagrams found]"]

    # Batched scoring
    scored = score_anagrams_batch(name, candidates, model, tokenizer)

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
        bonus = phrase_bonus_score(p)
        length_mul = phrase_length_multiplier(p, baseline_words)
        final_scores.append((p, (s + bonus) * length_mul))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    top_phrases = [p for p, _ in final_scores[:top_n]]

    return top_phrases
