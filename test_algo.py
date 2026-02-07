from anagram.generator import generate_anagrams
import random

name = "Grace Hopper"

results = generate_anagrams(name, max_words=3)

sample_anagrams = random.sample(results, 10)

for s in sample_anagrams:
    print(s)

