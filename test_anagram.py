from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from anagram.generator import generate_top_anagrams, score_anagrams_batch
from collections import Counter
from readchar import readkey, key
import torch
import re
import sys
import time
import random

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BOLD = '\033[1m'
RESET = '\033[0m'


# Load model
model_path = "./anagram_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test
print(YELLOW + "\nWelcome to the Name Anagram Generator!\n" + RESET)
full_name = input("Enter a name below.\n")
names = []
names.append(full_name)
for name in names:
    all_anagrams = []
    best_anagrams = []
    for n in range(len(name), 3, -1):
        print(RED + "\n##############################################################################\n" +
               f"\nFinding anagrams for {name} with starting word closest to length {n}...\n" + 
               "\n##############################################################################\n" + RESET)
        top_anagrams = generate_top_anagrams(name, model, tokenizer, top_n=10, rewrite=False, beam=1, limit=200, 
                                             baseline_words=10, first_word_length=n)
        
        # filter out already seen anagrams
        #print(f"All anagrams so far: {all_anagrams}")
        unique_phrases_this_round = set([frozenset(a.split()) for a in top_anagrams])
        #print(f"All unique phrases this round: {unique_phrases_this_round}")
        top_anagrams = [a.title() for a in top_anagrams if frozenset(a.split()) not in all_anagrams]
        #print(f"Top anagrams that have not yet been seen: {top_anagrams}")

        if all("[" not in a for a in top_anagrams):
            if top_anagrams:
                top_angrams_condensed = top_anagrams.copy()
                top_anagrams_condensed = [' '.join(sorted(b.split(), key=len, reverse=True)) for b in set([' '.join(a.split()) for a in top_anagrams])]
                while len(top_anagrams_condensed) > len(unique_phrases_this_round):
                    top_anagrams_condensed = [' '.join(sorted(b.split(), key=len, reverse=True)) for b in set([' '.join(a.split()) for a in top_anagrams_condensed])]
                #print(f"Top anagrams that are unique: {top_anagrams_condensed}")
                print(f"{len(top_anagrams_condensed)} found!\n")
            else:
                top_anagrams_condensed = []
            for i, a in enumerate(top_anagrams_condensed):
                all_anagrams.append(frozenset([b.lower() for b in a.split()]))
                deciding = True
                while deciding:
                    sys.stdout.write(f"{i+1}. \033[1m{a}\033[0m  👍? (↑/↓/→/←): ")
                    sorted_a = sorted([b.lower() for b in a if b != ' '])
                    sorted_name = sorted([c.lower() for c in name if c != ' '])
                    valid_check = sorted_a == sorted_name
                    print(f"Sorted anagram: {sorted_a}")
                    print(f"Sorted name: {sorted_name}")
                    print(f"Valid: {valid_check}")
                    if not valid_check:
                        raise Exception
                    sys.stdout.flush()

                    k = readkey()

                    if k == key.RIGHT:
                        a_words = a.split()
                        if len(a_words) > 1:
                            # rotate clockwise
                            a_words = a_words[-1:] + a_words[:-1]
                            a = ' '.join(a_words)
                        sys.stdout.write("\r\033[K")
                        sys.stdout.flush()
                        sys.stdout.write("\r\033[K")
                        continue
                    elif k == key.LEFT:
                        a_words = a.split()
                        if len(a_words) > 1:
                            # swap first two elements
                            a_words[0], a_words[1] = a_words[1], a_words[0]
                            a = ' '.join(a_words)
                        sys.stdout.write("\r\033[K")
                        sys.stdout.flush()
                        sys.stdout.write("\r\033[K")
                        continue
                    elif k == key.UP:
                        sys.stdout.write(GREEN + f"✔ Accepted\n" + RESET)
                        sys.stdout.flush()
                        time.sleep(0.5)
                        best_anagrams.append(a)
                    elif k == key.DOWN:
                        sys.stdout.write(RED + f"✘ Rejected\n" + RESET)
                        sys.stdout.flush()
                        time.sleep(0.5)
                        #sys.stdout.write("\r\033[K")
                    else:
                        print(YELLOW + "Please enter a valid arrow key." + RESET)
                        #sys.stdout.write("\r\033[K")
                        #sys.stdout.flush()
                        continue
                    deciding = False
            print()
    if best_anagrams:
        print(f"\nHere's how the computer ranks the best anagrams you selected for: {name}\n")
        scored_final_anagrams = sorted(score_anagrams_batch(name, best_anagrams, model, tokenizer, batch_size=max(len(best_anagrams), 5)), key=lambda x: x[1], reverse=True)
        for i, aPair in enumerate(scored_final_anagrams, 1):
            a = aPair[0]
            aScore = aPair[1]
            print(f"{i}. {a} (" + str(round(aScore, 2)) + ")")
        # don't ask user for favorite anagram if only one was selected
        if len(best_anagrams) == 1:
            fav_ans = "1"
        else:
            deciding_final = True
            print("\nBut what's your favorite?\n")
            while deciding_final:
                fav_ans = input()
                try:
                    fav_id = int(fav_ans)
                    temp = best_anagrams[fav_id-1]
                except:
                    print(YELLOW + f"\nPlease enter a valid number 1-{len(best_anagrams)}.\n" + RESET)
                    continue
                deciding_final = False
        print(f"\nWinning anagram for: {name}\n" + BOLD + GREEN + scored_final_anagrams[int(fav_ans)-1][0] + "\n" + RESET)
        print("Hope you enjoyed using the Name Anagram Generator!\n")
    else:
        print(f"Hmm...looks like there weren't any anagrams found for {name}.\n")
        print("Try running again with an increased search size.\n")
        print("Or, use a longer form of the name!\n")

