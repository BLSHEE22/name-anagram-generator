from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from anagram.generator import generate_top_anagrams, score_anagrams_batch
from readchar import readkey, key
import sys
import time

# ===================
# --- Font Colors ---
# ===================
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BOLD = '\033[1m'
RESET = '\033[0m'
    
# ==============
# --- Config ---
# ==============
BEAM_SIZE = 6
SEARCH_TIME_LIMIT = 5
RESULT_GROUP_MIN = 10

# ======================================================
# --- Helper function to ensure anagrams are correct ---
# ======================================================
def validate(name, anagram, debug=False):
    sorted_a = sorted([b.lower() for b in anagram if b.isalpha()])
    sorted_name = sorted([c.lower() for c in name if c.isalpha()])
    valid_check = sorted_a == sorted_name
    if debug:
        print(f"Sorted anagram: {sorted_a}")
        print(f"Sorted name: {sorted_name}")
        print(f"Valid: {valid_check}")
    if not valid_check:
        print(RED + f"INVALID ANAGRAM: '{anagram}' FOR NAME '{name}'\n" + RESET)
        print(f"Name letters: {sorted_name}")
        print(f"Anagram letters: {sorted_a}")
        raise Exception

# load model
model_path = "./anagram_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# run
print(YELLOW + "\nWelcome to the Name Anagram Generator!\n" + RESET)
full_name = input("Enter a name below.\n")
names = []
names.append(full_name)
print()
for name in names:
    all_anagrams = []
    top_anagrams_dict = dict()
    best_anagrams = []
  
    # find all anagrams by starting word length descending
    for n in range(len(name), 3, -1):
        # stop search if the minimum desired result groups exist
        if len(top_anagrams_dict.keys()) >= RESULT_GROUP_MIN:
            break
        sys.stdout.write(f"\rSearching for anagrams... ")
        sys.stdout.flush()
        start = time.perf_counter()
        top_anagrams = generate_top_anagrams(name, model, tokenizer, time_limit=SEARCH_TIME_LIMIT, top_n=10, beam=BEAM_SIZE, limit=200, 
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
                #top_anagrams_condensed = [' '.join(sorted(b.split(), key=len, reverse=True)) for b in set([' '.join(a.split()) for a in top_anagrams])]
                top_anagrams_condensed = [' '.join(c).title() for c in {frozenset(b.split()) for b in (a.lower() for a in top_anagrams)}]
                #print(f"Top anagrams condensed: {top_anagrams_condensed}")
                while len(top_anagrams_condensed) > len(unique_phrases_this_round):
                    print(RED + "NEED TO CONDENSE FURTHER" + RESET)
                    print(f"\nTop anagrams condensed: {top_anagrams_condensed}, length {len(top_anagrams_condensed)}.")
                    print(f"Unique phrases this round: {unique_phrases_this_round}, length {len(unique_phrases_this_round)}.")
                    top_anagrams_condensed = [' '.join(c).title() for c in {frozenset(b.split()) for b in (a.lower() for a in top_anagrams)}]

                    # lower_a = [a.lower() for a in top_anagrams_condensed]
                    # print(f"\nAll lowercase: {lower_a}")
                    # broken_a = [a.split() for a in lower_a]
                    # print(f"Broken up anagrams: {broken_a}")
                    # no_dup_a = set(frozenset(a) for a in broken_a)
                    # print(f"Removed duplicate words and ensuing anagrams: {no_dup_a}")
                    # rejoined_a = [' '.join(a).title() for a in no_dup_a]
                    # print(f"Rejoined/retitled anagrams: {rejoined_a}\n")
                    # top_anagrams_condensed = rejoined_a
                    ###
                    #top_anagrams_condensed = [' '.join(sorted(b.split(), key=len, reverse=True)) for b in set([' '.join(a.split()) for a in top_anagrams_condensed])]
                sys.stdout.write(f"{len(top_anagrams_condensed):2d} found, ")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"{0:2d} found, ")
                sys.stdout.flush()
                top_anagrams_condensed = []
            top_anagrams_formatted = []
            for i, a in enumerate(top_anagrams_condensed):
                validate(name, a)
                all_anagrams.append(frozenset([b.lower() for b in a.split()]))
                top_anagrams_formatted.append(a)
            top_anagrams_dict[n] = top_anagrams_formatted
        else:
            sys.stdout.write(f"{0:2d} found, ")
            sys.stdout.flush()
        end = time.perf_counter()
        elapsed = end - start
        sys.stdout.write(f"took {elapsed:5.2f} seconds.\n")
        sys.stdout.flush()
        sys.stdout.write("\r\033[K")
        #sys.stdout.flush()

    # user quality check
    i = 1
    initial_word_lengths = sorted(top_anagrams_dict.keys(), reverse=True)
    for n in initial_word_lengths:
        if top_anagrams_dict[n]:
            # don't move the cursor back for the first set
            lines_back = 7
            if n == max(top_anagrams_dict.keys()):
                lines_back = 1
                print()
            print(RED + f"\r\033[{lines_back}A\n##############################################################################\n" +
                f"\n Anagrams for {name} with initial word length closest to {n}...\n" + 
                "\n##############################################################################\n" + RESET)
            #print(f"{len(top_anagrams_dict[n])} found!\n")
            for a in top_anagrams_dict[n]:
                deciding = True
                while deciding:
                    sys.stdout.write(f"\r\033[K{i}. \033[1m{a}\033[0m  👍? (↑/↓/→/←): ")
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
                        sys.stdout.write(GREEN + f"✔ Accepted" + RESET)
                        sys.stdout.flush()
                        time.sleep(0.33)
                        best_anagrams.append(a)
                        sys.stdout.write("\r\033[K")
                        sys.stdout.flush()
                        #sys.stdout.write("\r\033[K")
                    elif k == key.DOWN:
                        sys.stdout.write(RED + f"✘ Rejected" + RESET)
                        sys.stdout.flush()
                        time.sleep(0.33)
                        sys.stdout.write("\r\033[K")
                        sys.stdout.flush()
                        #sys.stdout.write("\r\033[K")
                    else:
                        sys.stdout.write(YELLOW + "Please enter a valid arrow key." + RESET)
                        sys.stdout.flush()
                        sys.stdout.write("\r\033[K")
                        sys.stdout.flush()
                        continue
                    deciding = False
            
                    # update overall anagram index
                    i += 1


    # re-run of model to rank user selected anagrams
    if best_anagrams:
        print(f"\nHere's how the computer ranks the best anagrams you selected for: {name}\n")
        scored_final_anagrams = sorted(score_anagrams_batch(name, best_anagrams, model, tokenizer, batch_size=max(len(best_anagrams), 5)), key=lambda x: x[1], reverse=True)
        for i, aPair in enumerate(scored_final_anagrams, 1):
            a = aPair[0]
            aScore = aPair[1]
            print(f"{i:2d}. {a} (" + str(round(aScore, 2)) + ")")
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
        print(f"\nHmm...looks like there weren't any acceptable anagrams for {name}.\n")
        print("Make sure you are entering a name with 4 or more letters.\n")
        print("If you are, try running again with an increased beam size.\n")
        print("Or, throw in a middle name!\n")

