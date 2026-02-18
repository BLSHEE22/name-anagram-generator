from collections import Counter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from anagram.generator import generate_top_anagrams, finalize_scores, filter_valid_words, load_words, normalize, VALID_TWO_LETTER_WORDS
from readchar import readkey, key
import sys
import csv
import time

# load words (for determining search strategy)
words = load_words()

# ==============
# --- Config ---
# ==============
BEAM_SIZE = 102
SEARCH_TIME_LIMIT = 5000
RESULT_GROUP_MIN = 10
HUMAN_IN_THE_LOOP = False
ASK_USER_FOR_FAVORITE = False
JUST_MEASURING_SEARCH_DURATION = True

# ===================
# --- Font Colors ---
# ===================
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BOLD = '\033[1m'
RESET = '\033[0m'

# ======================================================
# --- Helper function to ensure anagrams are correct ---
# ======================================================
def validate(name, anagram, debug=False, analytics=False, test_dicts=False):
    sorted_a = sorted([b.lower() for b in anagram if b.isalpha()])
    sorted_name = sorted([c.lower() for c in name if c.isalpha()])
    #print(Counter(normalize(name)))

    ### DATA ANALYTICS
    if analytics:
        a_normal = "".join([a.lower() if a.isalpha() else ' ' for a in anagram])
        #print(a_normal)
        a_split = [a for a in a_normal.split() if a.isalpha()]
        #print(a_split)
        #print(f"Name: {name}")
        #print(f"Anagram: {anagram}")
        longest_anagram_word_length = max([len(a) for a in a_split])
        longest_ag_word_lengths[name] = longest_anagram_word_length
        #print(f"Longest word length in anagram: {longest_anagram_word_length}")
        #print(longest_anagram_word_length)
        # quick search to find out how many searches need to be made
        valid_word_dict = filter_valid_words(name, words, Counter(normalize(name)), len(name))
        #print(valid_word_dict)
        word_ct_dict = {key: len(value) for key, value in valid_word_dict.items()}
        #print(sum(word_ct_dict.values()))
        #print(word_ct_dict[longest_anagram_word_length])
        # scrabble = {
        #         'a':1, 'b':3, 'c':3, 'd':2, 'e':1, 'f':4, 'g':2,
        #         'h':4, 'i':1, 'j':8, 'k':5, 'l':1, 'm':3, 'n':1, 'o':1, 'p':3,
        #         'q':10, 'r':1, 's':1, 't':1, 'u':1, 'v':4,
        #         'w':4, 'x':8, 'y':4, 'z':10
        #         }
        #print(sum([scrabble[c] for c in a_normal.replace(" ", "")]))

    # check if anagram is legal
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
    
    # check that dictionaries contain the proper words to be able form given anagram
    if test_dicts:
        anagram = anagram.replace("-", " ")
        a_split = anagram.split()
        new_a = []
        for word in a_split:
            word = "".join([c.lower() for c in word if c.isalpha()])
            if word != "":
                new_a.append(word)
        for word in new_a:
            if len(word) == 2 and word not in VALID_TWO_LETTER_WORDS:
                print(RED + f"Word '{word}' not in valid two-letter words list." + RESET)
            if word not in words:
                print(RED + f"Word '{word}' not in dictionary." + RESET)
                raise Exception

# ===============================================
# --- Graph number of words inside given name ---
# ===============================================

def draw_histogram(data):
    # set chart size
    max_count = max(count for _, count in data)
    height = 20  # number of rows for the tallest bar
    col_width = 6

    # scale counts to fit height
    scaled = [(length, int(count / max_count * height)) for length, count in data]
    print("\nNumber of Words in Name By Word Length Descending\n")

    # print bar labels
    print("".join(str(count).center(col_width) for _, count in data))
    print()

    # build rows top-down
    for level in range(height, 0, -1):
        row = ""
        for _, value in scaled:
            if value >= level:
                row += " █ ".center(col_width)
            else:
                row += "   ".center(col_width)
        print(row)

    # print separator
    print("―" * (len(data) * col_width))

    # print labels
    print("".join(str(length).center(col_width) for length, _ in scaled))
    print("\n       Word Length In Letters       \n\n\n")
    


# load model
model_path = "./anagram_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# test anagrams
longest_ag_word_lengths = dict()
# speed mode only
names_to_test = ["John Swanson"]
with open("data/raw/anagrams.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader, 1):
        name = row["input"]
        anagram = row["output"] 
        if name in names_to_test:
            validate(name, anagram, analytics=True, test_dicts=True)

# run
# print(YELLOW + "\nWelcome to the Name Anagram Generator!\n" + RESET)
# full_name = input("Enter a name below.\n")
names = []
for name in names_to_test:
    #print(name)
    if name in longest_ag_word_lengths.keys():
        names.append((name, longest_ag_word_lengths[name]))
print()

# support working multiple names sequentially
for name, longest_word in names:

    # anagram storage
    all_anagrams = []
    top_anagrams_dict = dict()
    best_anagrams = []

    # quick search to find out how many searches need to be made
    valid_word_dict = filter_valid_words(name, words, Counter(normalize(name)), len(name))
    search_strategy = [(key, len(value)) for key, value in valid_word_dict.items()]

    # create visual aid
    if not JUST_MEASURING_SEARCH_DURATION:
        draw_histogram(search_strategy)

    # speed mode only
    #search_strategy = [(6, 1)]
    search_strategy = [(longest_word, 1)]
  
    # find anagrams using initial word lengths of various sizes (name_length, 0.75*name_length, etc.)
    for search_pair in search_strategy[:min(len(search_strategy), 10)]:
        first_word_length = search_pair[0]
        # stop search if the minimum desired non-empty result groups exist
        if len({key: value for key, value in top_anagrams_dict.items() if value}.keys()) >= RESULT_GROUP_MIN:
            print(f"Breaking loop because we have at least {RESULT_GROUP_MIN} non-empty groups!")
            print(top_anagrams_dict)
            break
        if not JUST_MEASURING_SEARCH_DURATION:
            sys.stdout.write(f"\rSearching for anagrams for {name} with a starting word of length {first_word_length:2d}... ")
            sys.stdout.flush()
        # start timer
        start = time.perf_counter()
        top_anagrams = generate_top_anagrams(name, model, tokenizer, time_limit=SEARCH_TIME_LIMIT, top_n=10, beam=BEAM_SIZE, limit=200, 
                                             baseline_words=5, first_word_length=first_word_length)
        
        # filter out already seen anagrams
        #print(f"All anagrams so far: {all_anagrams}")
        unique_phrases_this_round = set([frozenset(a.split()) for a in top_anagrams])
        #print(f"All unique phrases this round: {unique_phrases_this_round}")
        top_anagrams = [a.title() for a in top_anagrams if frozenset(a.split()) not in all_anagrams]
        #print(f"Top anagrams that have not yet been seen: {top_anagrams}")

        # skip failed searches
        if all("[" not in a for a in top_anagrams):
            if top_anagrams:
                top_angrams_condensed = top_anagrams.copy()
                top_anagrams_condensed = [' '.join(c).title() for c in {frozenset(b.split()) for b in (a.lower() for a in top_anagrams)}]
                #print(f"Top anagrams condensed: {top_anagrams_condensed}")
                while len(top_anagrams_condensed) > len(unique_phrases_this_round):
                    #print(RED + "NEED TO CONDENSE FURTHER" + RESET)
                    #print(f"\nTop anagrams condensed: {top_anagrams_condensed}, length {len(top_anagrams_condensed)}.")
                    #print(f"Unique phrases this round: {unique_phrases_this_round}, length {len(unique_phrases_this_round)}.")
                    top_anagrams_condensed = [' '.join(c).title() for c in {frozenset(b.split()) for b in (a.lower() for a in top_anagrams)}]
                if not JUST_MEASURING_SEARCH_DURATION:
                    sys.stdout.write(f"{len(top_anagrams_condensed):3d} found, ")
                    sys.stdout.flush()
            else:
                if not JUST_MEASURING_SEARCH_DURATION:
                    sys.stdout.write(f"{0:3d} found, ")
                    sys.stdout.flush()
                top_anagrams_condensed = []
            top_anagrams_formatted = []
            for i, a in enumerate(top_anagrams_condensed):
                validate(name, a)
                all_anagrams.append(frozenset([b.lower() for b in a.split()]))
                top_anagrams_formatted.append(a)
                #sys.stdout.write(f"\'{a}\'" + ", ")
                #sys.stdout.flush()
                # optionally skip user checking
                if not HUMAN_IN_THE_LOOP:
                    best_anagrams.append(a)
            top_anagrams_dict[first_word_length] = top_anagrams_formatted
        else:
            if not JUST_MEASURING_SEARCH_DURATION:
                sys.stdout.write(f"{0:3d} found, ")
                sys.stdout.flush()
        # end timer
        end = time.perf_counter()
        elapsed = end - start
        if JUST_MEASURING_SEARCH_DURATION:
            print(f"{elapsed:3.2f}")
        else:
            sys.stdout.write(f"took {elapsed:5.2f} seconds.\n")
            sys.stdout.flush()
            #sys.stdout.write("\r\033[K")


    # allows user to reorder anagrams
    if HUMAN_IN_THE_LOOP:
        for k, v in top_anagrams_dict.items():
            print(BOLD + f"{k}-Letter Starting Word Anagrams: \n" + RESET)
            for a in v:
                print(a)
            print()
        i = 1
        initial_word_lengths = sorted(top_anagrams_dict.keys(), reverse=True)
        print(f"\n{sum([len(l) for l in top_anagrams_dict.values()])} total!\n")
        for n in initial_word_lengths:
            if top_anagrams_dict[n]:
                # don't move the cursor back for the first set
                lines_back = 7
                if n == max(top_anagrams_dict.keys()):
                    lines_back = 1
                    print()
                print(RED + f"\r\033[{lines_back}A\n##############################################################################\n" +
                    f"\n Anagrams for {name} with initial word length of {n}...\n" + 
                    "\n##############################################################################\n" + RESET)
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


    # re-run model to rank user selected anagrams
    if not JUST_MEASURING_SEARCH_DURATION:
        if best_anagrams:
            print(f"\nHere's how the computer ranks the best anagrams you selected for: {name}\n")
            scored_final_anagrams = sorted(finalize_scores(name, best_anagrams, model, tokenizer, batch_size=max(len(best_anagrams), 5), beautify=True), key=lambda x: x[1], reverse=True)
            for i, aPair in enumerate(scored_final_anagrams, 1):
                a = aPair[0]
                aScore = aPair[1]
                print(f"{i:2d}. {a} (" + str(round(aScore, 2)) + ")")
            # don't ask user for favorite anagram if only one was selected
            if len(best_anagrams) == 1:
                fav_ans = "1"
            else:
                if ASK_USER_FOR_FAVORITE:
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
                else:
                    print("\nSkipping asking user for favorite anagram...")
                    fav_ans = 1
            print(f"\nWinning anagram for: {name}\n" + BOLD + GREEN + scored_final_anagrams[int(fav_ans)-1][0] + "\n" + RESET)
            print("Hope you enjoyed using the Name Anagram Generator!\n")
        else:
            print(f"\nHmm...looks like there weren't any acceptable anagrams for {name}.\n")
            print("Make sure you are entering a name with 4 or more letters.\n")
            print("If you are, try running again with an increased beam size.\n")
            print("Or, throw in a middle name!\n")

