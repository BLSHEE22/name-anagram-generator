# Name Anagram Generator

### Create anagrams out of your friends' names!

<br>

## INSTRUCTIONS

========================================================================

### Download requirements.

- pip3 install torch transformers datasets sentencepiece accelerate

========================================================================

### Add training data into data/raw if it does not already exist.

- anagrams.csv

========================================================================

### Train the model.

- python3 -m training.train

========================================================================

### Set up config.

> max number of words placed onto the stack in each pass 
BEAM_SIZE = 10

> duration (s) at which to stop searching and return results
SEARCH_TIME_LIMIT = 20

> number of found anagrams at which to stop searching in current group
MIN_ANAGRAMS = 5

> stop group search once MIN_ANAGRAMS results are found
STOP_GROUP_AFTER_MIN_FOUND = True

> number of non-empty groups found before stopping entire search
MIN_NON_EMPTY_GROUPS = 10

> allow for user editing/reordering of found anagrams
HUMAN_IN_THE_LOOP = True

> allow for use to decide favorite among found anagrams
ASK_USER_FOR_FAVORITE = True

> when enabled, skips all prints except for search duration (s)
JUST_MEASURING_SEARCH_DURATION = False

========================================================================

### Run!

- python3 test_anagram.py



