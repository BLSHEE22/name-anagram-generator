# Name Anagram Generator

### Create anagrams out of your friends' names!

![See 'runtime_projections.png' for more info.](ac_gif.gif)

## Setup

### Download requirements.

```pip3 install torch transformers datasets sentencepiece accelerate```
<br>

### Add training data into data/raw if it does not already exist.

- anagrams.csv

```
Alex Bregman,Lax Bargemen
Eric Musselman,Menu's Miracles
...
etc.
```
<br>

### Train the model.

```python3 -m training.train```
<br>

### Specify parameters.

- config.py

```
# max number of words placed onto the stack in each pass 
BEAM_SIZE = 6

# duration (s) at which to stop searching and return results
SEARCH_TIME_LIMIT = 20

# number of found anagrams at which to stop searching in current group
MIN_ANAGRAMS = 100

# stop group search once MIN_ANAGRAMS results are found
STOP_GROUP_AFTER_MIN_FOUND = True

# number of non-empty groups found before stopping entire search
MIN_NON_EMPTY_GROUPS = 5

# allow for user editing/reordering of found anagrams
HUMAN_IN_THE_LOOP = True

# allow for use to decide favorite among found anagrams
ASK_USER_FOR_FAVORITE = True

# when enabled, skips all prints except for search duration (s)
JUST_MEASURING_SEARCH_DURATION = False
```
<br>

### Run.

```python3 test_anagram.py```

