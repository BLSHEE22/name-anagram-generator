# Projecting Anagram Search Time

Goal: Predict how long it will take to find ALL anagrams for a given name.

Approach: Predict search times at each available starting word length N and sum times.

At a single N, the deciding factors for run time are:
1. Number of N-length words in name
2. Number of letters remaining from the name after N are removed

Example:

```
name = "Claire Tobin"
N = 6
```

Let's take #1 from above and call it C, for the number of N-length candidate words.

- A "candidate word" is a word that can be made using the letters of the name.

In this case, it's 6-letter words, and there are 96.

$\Large C = 96$

Now let's take #2 from above and call it L, for the number of remaining letters in the name after N are removed.

There are 11 letters in Claire Tobin, so L is 5.

$\Large L = 5$
 
## Projection Function:

$\Huge S = 0.1C * 2^L$

Substituting in values...

$\Large S = (0.1 * 96) * 2^5$\
$\Large S = 9.6 * 32$\
$\Large S = 307.2$

These times per N can be summed to find the total search duration.




More Examples:

