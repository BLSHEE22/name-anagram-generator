# Projecting Anagram Search Time

Goal: Predict how long it will take to find all anagrams for a given name.

## Approach
Predict search times at each available starting word length N and sum the times.

At a single N, the deciding factors for run time are:
1. Number of N-length words in name
2. Number of letters remaining from the name after N are removed

Example:

```
name = "Claire Tobin"
N = 6
```

Let's use C to represent the number of N-length candidate words in the name.

In this case, it's 6-letter words, and there are 179.

$\large C = 179$

<br>

Now let's use L to represent the number of remaining letters in the name after N are removed.

There are 11 letters in Claire Tobin, so L is 5.

$\large L = 5$

<br>
 
Now that we have values for C and L, we can form our projection function.

$\large S_n = 0.1C * 2^L$

After plugging in values, we find that an anagram search based on an initial 6-letter word would take 572.8 seconds.

$\large S_6 = (0.1 * 179) * 2^5$\
$\large S_6 = 17.9 * 32$\
$\large S_6 = 572.8$

This process can be repeated at each initial word length to find the total search duration.

$\large S_{11} = (0.1 * 0) * 2^0 = 0$\
$\large S_{10} = (0.1 * 1) * 2^1 = 0.2$\
$\large S_9 = (0.1 * 5) * 2^2 = 4$\
$\large S_8 = (0.1 * 36) * 2^3 = 28.8$\
$\large S_7 = (0.1 * 73) * 2^4 = 116.8$\
$\large S_6 = (0.1 * 179) * 2^5 = 572.8$\
$\large S_5 = (0.1 * 228) * 2^6 = 1459.2$\
$\large S_4 = (0.1 * 231) * 2^7 = 2956.8$\
$\large S_3 = (0.1 * 120) * 2^8 = 3072$\
$\large S_2 = (0.1 * 17) * 2^9 = 870.4$\
$\large S_1 = (0.1 * 5) * 2^10 = 512$


$\large S = 9,593 \hspace{1mm}seconds \hspace{1mm}\approx \hspace{1mm}2\hspace{1mm} hours\hspace{1mm} and\hspace{1mm} 40 \hspace{1mm}minutes$



## Final Projection Function

Where $n$ represents the length of the name:

$\Huge S = \sum_{i=0}^{n-1}S_n-i = 0.1C * 2^i$

<br><br>


Happy projecting!

