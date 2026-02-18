# Projecting Anagram Search Time

Goal: Project how long it would take to find all anagrams for a given name.

## Approach
Project search times at each available starting word length N and sum the times.

At a single N, the deciding factors for run time are:
1. Number of N-length words in name
2. Number of letters remaining from the name after N are removed

Example:

```
name = "Claire Tobin"
N = 6
```

Let's use $\large C_n$ to represent the number of N-length candidate words in the name.

In this case, it's 6-letter words, and there are 179.

$\large C_6 = 179$

Now let's use L to represent the number of remaining letters in the name after N are removed.

There are 11 letters in Claire Tobin, so L is 5.

$\large L = 5$

<br>

## Projection Function
 
Now that we have values for C and L, we can use our projection function to determine $\large S_n$, the number of seconds needed to search for all anagrams of the name with an initial N-letter word.

$\large S_n = 0.1C_n * 2^L$

After plugging in values, we find that searching for all anagrams of Claire Tobin with a 6-letter initial word would take 572.8 seconds.

$\large S_6 = (0.1 * 179) * 2^5$\
$\large S_6 = 17.9 * 32$\
$\large S_6 = 572.8$

This process can be repeated for all valid N to find the total search duration.

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
$\large S_1 = (0.1 * 5) * 2^{10} = 512$


$\large S = 9,593$
<br><br>
That's 2 hours and 40 minutes!
<br><br>


## Final Projection Function

Where $n$ represents the length of the name:

$\Huge S = \sum_{i=0}^{n-1} 0.1C_{n-i} * 2^i$

<br>

Happy projecting!

