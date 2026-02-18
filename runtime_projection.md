# Projecting Anagram Search Time

Question: How long does it take to find ALL anagrams for a given name?

Hypothesis: It comes down to **2 key factors**.

1. Number of candidate words at the max-length anagram word.
2. Number of letters remaining from the name after max-length anagram word is removed.


Here's how we'll set up our projection function:

$L = Length of name$

$N = Max anagram word length$

$C = Number of candidate words at length N$

- A "candidate word" is a word that can be made using the letters of the name.

$l = Number of letters remaining from name after max anagram word length is removed$

$l = L - N$

Function
$0.1C * 2^l$ 


Examples:

