- https://github.com/seatgeek/thefuzz
- https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

- This package basically helps you compare different strings

Motivation for the package:
- a human intern can identify that these shows are the same:
```
Cirque du Soleil Zarkana New York
Cirque du Soleil-Zarkana
Cirque du Soleil: Zarkanna
Cirque Du Soleil - Zarkana Tickets 8/31/11 (New York)
Cirque Du Soleil - ZARKANA (Matinee) (New York)
Cirque du Soleil - New York
```
- the human will also identify that these are different
```
Cirque du Soleil Kooza New York
Cirque du Soleil: KA
Cirque du Soleil Zarkana Las Vegas
```
- but it's hard for computers!
#### Usecases
- tells you how similar two strings are ([[Levenshtein distance]])
```
fuzz.ratio("NEW YORK METS", "NEW YORK MEATS") ⇒ 96
```
- smarter similarity based on similarities of substrings:
```
fuzz.partial_ratio("YANKEES", "NEW YORK YANKEES") ⇒ 100
fuzz.partial_ratio("NEW YORK METS", "NEW YORK YANKEES") ⇒ 69
```
- **out of order token similarity** (via sort then compare)
```
fuzz.token_sort_ratio("New York Mets vs Atlanta Braves", "Atlanta Braves vs New York Mets") ⇒ 100
```
- token set (the above fails when the strings are of very different lengths)
	- Solution: split the tokens into two groups: intersection and remainder.
	- Then use those sets to build up a comparison string.
```
fuzz.token_set_ratio("mariners vs angels", "los angeles angels of anaheim at seattle mariners") ⇒ 90
```