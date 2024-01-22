link: 
range: [0, 1] (lesser the better)
### summary

- Used to evaluate how good your model solves [[machine translation]] or [[transcription]] problems
It's derived from the [[Levenshtein distance]]

$WER = \frac{S + D + I}{N} = \frac{S + D + I}{S + D+ C}$
- _S_ is the number of substitutions,
- _D_ is the number of deletions,
- _I_ is the number of insertions,
- _C_ is the number of correct words,
- _N_ is the number of words in the reference (N=S+D+C)
### pros

### Cons
