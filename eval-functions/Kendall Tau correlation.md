link: https://datatab.net/tutorial/kendalls-tau
range: [-1, 1]
- **τ = 1** indicates a perfectly increasing relationship between the ranked variables.
	- (i.e. you got the rank perfect!)
- **τ = -1** indicates a perfectly decreasing relationship.
- **τ = 0** means there is no relationship
### summary
- This is a ranking metric: how well does your model-predicted ranking, correspond to the real ranking
- it's preferred over [[Spearman's correlation Coefficient]] since it's more robust
- τ = (number of concordant pairs - number of discordant pairs) / total number of pairs
	- - The pair is called **concordant** if the ranks for both X and Y increase together,
	- The pair is considered **discordant** if the rank for X increases and the rank for Y decreases.
- The best way to understand it is to go to this link and walk through the example:
	- https://datatab.net/tutorial/kendalls-tau
### pros

### Cons
