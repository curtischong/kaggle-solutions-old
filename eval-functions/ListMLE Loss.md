The original paper link is down :'(
range: 
### summary
- Listwise Maximum Likelihood Estimation.
- Use for [[learning to rank]]:
	- https://www.auai.org/uai2014/proceedings/individuals/164.pdf
	- ![[Pasted image 20240119184214.png]]
	- The above probability is defined according to PlackettLuce model.
- the loss is minimized when we maximize the probability of the observed ranking.
- more info: https://notesonai.com/ListNet+and+ListMLE
- nice github repo: https://github.com/allegro/allRank/tree/masterlink: 
- pytorch implmentation: https://github.com/Obs01ete/kaggle_latenciaga/blob/b6b55b16fd444d423946eab3d5c8b293f7dc8939/src/losses.py#L62
### pros

### Cons
- ListMLE cannot well capture the position importance, which is a key factor in ranking
	- https://www.auai.org/uai2014/proceedings/individuals/164.pdf
		- the paper mentions:
		- It views the ranking problem as a sequential learning process, with each step learning a subset of parameters which maximize the corresponding stepwise probability distribution. To solve this sequential multi-objective optimization problem, we propose to use linear scalarization strategy to transform it into a single-objective optimization problem, which is efficient for computation