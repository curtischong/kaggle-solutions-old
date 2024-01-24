
link: 
range: 
### summary
-  It evaluates how one probability distribution aligns or diverges from a second, expected probability distribution.
- you can use it:
	- as a model loss
	- to determine feature drift from your training data and real data, but  [population stability index](https://arize.com/blog-course/population-stability-index-psi/) (PSI) is preferred (since it's symmetric)

L(y_pred, y_true) = y_true * (log y_true - log y_pred)


- I'm not sure if KLDivergence is the same as adding label smoothing
	- I do see people doing KLDivergence AND label smoothing together
- I think 

- [Is label smoothing equivalent to adding a KL divergence term or a cross entropy term?](https://stats.stackexchange.com/questions/521006/is-label-smoothing-equivalent-to-adding-a-kl-divergence-term-or-a-cross-entropy)

- https://leimao.github.io/blog/Label-Smoothing/
### pros

### Cons

