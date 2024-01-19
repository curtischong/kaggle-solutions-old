https://paperswithcode.com/method/gelu
$$GELU(x) = 0.5x \left(1 + \tanh \left( \sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right)\right)$$
- Similar to [[ELU (Exponential Linear Unit)]].
	![[Pasted image 20240119143612.png]]
- It looks VERY similar to [[Swish]]
	- https://towardsdatascience.com/on-the-disparity-between-swish-and-gelu-1ddde902d64b
	- ![[Pasted image 20240119145024.png]]
		- Swish's bump (on the right) is a bit lower (and lasts longer)