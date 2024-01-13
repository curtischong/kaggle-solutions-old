$$L_\delta(a) = 
\begin{cases} 
\frac{1}{2}a^2 & \text{for } |a| \leq \delta, \\
\delta \cdot (|a| - \frac{1}{2}\delta) & \text{otherwise}.
\end{cases}$$
- Where $\delta$ is a tunable parameter
- Where a is the residual: $a = y - \hat{y}$


- It is the best of both [[MAELoss]] and [[MSELoss]]
- It's quadratic for small values of a, and linear for large values:
![[Pasted image 20240113111356.png]]
- Huber loss is green ($\delta=1$)
- Squared error loss is blue

It's a suitable alternative to [[MSELoss]] if your eval metric is [[Mean Rowwise Root Mean Squared Error]]