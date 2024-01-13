- LogCosh is similar to MAE with the difference being that it is a softer version that can allow smoother convergence. It was adapted from [https://github.com/tuantle/regression-losses-pytorch](https://github.com/tuantle/regression-losses-pytorch)

```python
class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t, y_t):
        ey_t = (y_t - y_prime_t)/3 # divide by 3 to avoid numerical overflow in cosh
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))
```
- Note: adding the 1e-12 is not necessary, but it can act as a small regularizer [[adding epsilon to regularize]]
	