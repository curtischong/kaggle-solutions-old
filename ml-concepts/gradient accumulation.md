- simulate a larger batch size by accumulating gradients from multiple small batches before performing a weight update
	- this helps if you cannot fit the entire batch in GPU memory

```python
optimizer.zero_grad()

for b in range(batch_size):
        r = batch[r] 
        loss = net(r)  # forward one subgraph 
        scaler.scale(loss).backward() #backward accumuate gradient

scaler.step(optimizer) #update net parameters
scaler.update()
```