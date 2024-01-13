A form of [[target encoding]] but you leave out the current row's target in the calculation.
- e.g. if you are trying to calculate target encoding and using the mean of the y values, calculating leave-one-out instead would be like:
$$\frac{\text{(sum\_target - target\_for\_cur\_row)}}{n-1}$$