
- for [[classification]] problems, rather than making your target variables binary (0 or 1), it smooths it, so it's 0.9
	- https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
	- If we do not use label smoothing, the label vector is the one-hot encoded vector [1, 0, 0]. **Our model will make _a_ ≫ _b_ and _a_ ≫ _c_**. For example, applying softmax to the logit vector [10, 0, 0] gives [0.9999, 0, 0] rounded to 4 decimal places.
	- If we use label smoothing with _α_ = 0.1, the smoothed label vector ≈ [0.9333, 0.0333, 0.0333]. The logit vector [3.3322, 0, 0] approximates the smoothed label vector to 4 decimal places after softmax, and it has a smaller gap. This is why we call label smoothing a regularization technique as **it restrains the largest logit from becoming much bigger than the rest.**
	- Label smoothing replaces one-hot encoded label vector _y_hot_ with a mixture of _y_hot_ and the uniform distribution:
	- _y_ls_ = (1 - _α_) * _y_hot_ + _α_ / _K_


I'm not sure how label smoothing is related to KLDivergence:
- https://leimao.github.io/blog/Label-Smoothing/
- https://proceedings.neurips.cc/paper_files/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf
	- is related to [[KLDivergenceLoss]]
	- "They show that label smoothing is equivalent to confidence penalty if the order of the KL divergence between uniform distributions and model’s outputs is reversed."
		- not sure what this means