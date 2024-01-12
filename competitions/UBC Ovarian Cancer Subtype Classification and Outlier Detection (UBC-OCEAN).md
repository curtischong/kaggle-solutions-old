Link: https://www.kaggle.com/c/UBC-OCEAN
Problem Type: [[Image Classification]] [[Unknown Class Classification]] [[Multiple Instance Learning]]
Input: [[Whole Slide Images]] and [[Image Masks]]
Output: [[Class Name (Thresholded)]]
Eval Metric: [[Balanced Accuracy]]
##### Summary
Given a medical image scan, predict one of these [subtypes of ovarian cancer](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2592352/): `CC, EC, HGSC, LGSC, MC, Other`. The `Other` class is not present in the training set; identifying outliers is one of the challenges of this competition.

There are two different types of data: TMA and WSI (much larger)
- it appears that the test set had more TMA examples. Figuring out how to classify both accurately was important

Halfway through the competition, additional image mask data was given:
https://www.kaggle.com/competitions/UBC-OCEAN/discussion/455890
- minor data cleaning issues. the color of the masks has a mix of red/green/blue after the img was resized. Basically, treat nonzero values as masks
- Important: An area marked as a tumor doesn’t imply it’s the sole tumor area in the slide: https://www.kaggle.com/competitions/UBC-OCEAN/discussion/455890#2529071

##### Solution Links
- (1st) Used phikon to extract features, then submit an ensemble of chowder models ([[Multiple Instance Learning]])
	- https://www.kaggle.com/competitions/UBC-OCEAN/discussion/466455
	- They used https://huggingface.co/owkin/phikon, a foundational model for digital pathology
		- [Chowder](https://arxiv.org/pdf/1802.02212.pdf) - a Multiple Instance Learning (MIL) model - is still on par with more recent MIL models (e.g. [TransMIL](https://arxiv.org/abs/2106.00908), [DTFD-MIL](https://arxiv.org/abs/2203.12081))
		- using the PNG file format took a lot of ram and was slow.
			- (SVS, TIFF, NDPI) store data pyramidally to prevent the need for loading the entire images into RAM
		- to analyze the large [[Whole Slide Images]] via a neural net, they split regions containing tissue into smaller patches (e.g. 224 x 224 px or 512 x 512 px)
			- they also split Tissue Microarrays (TMA) into smaller patches
		- 1) They used [Phikon](https://huggingface.co/owkin/phikon) to generate features -  a 2D tensor with shape `(n_patches, 768)`
		- 2) They fed these features into Chowder, which outperformed DeepMIL, MeanPool, and DSMIL.
		- most of the TMA were used for evaluation and not for training.
		- The loss was the [Cross-Entropy (CE) loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).
	- Two strategies were implemented to mitigate class imbalance:
		1. [Weighted sampling](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler) to create balanced batches for training,
		2. Using class weights in [CE loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss).
		- Chowder can be quite sensitive to weight initialization. Instead of training a single Chowder model, we decided to train an ensemble of N=50 Chowder models. The Chowder models in the ensemble only differ by their initialization. We found the ensemble to be more stable during training - and more efficient - than a single Chowder model.
			- code
				- 
				  ```python
					class ModelEnsemble(nn.ModuleList):
					
						def __init__(self, models: List[nn.Module]) -> None:
							super().__init__(modules=models)
					
						def forward(self,
									x: torch.Tensor,
									mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
							"""Forward pass."""
							predictions, scores = [], []
							for model in self:
								logits_, scores_ = model(x, mask)
								predictions.append(logits_.unsqueeze(-1))
								scores.append(torch.mean(scores_, dim=1, keepdim=True).unsqueeze(-1))
							predictions = torch.cat(predictions, dim=2)
							scores = torch.cat(scores, dim=2)
							return predictions, scores
					
					
					chowder_models = [Chowder(**chowder_kwargs) for _ in range(50)]
					
					model = ModelEnsemble(chowder_models)
				 ```
			- chowder uses dropout!
		- We hypothesized that given the low number of TMA in the train set, learning would be more efficient if we standardized all images (WSI or TMA) to a 20x resolution
			- so on test time, they used a logistic regression (trained on features extracted from the thumbnail of train images using a pretrained (ImageNet) ResNet18.) 
				- I assume so they can scale TMA images properly
		- How they detected outliers:
			- They used a threshold on the [[entropy]] of predictions, calculated as H = -sum(p*log(p))
				- since no outliers were provided, they tuned the threshold (of H) based on public leaderboard scores
			- They use the variance of predictions (across models in an ensemble of Chowder) to identify outliers. (probably using a threshold on variance as well)
				- this combined with entropy led to a good way to identify outliers
		- Ensembling
			-  it was more efficient to select specific repetitions and folds rather than ensembling all the 3 x 5 x 50 models
			- their submission is the **average prediction of 65 Chowder models** trained on different data splits
				- NOT the same chowder model but initialized with different seeds
			- "We calibrated these models using a logistic regression on their internal validation set (CV), which appeared to yield a slight improvement on the public LB"
				- I assume they changed hyperparms
			- then they just selected the best 50 models (to perform the final average with) based on each model's error
		- what didn't work
			- using a sliding window of 30-80% overlap for TMA (cause they got notebook timeout)
			- they tried to determine if an image was a "normal" image by identifying the percentage of the image that was a tumor
				- this didn't help the lb
			- using [[Ray Tune]] to automatically optimize hyperparms. didn't work
		- How they confirmed that the model generalized well to scaled TMA as well as WSI
			- 1) they averaged Phikon features across all patches
			- 2) they applied [[UMAP dimension reduction]] on each training example to see where their features would cluster after dimension reduction
			- 3) Since the features of TMA were mixed in with the features of WSI, they proved their model generalized well to different dimensions
		- Data quality issues:
			- some images were not of the correct scale. They had to count how many pixels each red blood cell was (win width) to properly scale the training data
- (2nd) use a feature extractor (unspecified) to train [[Multiple Instance Learning]] models
	- https://www.kaggle.com/competitions/UBC-OCEAN/discussion/465410
	- They used external data, which also had more classes. but it didn't improve scores
		- cause there were significant differences from competition data
	- they used the standard way of:
		- 1) **Crop** an entire WSI into thousands of **patches**;
			- note: they turned each img into a dataset
		- 2) Use extractors to **extract the features**;
		- 3) Train the **MIL** models.
	- The way they calculate the x,y, coords could mean that the images have overlap
	- code
		```python
		wsi_width, wsi_height = self.wsi.width, self.wsi.height
		thu_width, thu_height = thumbnail.width, thumbnail.height
		h_r, w_r = wsi_height / thu_height, wsi_width / thu_width
		down_h, down_w = int(self.patch_size / h_r), int(self.patch_size / w_r)
		cors = [(x, y) for y in range(0, thu_height, down_h) for x in range(0, thu_width, down_w)]
		```
	- models
		- ABMIL
		- DSMIL
		- TransMIL
	- We collected some samples that did not belong to the original five classes as independent Other class, so we only selected the one with the highest score as the prediction result.
		- I think this means that their external data WAS important, cause it helped them predict the "other" class
			- "However, there is almost no significant improvement compared to not using the Other class."
	- To speed up training, they disregarded patches from large images (rather than the smaller TMA patches)
		- This also helps the class imbalance a bit cause TMA represents more of the test data
		- the R_ratio variable determines what percentage of patches to retain
	- implementation detaills for training: https://www.kaggle.com/code/zznznb/wsi-train
		- NystromAttention
			-  # pad so that sequence can be evenly divided into m landmarks - landmarks are specific to this type of attention: https://jaketae.github.io/study/nystrom-approximation/
- (3rd) Used Lunit-DINO to extract features, then train on CLAM. Relied heavily on external data sources to prevent overfitting
	- https://www.kaggle.com/competitions/UBC-OCEAN/discussion/465527
	- they generated some "Other" synthetic images by cropping small tiles that were marked as healthy or as stroma.
	- used a **pretrained model Lunit-DINO to extract smaller size features** that he could make a prediction on
		- used 16-bit. didn't see much negative impact
	- **I filtered the tiles containing tissue using the thumbnails and then cropped the tissue tiles using PyVips**
		- pyVips was the only library that fit within the kaggle resource limits
		- For the feature extraction step I'm using PyTorch Dataloader with num_workers = 4 and do the image loading and cropping of tiles with PyVips inside the Dataloader. This way the tiles get prepared asynchronously on multiply threads while the the feature extractor is doing its work. This approach is more memory hungry, but PyVips is very resourceful if you load the images in the "sequential" mode. Sequential mode allows you to read the image only top to bottom, so I always load all tiles in one row at once.
	- He finally trained on **[CLAM](https://github.com/mahmoodlab/CLAM)** (similar to [[Multiple Instance Learning]], but uses attention to weigh each tile)
		- Since he trained an "other" category, he had to update the instance-level loss function (I think he means the loss function for each crop of the scan), so if the tile didn't show cancer, it would show "other"
	- used dropout
	- used 5-fold CV
		- made sure that multiple images from the same patient were in each fold
			- this only seems ok if the same patient wasn't in the test fold. I assume he means that "all of patent's images" are in the same fold
				- cause if it was in the test fold, the model just needs to identify if it has seen that patent before
		- Later I excluded the data from the Harmanreh lab completely for validation which lead to much more reliable cross-validation scores. [[Identify poor data sources]]
	- 
- (4th) Use the mask to identify cancerous areas, and train on a small and resized 768x768 pixels (very efficient)
	- https://www.kaggle.com/competitions/UBC-OCEAN/discussion/465811
	- TMA images are centre cropped (eg. 3000 -> 2500) and resized to 768x768 pixels.
	- Since the WSI images are larger, he used the thumbnail images to:
		- 1) find cancerous locations (via an tumor mask detector he trained on the supplemental data)
		- 2) The location of the pixel with highest probability of being cancerous is selected on the WSI and a region of 1536x1536 pixels around it is cropped and resized to 768x768 pixels, which is then used to predict scores.
	- used sigmoid activation, rather than softmax in his ensemble
		- so each model trained was probably tailored to one specific class.
		- Median averaging is used for generating score.
	- His data was small enough to use these augmentations:
		- Stain augmentation, scaling, rotation, flipud, fliplr, random contrast, random brightness, and random hue (thought it might work as stain augmentation).
	- This is how they calculate the final probability
		- `p = (N_CLF*np.median(p,axis=0)+N_FPN*np.median(p1,axis=0))/(N_CLF+N_FPN)`
		- the average of the median of all the predictions
			- is the the median value of all tiles? Either way, it's nice to see a solution that isn't just "average all results"
		- interesting comment:
			- `if np.max(p1_[:,j]) > 0.05: # A single title is selected for each label above threshold.`
	- To predict the "other" class, he:
		- 1) finds the predicted class with the highest probability
		- 2) if this probability is < 10%. it's labelled as other.
- (5th) Did NOT use MIL
	- https://www.kaggle.com/competitions/UBC-OCEAN/discussion/466017
	- when doing inferencing, they used a tile selection model (only trained on WSI)
		- then a classification model (WSI and TMA)
	- WSI training:
		- 6-class classification (Hubmap external data as “Other”)
	
	- how they identified important tiles
		1) used a tile classification model (not used for inference)
			- WSI label as tile label
			- ConvNeXt-base
			- Random cropped tiles at 1536x1536, background excluded
			- Augmentations: Random horizontal and vertical flips, RandomRotation, RandAugment, RandomGrayscale, RandomErasing
		2) Segmentation helper model to identify tumors (not used for inference)
			- uses the supplementary dataset
	- After labelling, he created a heatmap of the img
		- where Heatmap Ground truth is 0.5 classification confidence + 0.5 tumor confidence
		- FINALLY, the trains a segmentation model SEResNeXt101 UNet to pass onto the classification model
	- classification model (for inference)
		- he had to train 2 rounds
		- How he labelled the tiles
			- 1. Predict all foreground tiles
			    1. Confidence <0.3 tiles are pseudo labeled “Other” in 2nd round of training
			    2. Confidence 0.3-0.6 tiles are ignored
			    3. Confidence >0.6 tiles are pseudo labeled as WSI label
			- 2. train 2nd round
	- TLDR: he had like 5 rounds of training
		- the advantage is that he was working with lower resolution imgs so it was probably faster
	- inferencing
		- for WSI images, only the top-5 confidence tiles are selected for classification
			- for TMS, no selection needed
	- stain normalization: [https://github.com/EIDOSLAB/torchstain](https://github.com/EIDOSLAB/torchstain)
#### Takeaways
- The first place solution was far ahead of others. I think using [[entropy]] to identify the "other" class was a differentiator
- The top three all used feature extractors, rather than just treating it as a plain img classification problem
- If you wanted to be efficient, you had to run one model to identify interesting tiles, then another model to classify that tile.
	- Reading (1st)'s paper: https://www.medrxiv.org/content/10.1101/2023.07.21.23292757v1
		- it doesn't seem like they did this efficiency step, which probably also explains their significantly higher score
- [[Multiple Instance Learning]] is pretty good!
- when you have different types of images: TMA and WSI, you'll have different image transformation procedures, but once they look similar, you can use one model to make decent predictions, no matter the type