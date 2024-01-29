### The goal:
- to identify ALL the techniques used by the best teams
- Why? Because knowing these tips helps you build intuition on what to do. It makes you better
### Why study Kaggle competitions?
- Cause people have to go through the entire ml process
- People need to be considerate of computational resources
- Many people have iterated on the same problem -> better ideas
- It's a competition, so people try a lot harder to figure out what works/doesn't
- They aren't motivated to only show positive results. People will talk about their failures as well
- People will post their code. So if there's something confusing, you can read the implementation to fully understand it
- Kaggle writeups are simpler to read than papers.
- Writeups will self-select important topics for you to learn more about, especially if you see it between competitions

### Why use this resource:
- I looked at each competition holistically, scouring discussions as if I were actually competing
- I verify nuances by looking at the specific implementations written on their github / jupyter solutions

### The best way to view this resource:
- 1) Clone this repo locally
- 2) Open this repo using [obsidian.md](https://obsidian.md/)
- 3) Download and enable these plugins:
	- Image Toolkit (so you can click on an image and it'll enlargen)
- 4) Turn on backlinks (so you can see the other pages that references the current page)
	1. Open the Command palette. (`cmd/ctrl + p`)
	2. Select `Backlinks: Toggle backlinks in document.`

### In general:
- things that work for some teams (models / features) may not work for every team
- but some features / techniques always works on a problem
- pretrained neural net embeddings are OP. just use them, even with small datasets
- hyperparam tuning won't save you. garbage features => bad predictions
- winning teams try things that aren't done in the public notebooks.
- trust your CV, not your ego. Don't get attached to implementations that *should* work
- The best performers typically work the smartest and put in the most work
	- They've found a magic trick that works well
	- Their iteration times are much shorter than yours
		- They have amazing software tools (weights and biases, optuna)
	- They have lots of computational resources

### How to review Kaggle problems efficiently
1) read the intro
2) sort all of the notebooks by most votes (not hottest). this will usually give you an indication of what to do / what the competition is
3) fill in the objective / input / output fields in the competition template
4) write down what you learn in the notebook in the popular notebooks
5) start reading solution notebooks
	1) make sure you read the comments section, because that is where ppl point out important techniques

### Finding worthwhile competitions to look at
- 1) look at competitions with different input types
- 2) look for competitions with a high leaderboard shake-up
	- Shakeup occurs when most people have a poor [[cross validation]], but some people have amazing CV
		- you can learn how to create a GOOD CV faster by looking at these competitions
		- you an also learn unconventional opinions that are against majority opinion
	- https://www.kaggle.com/code/jtrotman/meta-kaggle-competition-shake-up

If you want to search for winning solutions using a technique, search for:
"kaggle 2nd place solution conditional probability"
- and iterate the "2nd"

#### Obsidian tips
- cmd + shift + leftarrow to select entire line until the bullet (before deleting)
- here's how to resize images: `![[Pasted image 20240129154843.png|400]]`

#### Cool resources:
- https://farid.one/kaggle-solutions/ (has an exhaustive list of all competitions)
- https://www.kaggle.com/code/sudalairajkumar/winning-solutions-of-kaggle-competitions
	- this is good cause they have tags on the types of competitions
- https://notesonai.com/ a cool resource for general ML