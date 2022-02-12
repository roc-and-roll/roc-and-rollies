# Calculate Dice Scores

## Previous Work

- Dice Calulator using TF
	- https://towardsdatascience.com/calculating-d-d-damage-with-tensorflow-88db84604f0a
	- https://github.com/sugi-chan/2-stage-dice-pipeline
- [Paper](https://digitalcommons.wku.edu/cgi/viewcontent.cgi?article=1004&context=seas_faculty_pubs)

## Questions

- will 6 vs 9 be a problem? 1/2 vs 12?
- icons instead of numbers (1/20)?
- what if two dice in one image?
- Object detection first?

## Dataset

- https://www.kaggle.com/ucffool/dice-d4-d6-d8-d10-d12-d20-images

### Prep

- Scouting
	- look into folders and look at color distribution of dice
	- how large are images?
	- All same size?
- Change image size?
- don't sort by dice but sort by number instead?
	- dice size doesn' matter
	- except for d4 - they suck
	- two stage? classify something as d4 and use different classifier?
- maybe balancing
- Augmentation:
	- Color Shift
	- Inversion
- Binarisation/To Grayscale
- Gap between input for training vs input in real-world scenario
	- close via patches

## Evaluation

- Confusion Matrix for validation data
- maybe more out-of-domain data
