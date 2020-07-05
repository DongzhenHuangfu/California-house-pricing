# California House Pricing - Multiple Regression

In this project, I will try to model the "California House Pricing" data from scikit-learn. By doing the multiple regression, the house price will be predicted. Features will be selected by SFFS (Sequential Floating Forward Selection), and the Monte-Carlo Cross validation method will be implemented for model valication.

PS: No easy libraries, I will try to realize the whole process myself.

Let's do it!

## Overview of the Dataset

| | MedInc | HouseAge | AveRooms | AveBedrms | Population | Aveoccup | Latitude | Longitude | MedHousVal |
| ---- | ------ | -------- | -------- | --------- | ---------- | -------- | -------- | --------- | ---------- |
| count | 20640  | 20640    | 20640    | 20640     | 20640      | 20640    | 20640    | 20640     | 20640      |
| mean | 3.870671 | 28.639486 | 5.429000 | 1.096675  | 1425.476744| 3.070655 | 35.631861| -119.569704 | 2.068558 |
| std  | 1.899822 | 12.585558 | 2.474173 | 0.473911 | 1132.462122 | 0.386050 | 2.135952 | 2.003532 | 1.153956 |
| 25% | 2.563400 | 18.000000 | 4.440716 | 1.006079 | 787.000000 | 2.429741 | 33.930000 | -121.800000 | 1.196000 |
| 50% | 3.534800 | 29.000000 | 5.229129 | 1.048780 | 1166.000000 | 2.818116 | 34.260000 | -118.490000 | 1.797000 |
| 75% | 4.743250 | 37.000000 | 6.052381 | 1.099526 | 1725.000000 | 3.282261 | 37.710000 | -118.010000 | 2.647250 |
| max | 15.000100 | 52.000000 | 141.909091 | 34.066667 | 35682.000000 | 1243.333333 | 41.950000 | -114.310000 | 5.000010|

## Correlation between features

![Correlation between features](https://github.com/DongzhenHuangfu/California-house-pricing/blob/feature/presenting/figures/Correlation.png)

Seems that the feature MedInc is most linear relevant to MedHouseVal:

![Correlation between MedInc and MedHouseVal](https://github.com/DongzhenHuangfu/California-house-pricing/blob/feature/presenting/figures/Correlation_Inc_Val.png)

## Data isualization by position

Notice that there are two features called "Latitude" and Longitude, data can be visualized by their positions. For example: MedInc and MedHouseVal.

### 3D Figure - 2D Hotmap:
![HotMap of MedInc](https://github.com/DongzhenHuangfu/California-house-pricing/blob/feature/presenting/figures/Avl_MedInc.png)

![HotMap of MedHouseVal](https://github.com/DongzhenHuangfu/California-house-pricing/blob/feature/presenting/figures/Avl_MedHouseVal.png)

### 3D Figure:
![Distribution of MedHouseVal](https://github.com/DongzhenHuangfu/California-house-pricing/blob/feature/presenting/figures/3d_La_Lon_Val.png)

### 4D Figure - 3D HotMap:
![HotMap of MedHouseVal wrt. Latitude, Longitude and MedInc](https://github.com/DongzhenHuangfu/California-house-pricing/blob/feature/presenting/figures/4d_La_Lon_MedInc_MedHouseVal.png)

## Model training

### Data modeling
Data will be modeled as:

$$ f(x) = \sum_{\xi \in \Xi}\beta_{\xi}\xi(x) $$

where:

$$ \xi(x) = \prod_{i \in \\{1, 2, ..., 8\\}}x_{i}^{\alpha_{\xi_{i}}} $$

and:

$$ \Xi = \\{ \xi | \forall\alpha \in \mathbb{N}^{+8}: \sum_{i \in \\{1, 2, ..., 8\\}} \alpha_{\xi_{i}} \leq 4 \\} $$

In that case, the model can be treated as a multiple linear regression.

### Sequential Floating Forward Selection
Start from starts from the empty set. At each forward step includes a subset L of the whole set, which makes the evaluation score maximum. At each backward step, pick out a subset of the selected set, which makes the evaluation score maximum.

### Monte-Carlo Cross Validation 
Randomly select (without replacement) some fraction of your data to form the training set, and then assign the rest of the points to the test set. This process is then repeated multiple times, generating (at random) new training and test partitions each time.

### Multiple linear regression
Consider a multiple linear function in the form:

$$ f(\mathbf{x}) = \mathbf{w} * \mathbf{x} = \sum_{i=0}^{n}w_{i}x_{i} $$

Error will be calculated as:

$$ e = 0.5 * (y - f(\mathbf{x}))^{2} $$

where (\mathbf{x}, y) is the training data.

The gradient of the error is:

$$ \mathbf{gradient} = \mathbf{x} * (y - f(\mathbf{x})) $$

So the weights can be updated:

$$ \mathbf{w} = \mathbf{w} - \mathbf{gradient} $$

### Regularization
To avoid the case that one weight is too large (overfitting problem), Regularization method will be implemented. There, a L2 Regularization method will be used:

$$ L_2 = \lambda \sum_{i=1}^{n}w_{i}^{2} $$

Now, the error, or in pther word, the loss value will be:

$$ loss = 0.5 * (y - f(\mathbf{x}))^{2} + \lambda \sum_{i=1}^{n}w_{i}^{2} $$

The gradient will be:

$$ \mathbf{gradient} = \mathbf{x} * (y - f(\mathbf{x})) + 2 * \lambda * \mathbf{w} $$

### Data Standardization
The features here have different units, some values can be more than 1000 and the others are only between 0 and 1. This might cause too large weights for some features and too small weights for the others.

In this case, a standardization is helpful:

$$ \mathbf{x} = \frac{\mathbf{x} - \mathbf{mu}}{\mathbf{\sigma}} $$

where,

$$ \mathbf{mu} = mean(\mathbf{x}) $$

and

$$ \mathbf{\sigma} = max(\mathbf{x}) - min(\mathbf{x}) $$

or 

$$ \mathbf{\sigma} = std(\mathbf{x}) $$

## Result

Selected features: ['AveBedrms', 'AveRooms', 'MedInc', 'Latitude']

Error on Test data: 0.3384877315359696