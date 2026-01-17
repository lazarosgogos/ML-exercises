# About
This repository contains exercises, provided as Python notebooks, from the ML course of the DWS MSc at Aristotle University of Thessaloniki.

In addition to the individual ML exercises, the repository includes the final project description below. It provides a complete walkthrough of the Python notebook workflow used to complete the competition hosted at [drivendata.org](https://www.drivendata.org/competitions/305/competition-worldbank-poverty/page/963/).

# == FINAL PROJECT - POVERTY PREDICTION CHALLENGE ==

## Q1 - Dataset preprocessing

### What changes are to be done to the dataset
Treat each column type differently by filling missing values:
* Binary columns are set to 0, if the value is missing
* Nominal categorical codes are unordered, use the mode (whatever appears most often)
* For counts/discrete quantities, use the median
* For continuous values

## Q2 - Dataset analysis

### Column information:
**General Household Information**

* hhid: A unique identifier for each household in the dataset.

* com: A unique identifier for individual members within a household.

* weight: The sampling weight assigned to the household for statistical representation.

* strata: The stratification variable used in the survey design.

* cons_ppp17: The target variable representing per-capita daily expenditure in Purchasing Power Parity (PPP) terms for 2017.

* utl_exp_ppp17: The total expenditure of the household on utilities, measured in 2017 PPP.

* male: Indicates whether the head of the household is male.

* hsize: The total number of members residing in the main household.

* num_children5: The count of children under the age of 5 in the household.

* num_children10: The count of children between the ages of 5 and 10 in the household.

* num_children18: The count of children and adolescents between the ages of 10 and 18 in the household.

* age: The age of the household head in years.

* num_adult_female: The number of female adults aged 18 to 69 in the household.

* num_adult_male: The number of male adults aged 18 to 69 in the household.

* num_elderly: The number of elderly individuals aged 70 or older in the household.

**Dwelling and Infrastructure**
* owner: Indicates whether the household owns their current dwelling.

* water: Indicates if the dwelling has access to a formal water supply system.

* toilet: Indicates the presence of toilet facilities within the dwelling.

* sewer: Indicates whether the toilet facilities are connected to a sewer system.

* elect: Indicates whether the dwelling has access to electricity.

* water_source: Specifies the primary source of drinking water for the household.

* sanitation_source: Identifies the main type of sanitation facility used by the household.

* dweltyp: Categorizes the physical type of the dwelling structure.

* urban: Indicates whether the household is located in an urban or rural area.

**Employment and Education**
* employed: Indicates whether the household head is currently employed.

* sworkershh: The share of adults in the household who are currently working.

* share_secondary: The proportion of adults in the household who have completed secondary education.

* educ_max: The highest education level attained by any member of the household.

* sfworkershh: The share of working adults in the household who are in formal employment.

* any_nonagric: Indicates if any household member works in a non-agricultural sector.

* sector1d: Specifies the employment sector of the household head using the 1-digit ISIC classification.

**Geographic Regions**
* region1 - region7: Binary indicators representing the specific geographic region where the household is located.

**Food Consumption Indicators**
* consumed100 - consumed5000: These binary features indicate whether the household consumed specific food items, including breads, various meats (beef, chicken, pork), grains (rice, corn, quinoa), vegetables, fruits, and dairy products.

**Poverty Benchmarks**
* _pline5 - _pline95: These represent various poverty lines corresponding to specific percentiles (from the 5th to the 95th) of the welfare distribution in the training data.

## Q3 - Applying 3 different ML algorithms

We have selected a Random Forest, an XGBoost, a KNeighbors and an SGD regressor, in order to compare various techniques in tabular data prediction.
* **Random Forest**: A Random Forest is a standard way of utilizing Decision Trees to train on tabular data and predict unknown values. It performs adequately, as shown in the table of results below, especially given its short training time (meaning we can iterate and find a better solution by tuning some hyperparameters quite fast).
* **XGBoost**: XGBoost trees are one of the most commonly used models, given they are highly efficient, accurate and able to handle large datasets, by utilizing parallel processing, regularization, and optimized algorithms to build strong models by sequentially combining many weak decision trees. This method proves to be the most effective in this specific task.
* **KNeighbors**: In order to make predictions, this method utilizes local interpolation of the targets associated to the nearest neighbors in the training set.
* **SGD**: The Stochastic Gradient Descent regressor utilizes a linear model under the hood, fitted  by minimizing a regularized empirical loss with SGD. While in different settings SGD is an invaluable technique, it does not appear to work well at all in this given task.

The most adequate model seems to be the XGBoost regressor, out of the ones mentioned above.

#### Table of results with accuracy of predictions


Metric   | Random Forest | XGBoost | KNeighbors |   SGD   | MLP | Custom NN | Description
:-------:|:-------------:|:-------:|:----------:|:-------:|:---:|:--:|:----------:
Accuracy |     63.13%    | 64.32%  |   20.02%   |    -    | 58% | - | higher is better
Competition metric| 8.48 | 8.95    |    19.83   |   844   |  -  | 545.70 | lower is better
ws-MAPE  |    0.08       |   0.07  |    0.18    |   2.32  |     | 1.00 | lower is better
wMAPE    |      0.14     |   0.29  |    0.40    |   63.55 |     | 45.61 | lower is better


#### Why is the SGD accuracy not reported?
A model can regress values with at most 100% accuracy. However, given the way that accuracy is calculated (R2 score), it can be arbitrarily bad in its predictions. In this case, the model produces bad results (< 0%), hence why they are not reported. In this case, the result is some negative number in the millions, which makes no sense intuitively, but it just states how badly the model regresses values.

### Key takeaways
As is empirically shown, the most potent models are the tree-based ones. The worst contender is the MLP/Neural Networks. No matter how they are trained, they do not perform well, nor can they outdo the tree-based models in tabular data regression. Even after 200+ epochs of training, they cannot outperform a Random Forest Regression, which would take less than 2 minutes (without requiring a GPU), in contrast to the Neural Network, which requires a GPU in order to be trained.

## Q4 - Position at DrivenData competition
The best attempt has achieved a position of 236 out of 1023 participants currently, with a score of 15525 on the competition's metric.

## Q5 - Discussion
Here we discuss what worked, what did not, which features the models rely on, and how predictions could be improved.

### What worked, and what did not
We evaluated models from several families (tree-based models, k-nearest neighbors, stochastic gradient descent, and multilayer perceptrons). As shown empirically in this study and consistently reported in the literature, tree-based models perform best for regression on tabular data, while neural network–based models perform worst for this type of task.

### Which features do tree-based models rely on most?
The tree-based models primarily base their predictions on the *Expenditure on utilities (PPP17)* feature (`utl_exp_ppp17`). This variable dominates the upper splits of the trees, after which deeper nodes further discriminate between samples to produce final predictions.

### How could predictive performance be improved?
One of the most promising approaches to reducing prediction error is feature engineering. Although the dataset already contains substantial information, incorporating domain expertise could lead to more informative features and significantly improved model performance.
