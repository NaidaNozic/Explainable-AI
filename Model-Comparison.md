# Preprocessing steps
After loading and showing the dataset, we noticed that we need to convert the DATE column into a proper date format. We also renamed the city of De Bilt, represented as "DE_BILT", to "DEBILT" as each variable is named "[City]_[weather characteristic]" and it is easier to handle the data with streamlined city names. Afterwards, we checked the data types, especially if there are non-numeric variables in the dataset, which was only the case for DATE. There were no missing values found. We also checked the distribution of data to look for potential erroneous data, by using boxplots in particular. We found a minimum `cloud_cover` for STOCKHOLM of -99, which is obviously wrong, and decided to impute the mean value instead of changing it to missing as Ridge Regression cannot handle missing values properly and we wanted to preserve the data row. Moreover, a minimum `pressure` of -0.099 (= -99 hPa) in STOCKHOLM and 0.0003 (= 0.3 hPa) in TOURS are also impossible in the real word, so we imputed the according mean values again. A `cloud_cover` of 9 translates to "sky view obstructed" which could be the case, but woult interfere with our model with values from 0 to 8, so we imputed the mean value.

We then saved the cleaned dataset to a pickle file as the starting point for our analysis.

# Model Comparison
The following sections present the training and evaluation of the Linear regression, CNN and Random Forest machine learning models. A comparison is provided, showcasing which models performed best, with additional insights derived from explainable AI techniques. These explanations help us interpret models decisions and performance differences.

## Linear Regression

### Training and Evaluation
We started by using a standard multiple linear regression model, at first for one city, BASEL. We checked the feature coefficient values and Mean Squared Error (MSE) as well as R² values. This showed a relatively good prediction, but when looking at **multicollinearity**, most of the Variance Inflation Factors (VIF) were high, i.e., larger than 10, showing that the variance of the according coefficients is increased by their correlation with other variables.

As this problem persisted for the other cities, too, we decided to address multicollinearity by using **Lasso**. Lasso adds a penalty to the loss function to reduce coefficient values and sets coefficient values of strongly multicollinear features to zero. This helps us to mitigate multicollinearity and helps to make the model better interpretable. The downside it that in case of high VIF values, Lasso tends to randomly select one feature and ignores the others, which can lead to a loss of information.

Therefore, we applied **Ridge** which could not provide this automatic feature selection like Lasso, but it reduces coefficient values more efficiently by penalizing large coefficient values better than Lasso. When looking at the results, the table below shows its supremacy for our dataset:

| City       | Ridge CV MSE | Lasso CV MSE | Ridge MSE | Lasso MSE | Ridge R² | Lasso R² |
|------------|--------------|--------------|-----------|-----------|----------|----------|
| BASEL      | 1.904        | 3.476        | 1.796     | 3.423     | 0.907    | 0.822    |
| BUDAPEST   | 1.728        | 4.270        | 1.623     | 4.113     | 0.919    | 0.794    |
| DEBILT     | 1.362        | 4.093        | 1.281     | 3.784     | 0.918    | 0.757    |
| DRESDEN    | 2.068        | 4.383        | 2.250     | 4.817     | 0.890    | 0.764    |
| DUSSELDORF | 1.467        | 3.684        | 1.388     | 3.510     | 0.924    | 0.808    |
| HEATHROW   | 1.826        | 4.100        | 1.835     | 4.228     | 0.885    | 0.735    |
| KASSEL     | 2.691        | 6.053        | 2.734     | 6.145     | 0.846    | 0.653    |
| LJUBLJANA  | 1.238        | 3.258        | 1.342     | 3.035     | 0.934    | 0.852    |
| MAASTRICHT | 1.477        | 3.751        | 1.365     | 3.565     | 0.920    | 0.792    |
| MUENCHEN   | 1.370        | 3.594        | 1.253     | 3.842     | 0.943    | 0.825    |
| OSLO       | 6.356        | 8.602        | 4.999     | 7.626     | 0.783    | 0.668    |
| ROMA       | 1.808        | 4.365        | 1.607     | 4.004     | 0.900    | 0.751    |
| SONNBLICK  | 3.630        | 5.203        | 3.845     | 5.071     | 0.813    | 0.753    |
| STOCKHOLM  | 5.276        | 5.886        | 5.532     | 6.062     | 0.770    | 0.748    |

It results in a lower cross-validation measn squared error (**CV MSE**) than Lasso for all cities, so better performs on unseen data when doing multiple folds, in this case 5. For a specific, separate test set the results are the same. The **R²** values are higher for all cities when using Ridge, meaning that it better fits the data and better explains the variance in sunshine. As a consequence, we choose Ridge over Lasso.

### Feature Importance
For every city we have calculated the feature importances based on the reduction in mean squared error (MSE) and by using the SHAP values that explain each feature's contribution to individual predictions and overall model behavior.

The **Most important** features based on the reduction in MSE: `global_radiation`, `cloud_cover`.
(Exceptions: KASSEL and MUENCHEN, where 2nd most important feature is `temp_max`, for OSLO `temp_mean`, for SONNBLICK the most important feature is `cloud_cover`, 2nd is `global_radiation`.)

The **Least important** features based on the reduction in MSE: `precipitation`, `pressure`.
(Exceptions: DEBILT `wind_gust` and `pressure`, DRESDEN `precipitation` and `temp_mean`, LJUBLJANA `precipitation` and `humidity`, ROMA `humidity` and `pressure`, SONNBLICK `humidity` and `precipitation`.)

As for STOCKHOLM there are no GLOBAL_RADIATION data available, the linear regression model with Ridge used `temp_max` and `temp_min` as the two most important features. But this turned out to be problematic, linear regression (even when applying Lasso/Ridge) has problems to sufficiently predict sunshine duration for STOCKHOLM, resulting in the lowest R² value of all of the cities.

### Detected relationships from the SHAP analysis and Counterfactual example using Ridge:
The table below shows the SHAP feature importance values for each feature per city, sorted by the mean value of each feature.

| Feature           | BASEL | BUDAPEST | DEBILT | DRESDEN | DUSSELDORF | HEATHROW | LJUBLJANA | MAASTRICHT | MUENCHEN | OSLO   | ROMA   | SONNBLICK | STOCKHOLM | KASSEL  |
|-------------------|-------|----------|--------|---------|------------|----------|-----------|------------|----------|--------|--------|-----------|-----------|---------|
| global_radiation  | 1.96  | 2.48     | 2.41   | 2.18    | 2.31       | 2.53     | 2.48      | 2.09       | 2.32     | 2.29   | 2.36   | 1.36      | 2.33      |
| cloud_cover       | 1.55  | 1.54     | 1.41   | 1.49    | 1.44       | 1.22     | 1.48      | 1.52       | 1.37     | 1.73   | 1.52   | 2.21      | 2.27      |
| temp_min          | 1.33  | 0.31     | 0.63   | 0.32    | 0.46       | 0.87     | 0.15      | 1.21       | 1.36     | 0.08   | 0.07   | 3.41      | 1.82      |
| temp_max          | 1.08  | 0.03     | 0.67   | 0.53    | 0.38       | 0.61     | 0.12      | 0.03       | 1.67     | 0.51   | 0.35   | 0.41      | 3.78      | 2.25    |
| temp_mean         | 0.27  | 0.11     | 1.44   | 0.03    | 0.4        | 0.54     | 0.64      | 0.31       | 0.53     | 1.9    | 0.14   | 0.57      | 1.33      | 0.51    |
| wind_gust         | -     | -        | 0.05   | 0.23    | 0.35       | -        | -         | 0.05       | 0.25     | 0.11   | -      | -         | -         | 0.34    |
| wind_speed        | -     | -        | 0.2    | 0.26    | 0.39       | 0.03     | 0.14      | 0.13       | 0.07     | -      | -      | -         | -         | 0.33    |
| humidity          | 0.04  | 0.19     | 0.06   | 0.17    | 0.16       | 0.14     | 0.04      | 0.19       | 0.31     | 0.06   | 0.01   | 0.31      | -         | 0.5     |
| precipitation     | 0.04  | 0.02     | 0.07   | 0.04    | 0.02       | 0.03     | 0.05      | 0.01       | 0.02     | 0.02   | -      | 0.16      | 0.12      | 0.06    |
| pressure          | 0.01  | 0.01     | 0.01   | -       | 0.01       | 0.02     | 0.01      | 0.01       | 0.01     | 0.01   | 0.01   | -         | 0.03      | 0.04    |

The more cloudy the sky (higher *cloud cover*), the less sunshine there is, which makes sense. But a higher *global radiation* can dampen the negative effect of cloudy skies. For STOCKHOLM, `temp_max` and `temp_min` are the most important features, probably because of missing `global_radition` data.

Interestingly, in general, when using Ridge, `pressure` and `humidity` are the 2nd and 3rd most important features with the highest mean absolute coefficient value, and `cloud_cover` is only on 4th rank. Higher `pressure` leads to a longer sunshine duration (except for OSLO, notably), which is comprehensible, as it is associated with periods of fine weather. More `humidity` leads to lower sunshine duration in general, but not for BASEL, OSLO, and ROME.

But when using SHAP based on Ridge, `pressure` and `humidity` are among the least important to explain sunshine duration.

For counterfactual explanations, we took random samples out of the dataset for each city. For example, to increase sunshine duration by 1 hr from 8.52 hrs to 9.52 hrs in BUDAPEST, `global_radiation` must increase from 2.17 to 2.52. For a 1 hr shorter sunshine duration in DUSSELDORF, 2.66 hrs instead of 3.66 hrs, `cloud cover` must be 7.23 instead of 6.



## Random Forest

### Training and Evaluation

A Random Forest model is trained and evaluated for each city. It predicts sunshine duration
using an 80-20 train-test split. Hyperparameter tuning is done with GridSearchCV using 5-fold Cross-Validation. As a scoring metric in Cross-Validation we have used Mean Squared Error (MSE).

The table below includes the cities with their corresponding **CV MSE** (Cross Validation Mean Squared Error), **Test MSE** (Mean Squared Error from the test set), and **Test R²** values.

| City       | CV MSE | Test MSE | Test R² |
|------------|--------|----------|---------|
| BASEL      | 1.631  | 1.473    | 0.924   |
| BUDAPEST   | 1.540  | 1.405    | 0.930   |
| DEBILT     | 1.135  | 1.046    | 0.933   |
| DRESDEN    | 1.818  | 1.873    | 0.908   |
| DUSSELDORF | 1.126  | 1.023    | 0.944   |
| HEATHROW   | 1.483  | 1.355    | 0.915   |
| KASSEL     | 2.002  | 1.868    | 0.895   |
| LJUBLJANA  | 1.179  | 1.199    | 0.941   |
| MAASTRICHT | 1.343  | 1.212    | 0.929   |
| MUENCHEN   | 1.078  | 0.901    | 0.959   |
| OSLO       | 4.959  | 4.623    | 0.799   |
| ROMA       | 1.490  | 1.239    | 0.923   |
| SONNBLICK  | 2.799  | 3.073    | 0.850   |
| STOCKHOLM  | 4.964  | 5.469    | 0.773   |

Overall, the **best results** are obtained with the model for **MUENCHEN**, with the lowest Test MSE (0.901) and the highest Test R² (0.959).

The model for **STOCKHOLM** has the **worst results**, with the Test MSE of 5.469 and the Test R² of 0.773.

Advantages of using Random Forest:
- handles multicollinearity better than linear regression,  because it randomly selects a subset of features for each tree node, making it unlikely that two correlated features will be selected for the same model.
- perform well on large data sets with high dimensionality,
- less prone to overfitting, than single decision tree,
- can capture complex non-linear relationships,

Disadvantages of using Random Forest:
- computationally expensive for complex datasets.
- harder to interpret compared to a single decision tree

### Feature Importance

For every city we have calculated the feature importances based on the reduction in mean squared error (MSE) and by using the SHAP values that explain each feature's contribution to individual predictions and overall model behavior.

The **Most important** features based on the reduction in MSE: `global_radiation`, `cloud_cover`.

The **Least important** features based on the reduction in MSE: `precipitation`, `wind_gust`.

### Detected relationships from the SHAP analysis and Counterfactual example using Random Forest:

Sunshine levels decrease with increasing *cloud cover*. However, high *global radiation* values reduce the negative cloud cover impact. The places where we have high radiation, the cloud cover is lower and the sunshine levels are higher.

In general, if the global radiation increases by 0.2-0.5, the sunshine levels rise by approximately 1.

If the cloud cover increases by 0.5-1.5, the sunshine levels decrease by approximately 1.

Random forest has detected the influence of *humidity* in cities such as Dresden, Budapest, Dusseldorf, Sonnblick, Kassel, Oslo. In general higher humidity leads to lower sunshine levels.

In the city of Kassel, the relationship between the *wind speed* and *global radiation* is detected, with higher valus of radiation are associated with lower wind speed.

Stockholm is the only city that is missing the global radiation feature. Most importance is placed on the *cloud_cover* and *temp_max*. The higher the maximum temperature, the more sunshine levels the city will have.

## CNN

### Training and Evaluation

A CNN model is trained and evaluated for each city. It predicts sunshine duration using a 80-20 train-test split. The hyperparameters where choosen by manual tuning them. The Conv1D layers have 64 and 32 filters with a kernel size of 3 and the dense layer was set to contain 32 units.

The table below includes the cities with their corresponding **Train MSE** (Mean Squared Error from the training set), **Test MSE** (Mean Squared Error from the test set), and **Test R²** values.


| City       | Train MSE | Test MSE | Test R² |
|------------|--------|----------|---------|
| BASEL      | 1.520  | 1.405    | 0.927   |
| BUDAPEST   | 1.455  | 1.414    | 0.929   |
| DEBILT     | 1.482  | 1.508    | 0.903   |
| DRESDEN    | 1.702  | 1.925    | 0.906   |
| DUSSELDORF | 1.154  | 1.120    | 0.939   |
| HEATHROW   | 1.461  | 1.474    | 0.908   |
| KASSEL     | 1.943  | 2.177    | 0.877   |
| LJUBLJANA  | 1.297  | 1.380    | 0.933   |
| MAASTRICHT | 1.436  | 1.359    | 0.921   |
| MUENCHEN   | 1.073  | 0.970    | 0.956   |
| OSLO       | 5.683  | 5.029    | 0.781   |
| ROMA       | 1.612  | 1.498    | 0.907   |
| SONNBLICK  | 2.751  | 3.155    | 0.846   |
| STOCKHOLM  | 4.617  | 5.157    | 0.786   |

Overall, the **best results** are obtained with the model for **MUENCHEN**, with the lowest Test MSE (0.970) and the highest Test R² (0.956).

The model for **OSLO** and **STOCKHOLM** have the **worst results**, with the Test MSE of 5.029 and 5.157 and the Test R² of 0.781 and 0.786.

### Feature Importance

For every city we have calculated the feature importances by using the SHAP values that explain each feature's contribution to individual predictions and overall model behavior.

The **Most important** features based on the SHAP values: `global_radiation`, `cloud_cover`. (Exceptions: KASSEL, where `global_radiation` and `temp_max` are pretty equally important, MUENCHEN, where 2nd most important feature is `temp_max`. For SONNBLICK the most important feature is `cloud_cover`, 2nd is `global_radiation`. And for STOCKHOLM, the most important feature is `temp_max`, followed by `cloud_cover` and `global_radiation` data is missing.)

The **Least important** features based on the SHAP values: `pressure`, `precipitation`.

### Detected relationships from the SHAP analysis and Counterfactual explanations using CNN:

As described previously, *cloud_cover* and *global_radiation* contribute the most to the sunshine level of a city.

For the counterfactual explanations we used 3 random samples for each city and checked how one of the most important features would have to change in order to gain 1h ouf sunshine duration. For example, to get from 4.64h on sunshine duration to 3.64h in DUSSELDORF, the *cloud_cover* would have to change from 3.0 to 4.13, with the other features beeing the same. Also, to get from 1.17h to 2.17h again in DUSSELDORF, the *global_radiation* would have to change from 0.43 to 1.0.