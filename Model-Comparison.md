# Model Comparison
The following sections present the training and evaluation of the Linear regression, CNN and Random Forest machine learning models. A comparison is provided, showcasing which models performed best, with additional insights derived from explainable AI techniques. These explanations help us interpret models decisions and performance differences.

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

### Detected relationships from the SHAP analysis and Counterfactual example:

Sunshine levels decrease with increasing *cloud cover*. However, high *global radiation* values reduce the negative cloud cover impact. The places where we have high radiation, the cloud cover is lower and the sunshine levels are higher.

In general, if the global radiation increases by 0.2-0.5, the sunshine levels rise by approximately 1.

If the cloud cover increases by 0.5-1.5, the sunshine levels decrease by approximately 1.

Random forest has detected the influence of *humidity* in cities such as Dresden, Budapest, Dusseldorf, Sonnblick, Kassel, Oslo. In general higher humidity leads to lower sunshine levels.

In the city of Kassel, the relationship between the *wind speed* and *global radiation* is detected, with higher valus of radiation are associated with lower wind speed.

Stockholm is the only city that is missing the global radiation feature. Most importance is placed on the *cloud_cover* and *temp_max*. The higher the maximum temperature, the more sunshine levels the city will have.


