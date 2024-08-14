### 1. Assess performance of flexible and inflexible method

Remember: Flex works better under high var, low bias cases. Inflex works better under low var, high bias.

|Case |Flexible method | Inflexible method |
|-|-|-|
|*Extremely large sample size $n$, few predictors $p$*|  | - Large $n$ = law of large numbers ensures sample averages converge to population parameters, reducing variability in estimates<br>- Few predictors $p$ = fewer params to fit to the data. Inflex less likely to capture noise or irrelevant patterns when $p$ is small (lower variance)-> better generalization<br>- Inflex typically have higher bias but bias here is less of a concern because of large $n$|
|*Extremely large $p$, small $n$* | Both methods will struggle without further model selection techniques or feature engineering like regulaziation, dimensionality reduction, or feature selection with ensemble methods. <br><br>- Small $n$ means high bias, large $p$ means more params to fit to the data, more noise = more variance, hard to generalize <br> - Flex methods with high complexity (high var, low bias) can overfit the data due to small $n$ due to high variance/more noise/ensitivity to data change | - Inflex methods with lower complexity (low var, high bias), less likely to overfit, but cannot capture the relationships to make accurate predictions and thus higher bias|
|*Highly non-linear relationship bewteen the predictors and response* | Non-linear means complex relationship => flexible/complex methods can capture this but are exposed to overfitting if small $n$ or much noise | Inflex is likely to fail to capture the complex relationships, leading to underfitting and poor accuracy prediction |
|*Variance of error terms is extremely high* <br> = large amount of variability in $y$ is not explained by $X$ | In the MSE, we need both variance and bias of the model to low <br><br> Flex methods might capture the complex relationships but there might be too much noise introduced by the high variance of error term => overfitting and poor generalization | Less affected by high variance of error terms because inflex can give low variance (smooth, less sensitive, but still high bias) predictions. But fail to capture complex relationships, so maybe underfitting. So, holding everything else constant, inflex might do better. 

### 2. Identify components of a problem

(a) We collect a set of data on the top 500 firms in the US. For each firm we record profit, number of employees, industry and the CEO salary. We are interested in understanding which factors affect CEO salary.
- Regression: assess CEO salary
- Inference: see the relationship between $y$ and the other features
- $n$ is 500 for 500 rows of observations on top 500 firms, $p$ is 3. 

(b) We are considering launching a new product and wish to know whether it will be a success or a failure. We collect data on 20 similar products that were previously launched. For each product we have recorded whether it was a success or failure, price charged for the product, marketing budget, competition price, and ten other variables
- Classification: predict success/failure
- Prediction
- $n$ is 20 and $p$ is 13

(c) We are interested in predicting the % change in the USD/Euro exchange rate in relation to the weekly changes in the world stock markets. Hence we collect weekly data for all of 2012. For each week we record the % change in the USD/Euro, the % change in the US market, the % change in the British market, and the % change in the German market.
- Regression: % change in USD/Euro exchange rate
- Prediction
- $n$ is 52 weeks and $p$ is 3.

### 3. Bias-variance tradeoff