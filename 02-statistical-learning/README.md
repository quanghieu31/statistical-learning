## Estimate model $f(X)$, and how to do so

| Prediction     | Inference     |
| ------------- | ------------- |
| Desire to accurately predict outcome given the predictors | Desire to understand the role of the features and questions about relationships between variables |
| Aim to reduce average of the squared difference (expected value): <br> $\mathbb{E}(Y - \hat{Y})^2 = \mathbb{E}[f(X) + \epsilon - \hat{f}(X)]^2 = [f(X) - \hat{f}(X)]^2 + \text{Var}(\epsilon)$. <br> First term is reducible error and second is irreducible error (upper bound on prediction accuracy). | Need to know exact form of $f$ <br> Understand association between $Y$ and $X_1, \cdots, X_p$ |
| Accuracy of $\hat{Y}$ as a prediction for $Y$ depends on <br> - *reducible error* (improved by using more appropriate technique) and <br> - *irreducible error* (aka $\epsilon$ which can "never" help predict $Y$ correctly using $X$, i.e. unobserved/unmeasured variables). | Answer these questions: <br> - which predictors are associated with the outcome (identify important ones) <br> - relationship between the outcome and each predictor <br> - these relationships are linear or more complicated |

Generic notations:
- $n$ data points/observations/examples = this collection/set is called training data/set
- $p$ number of features/predictors/regressors/independent variables
- $x_{ij}$ = value of the $j$-th feature for observation $i$-th; in which $i=1,2,\cdots,n$ and $j=1,2,\cdots,p$.
- $y_i$ = outcome/prediction/response variable for $i$-th observation
- training data is consisted of $\{ (x_1, y_1), (x_2,y_2), \cdots, (x_n, y_n) \}$ where $x_i=(x_{i1}, x_{i2}, \cdots, x_{ip})^T$

| Parametric methods (model-based) | Non-parametric methods |
| ------------------ | ---------------------- |
|Reduce the problem of estimating $f$ down to estimating a (small) set of params | No explicit assumption about the functional form of $f$ <br> => estimate of $f$ before estimating params |
| 1. Make an assumption about functinal form/shape of $f$ <br> Example: Assume $f$ is linear, $f(X) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p$ <br> Goal: estimate the coefficients to get a model | - Requires a lot of training data or observations <br> - Requires a selection of level of smoothness (?)|
| 2. Then, fit/train the model with given data <br> Example: Fit with OLS | Examples: <br> - KNN <br> - Decision trees <br> - Random forests <br> Kernel density estimation |
| Disadvantage: the assumption is inappropriate, <br> i.e. the model is too far from the true form of $f$ <br> think about how flexible the model can be but see if it is overfitting | Disadvantage: data hungry, computationally expensive, overfitting |


## Trade-off between model interpretability and prediction accuracy

![](/static/fig1.png)

Often, obtain more accurate predictions using a less flexible method (accounting for overfitting in highly flex ones).

## Supervised vs unsupervised

- Supervised: fit a model based on a set of predictors and the response, so as to accurately predict the response for future obs (prediction) or understand the relationships (inference).
- Unsupervised: no response variables -> seek to understand the current available variables' relationships with clustering/segmentation on multiple characteristics/variables. $p(p-1)/2$ pairs of variables for scatterplots are not viable, so automated clustering is nice
- Semi-supervised: use a method to incorporate $m$ obs for which response measurements are available as well as the $n-m$ obs for which they are not


## Assess model accuracy overview (quick)

- Quality of fit: commonly used $\text{MSE}=\frac{1}{n}\sum_{i=1}^n(y_i-\hat{f}(x_i))^2$. We want to know whether unseen test data can make the trained model give lower MSE or lower test MSE. No guarantee that lowest training MSE gives lowest test MSE. The degrees of freedom quantifies the flexibility of a curve (higher df = higher flex). As the flex increases, there is monotone decrease in training MSE and U-shape in test MSE -> maybe overfitting (less flex model might have likely yielded a smaller test MSE)
- In practice, one can usually compute the training MSE with relative ease, but estimating the test MSE is considerably more difcult because usually no test data are available. Cross-validation can be used to estimate the test MSE using the training data.
- Bias-variance trade-off: lower bias, high variance, higher overfit, more flexiable model. 
    - Expected test MSE is $
E \left( y_0 - \hat{f}(x_0) \right)^2 = \underbrace{\text{Var}(\hat{f}(x_0))}_{\text{variance of f}} + \underbrace{\left[ \text{Bias}(\hat{f}(x_0)) \right]^2}_{\text{squared bias of f}} + \underbrace{\text{Var}(\epsilon)}_{\text{variance of error}} $
    - This is the avg test MSE if repeatedly estimate $f$ using a large number of training sets (i.e. boostrapping or cross-valid) and tested each at $x_0$ (one singple observation/row). So, doing this for all observations (all $x_i$) and then averaging them is the overall expected test MSE.
    - To get this MSE down, needs both both variance and bias down. The variance of a statistical learning method refers to how much the estimated function $\hat{f}$ would change if we used a different training dataset; higher variance means the model is highly sensitive to the training data. Bias refers to the error introduced by approximating a complex real-life problem with a simpler model; higher bias means the model makes stronger assumptions that simplify the true relationship, often leading to systematic errors. Generally, more flexible methods have higher variance but lower bias, while less flexible methods have lower variance but higher bias.

- The classification examples

Training/test error rate = $ \frac{1}{n} \sum_{i=1}^{n} I(y_i \neq \hat{y}_i) $ in which $I$ is an indicator variable (=1 if different, =0 otherwise). 

Examples:

|Bayes classifier | KNN |
|--|--|
| - requires a known distribution which is hard <br> - assign each observation with predictor vector $x_0$ <br> to the most likely class $j$: $Pr(Y=j\|X=x_0)$ <br> example: if there is class $j=2$, $Pr(Y=2\|X=x_0)>0.5$, <br> then observation with $x_0$ belongs to class $j=2$ | - helps Bayes classifier by i.e. estimating the conditional distribution of $Y$ given $X$, <br> - then classify a given observation to the class with the highest *estimated* probability <br> - given a positive integer $K$ and a test observation $x_0$, <br>  the KNN classifier identifies the $K $ points in the training <br> data that are closest to $x_0$, represented by $N_0$. <br> - it then estimates the conditional probability for class $j$  <br> as the fraction of points in $N_0$ whose response values equal $j$:  $\Pr(Y = j \| X = x_0) = \frac{1}{K} \sum_{i \in N_0} I(y_i = j)$ <br> - finally, KNN classifies the test observation $x_0$ to the class with the largest probability.|
|error rate at $X=x_0$ is $1 - max_j Pr(Y=j\|X)$ <br> overall error is $1 - E(max_j Pr(Y=j\|X))$ | $K$ is a hyperparam. Higher $K$ (big cluster) = less flex = lower variance = higher linear = higher bias |
| | $K=1$ (more flex) means training error is 0 but test error may be high <br> - classification: training error rate declines but test error not<br>- regression: training error rate declines but test error is U-shape as $k$ increases (correlation between error rates and inverse $K$ or $1/K$ (?) because small $K$ is highly flex => $1/K$ is large|

