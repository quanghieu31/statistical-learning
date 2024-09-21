First steps after receiving data:
- Data types, variable types, shape, meanings (data dictionary)
- Missing data, imputation or removal or what
- Depending on situations, need to create new features? delete features? do it if we know for sure what we want given the research question, otherwise, below:
- Think of what kinds of testing analysis on data given the variable types (correlations between continous, check distribution of categorical variables, multicollinearity or not, normality testing, outlier)
    - Analyze each of all numerical, continous, and then interations, correlations, outliers, normality (unit, measure consistency)...
    - Analyze each of all categorical, binary, ordinal, cardial, and then distribution, skewness,...
    - Analyze the interactions between the outcome variable(s) and these above independent variables => with testing methods
    - Do variable selection/dimensionality reduction/variance ranking/LASSO... to pick the best predictors with good explanatory power given some kind of metrics/methods
- After then, finalize the transformations or conversions (encoding, normalize,...), dataframe memory use reduction, finalize questions