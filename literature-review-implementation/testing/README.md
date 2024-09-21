How to choose appropriate statistical testing? It depends on a variety of situations and the nature of research questions and domain knowledge, but if we are given solely the variables and data points, at least we can do something like this:


Given a linear test, we have independent variable (X), dependent variable (y), trying to estimate the effect of X on y (not to mention other control variables)

- X is binary, 
    - y is categorical: Chi-squared
    - y is continuous:  
        - y is same experimental unit: Paired T-test (nonparametric: Wilcoxon signed-rank test)
        - y is independent: Independent T-test (nonparametric: Mann Whitney U)
        - y is internally normalized already: 1-sample T-test
- X is continous,
    - y is binary: Logistic regression
    - y is categorical: ?
    - y is continous: Pearson's regression (nonparametric Spearman's correlation)
- X is categorical 1-factor (i.e. no interactions)
    - y is categorical: Chi-squared
    - y is continuous: 
        - y is same unit: Repeated measures ANOVA (nonparametric: Friedman's ANOVA)
        - y is independent: 1-way ANOVA (nonparametric: Kruskal Wallis)
    - y is a vector of multiple-continuous variables: MANOVA
- X is categorical 2-factor (i.e. interactions between gender and schooling)
    - y is categorical: Loglinear analysis
    - y is continous: 2-way ANOVA factorial ANOVA
    - y is a vector of multiple-continuous variables: Factorial MANOVA
