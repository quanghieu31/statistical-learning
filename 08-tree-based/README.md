Gradient boosting and random forests (ensemble) models seem to be very popular

- Gradient boosting builds trees sequentially, where each tree corrects the errors of the previous one, optimizing for a loss function (residual errors iteratively)

    At each iteration $t$, add a new tree $h_t(x)$ to minimize the residuals: $$ h_t(x) = \arg\min_h \sum_{i=1}^n L\left(y_i, f_{t-1}(x_i) + \eta h(x_i)\right) $$

    The model updates predictions as: $$f_t(x) = f_{t-1}(x) + \eta h_t(x)$$

- Random forests build trees independently in parallel, using bootstrapped samples and random feature selection, then aggregate results like majority vote or average/bagging over trees/samples => avoid high varianc/sensitive from a single tree model
    - bootstrap data *randomly* and with replacements -> each sample has a tree
    - select only *random* subset of all features for training a tree

- Feature importance (in both): https://stats.stackexchange.com/questions/311488/summing-feature-importance-in-scikit-learn-for-a-set-of-features