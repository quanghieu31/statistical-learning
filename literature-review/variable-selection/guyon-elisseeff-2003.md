**An Introduction to Variable and Feature Selection** by Guyon and Elisseeff (https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf)

Variable and feature selection

# Check list
- **domain knowledge**: use it to create better features if you have it
- **feature commensurability**: normalize features if they aren't comparable
- **feature interdependence**: if suspected, expand your feature set with combinations or products of features
- **pruning variables**: if necessary (for cost, speed, or clarity), prune by creating disjunctive features or weighted sums
- **feature assessment**: use a variable ranking method to assess features, especially if the number is large
- **need for a predictor**: if you don’t need a predictor, stop here
- **data quality**: check for and remove outliers using top-ranked variables if your data may be "dirty"
- **first steps**: if unsure, start with a linear predictor and explore feature selection methods; compare and refine
- **exploration**: if resources and ideas allow, compare different feature selection methods and predictors
- **solution stability**: to improve performance or understanding, resample data and redo the analysis multiple times

## Determine
After cleaning, understanding the unique values, and label-encode, onehot-encode if possible, we need to know which variables are categorical in the form of ordinal or non-ordinal values or simply numerical, continuous, or binary.

- how to make sure that categorical data are not encoded with labels that are not ordinal (or simply we want them to be nominal)? one-hot encoded (if few columns), target encoding (replace with a mean computed from target variable, but overfitting), frequency encoding, embedding layers, feature hashing (apply a hash function)
- why do so? ordinal labels might be misleading because ordinal implies that one category is greater or lesser than another, which can affect the interpretations, while encoding non-ordinal categorical variables as ordinal can imply that some categories are inherently better or worse than others

# Variable ranking method

- **principle of the method**: variable ranking involves scoring input variables based on their relevance to the output. high scores indicate valuable variables. this method is a preprocessing step and is computationally efficient, as it only requires computing and sorting scores. even if not optimal, it is robust against overfitting and scales well with large data.
- **correlation criteria**: for continuous outcomes, the pearson correlation coefficient is used to measure linear relationships between variables and the target. ranking variables by the square of this coefficient (r(i)²) emphasizes linear fit quality. this can be extended to classification problems, linking to criteria like fisher’s criterion or the t-test.
- **single variable classifiers**: variables can be ranked based on their predictive power using a classifier built with a single variable. metrics like error rate, false positive/negative rates, and roc curves are used to evaluate predictive power.
- **information theoretic ranking criteria**: mutual information measures the dependency between each variable and the target. estimating mutual information can be challenging, particularly for continuous variables, due to the difficulty in estimating probability densities. discretization or non-parametric methods like parzen windows can be used to approximate these densities.

# Examples/assessment of selecting subsets of variables that together have good predictive power
as opposed to their individual power

## Can Presumably Redundant Variables Help Each Other?

key concept: variable redundancy and feature selection

**variable ranking**: this is a method used in feature selection to rank variables based on their importance to the model. however, this method can sometimes lead to selecting variables that are redundant (i.e., they don't add new information to the model).

**redundant subset**: if you pick a bunch of variables that give you the same information, you might end up with a bigger set of features than you need, and this might not improve your model's performance.

the experiment:

**figure 1 (a)**: imagine you've got two variables (let's call them `x1` and `x2`) and you’re trying to classify data points into two classes. in this setup, the variables are independently and identically distributed (i.i.d.), which means they’re drawn from the same distribution and don’t influence each other.

**scatter plot**: the data points are plotted based on these two variables, with the classes centered at coordinates (-1, -1) and (1, 1).

**observation**: the scatter plot suggests that each variable on its own might not clearly separate the classes.

the twist: combining variables improves separation

**figure 1 (b)**: now, what happens if you rotate the plot by 45 degrees? this rotation is a clever way of showing that combining these two variables can actually improve class separation.

**improved separation**: after rotation, the separation between the two classes along the x-axis is now better by a factor of √2. this improvement happens because the combination (or averaging) of these two i.i.d. variables reduces the noise (variance) in the data.

conclusion: redundant variables aren’t always redundant!

**why this matters**: even though these variables seem redundant on their own (since they are i.i.d.), combining them can still lead to better performance because of noise reduction. this means that adding variables, even if they seem redundant, can sometimes improve the model by making the class separation clearer.

## How Does Correlation Impact Variable Redundancy?

Correlation and redundancy in variables

**basic idea of correlation**: correlation measures the relationship between two variables. if two variables are highly correlated (positively or negatively), it often suggests that they carry similar information.

**redundancy**: when variables are highly correlated, they might seem redundant because one can often be predicted from the other.

Examples:

**figure 2.a**: high correlation along the class center line
- **setup**: imagine you have two variables, and the class centers (mean positions of each class) are along a line. the variables are highly correlated along this line.
- **result**: the class distributions are stretched along the class center line (high covariance along this line). because of this correlation, if you combine these two variables (e.g., by summing them), you don’t really gain any additional separation power between the classes compared to using just one variable.
- **takeaway**: when variables are perfectly correlated along the direction of the class separation, they are truly redundant. adding one doesn’t improve your model’s performance because they both carry the same information.

**figure 2.b**: high correlation perpendicular to the class center line
- **setup**: now, the class centers are still positioned similarly, but the correlation is perpendicular to the class center line.
- **result**: the covariance (spread) is high in the direction perpendicular to the line joining the class centers. in this case, combining the two variables significantly improves class separation. even though they are correlated, they add complementary information.
- **takeaway**: here, even though the variables are correlated, they are not redundant. in fact, their combination is more powerful in distinguishing between classes.

**intra-class covariance**: the spread of the class data within each class (how the data points are distributed within the same class). when this spread (covariance) is in a direction that doesn’t align with the class center line, combining variables can help improve the model’s performance.

**correlation doesn’t always mean redundancy**: just because two variables are correlated doesn’t mean they are redundant. if the correlation is aligned in a way that doesn’t contribute to class separation, then they’re redundant. if the correlation exists but in a way that provides new information (like in figure 2.b), then they’re not redundant and can be very useful when combined.

**what this means for variable selection**

**be careful with simple correlation-based selection**: if you only look at correlation when selecting variables, you might mistakenly throw away variables that, while correlated, could provide complementary information.

**use models that consider interactions**: when you have correlated variables, use models that can capture interactions (like decision trees, random forests, or even neural networks) to see if those correlations actually contribute useful information.

**test for redundancy by combining variables**: before dropping correlated variables, try combining them in ways that capture potential complementary information. for instance, consider their sum, difference, or even more complex transformations.

**summary**: correlation alone isn’t a clear indicator of redundancy. correlated variables might seem redundant, but depending on how the correlation aligns with the class structure, they could still provide valuable information. always check how variables interact with each other in the context of your specific problem before deciding to remove them based on correlation alone.


## Can a Variable that is Useless by Itself be Useful with Others

"One concern about multivariate methods is that they are prone to overfitting. The problem is aggravated when the number of variables to select from is large compared to the number of examples. It is tempting to use a variable ranking method to filter out the least promising variables before using a multivariate method. Still one may wonder whether one could potentially lose some valuable variables through that filtering process"

**can a variable that is useless by itself be useful with others?**

- **overfitting concern**: multivariate methods can overfit the data, especially if there are many variables compared to the number of examples. filtering out variables before applying these methods might be tempting but can lead to losing potentially valuable variables.

- **example 1: useless by itself but useful together**:
  - **scenario**: imagine two variables with identical covariance matrices, where each variable alone doesn’t provide useful separation between classes. however, when both variables are used together, they improve class separability.
  - **figure 3.a**: demonstrates that even though each variable might be “useless” by itself, combining them can lead to better performance.

- **example 2: both variables useless by themselves but useful together**:
  - **scenario**: consider a situation inspired by the xor problem where four gaussian clusters are placed at the corners of a square, and class labels are assigned based on the xor function. here, projections on individual axes offer no class separation, but the classes can be separated effectively in the two-dimensional space.
  - **figure 3.b**: shows that even if two variables don’t provide separation individually (e.g., due to overlapping class densities), using them together can reveal class separability.

**key takeaways**:
- **combination of variables**: variables that seem useless individually can provide significant performance improvements when used in combination with others.
- **complex data structures**: some problems (like xor) require multiple variables to capture complex relationships and separations that single variables cannot reveal.

this highlights the importance of considering the interaction between variables rather than relying solely on individual variable performance.
