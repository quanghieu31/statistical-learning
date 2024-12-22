**An Introduction to Variable and Feature Selection** by Guyon and Elisseeff (https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf)

# Quick notes

- domain knowledge for feature engineering/creation/removal/transformation
- normalize features if they aren't comparable
- feature interdependence: if suspected, expand feature set with combinations or products of features
- pruning variables: if necessary (for cost, speed, or clarity), prune by creating disjunctive features or weighted sums
- feature assessment: use a variable ranking method to assess features, especially if the number is large
- always check for outliers and distribution/skewness and missing data points
- first steps: if unsure, start with a linear predictor and explore feature selection methods; compare and refine
- exploration: if resources and ideas allow, compare different feature selection methods and predictors
- solution stability: to improve performance or understanding, resample data and redo the analysis multiple times

# 1. Determine

After cleaning, understanding the variable types carefully 

- for categorical: unique values, and label-encode, onehot-encode if possible, we need to know which variables are categorical in the form of ordinal or non-ordinal values or just binary, or 
- for numerical, continuous (normalize, standardize if needed)

- how to make sure that categorical data are not encoded with labels that are not ordinal (or simply we want them to be nominal)? one-hot encoded (if few columns), target encoding (replace with a mean computed from target variable, but overfitting), frequency encoding, embedding layers, feature hashing (apply a hash function)
- why do so? ordinal labels might be misleading because ordinal implies that one category is greater or lesser than another, which can affect the interpretations, while encoding non-ordinal categorical variables as ordinal can imply that some categories are inherently better or worse than others

# 2. Variable ranking method

- **principle of the method**: variable ranking involves scoring input variables based on their relevance to the output. high scores indicate valuable variables. this method is a preprocessing step and is computationally efficient, as it only requires computing and sorting scores. even if not optimal, it is robust against overfitting and scales well with large data.
- **correlation criteria**: for continuous outcomes, the pearson correlation coefficient is used to measure linear relationships between variables and the target. ranking variables by the square of this coefficient (r(i)²) emphasizes linear fit quality. this can be extended to classification problems, linking to criteria like fisher’s criterion or the t-test.
- **single variable classifiers (not simply correlation, and for all conti and categ)**: variables can be ranked based on their predictive power using a classifier built with a single variable. metrics like error rate, false positive/negative rates, and roc curves are used to evaluate predictive power.
- **information theoretic ranking criteria**: mutual information measures the dependency between each variable and the target. estimating mutual information can be challenging, particularly for continuous variables, due to the difficulty in estimating probability densities. discretization or non-parametric methods like parzen windows can be used to approximate these densities.

# 3. Examples/assessment of selecting subsets of variables that together have good predictive power

interaction, as opposed to their individual power

## Can Presumably Redundant iid variables Help Each Other?

key concept: variable redundancy and feature selection

- **variable ranking**: this is a method used in feature selection to rank variables based on their importance to the model. however, this method can sometimes lead to selecting variables that are redundant (i.e., they don't add new information to the model) => remove multicollinearity first? but what if there are powerful interactions?
- **redundant subset**: if you pick a bunch of variables that give you the same information, you might end up with a bigger set of features than you need, and this might not improve your model's performance.

**Case 1**: imagine two variables (`x1` and `x2`) and we’re trying to classify data points into two classes. In this setup, the variables are iid, which means they’re drawn from the same distribution and also don’t influence each other.

- **scatter plot**: the data points are plotted based on these two variables, with the classes centered at coordinates (-1, -1) and (1, 1).

- **observation**: the scatter plot suggests that each variable on its own might not clearly separate the classes.

Twist: combining variables improves separation

**Case 2**: now, what happens if rotate the plot by 45 degrees? this rotation is a clever way of showing that combining these two variables can actually improve class separation.

- **improved separation**: after rotation, the separation between the two classes along the x-axis is now better by a factor of √2. this improvement happens because the combination (or averaging) of these two i.i.d. variables reduces the noise (variance) in the data.

Conclusion: redundant variables aren’t always redundant!

- **why this matters**: even though these variables seem redundant on their own (since they are i.i.d.), combining them can still lead to better performance because of noise reduction. this means that adding variables, even if they seem redundant, can sometimes improve the model by making the class separation clearer.

## How Does Correlation Impact Variable Redundancy?

Correlation and redundancy in variables

**basic idea of correlation**: correlation measures the relationship between two variables. if two variables are highly correlated (positively or negatively), it often suggests that they carry similar information.

**redundancy**: when variables are highly correlated, they might seem redundant because one can often be predicted from the other.

Examples:

**Case 1**: high correlation along the class center line
- **setup**: imagine you have two variables, and the class centers (mean positions of each class) are along a line. the variables are highly correlated along this line.
- **result**: the class distributions are stretched along the class center line (high covariance along this line). because of this correlation, if you combine these two variables (e.g., by summing them), you don’t really gain any additional separation power between the classes compared to using just one variable.
- **takeaway**: when variables are perfectly correlated along the direction of the class separation, they are truly redundant. adding one doesn’t improve your model’s performance because they both carry the same information.

**Case 2**: high correlation perpendicular to the class center line
- **setup**: now, the class centers are still positioned similarly, but the correlation is perpendicular to the class center line.
- **result**: the covariance (spread) is high in the direction perpendicular to the line joining the class centers. in this case, combining the two variables significantly improves class separation. even though they are correlated, they add complementary information.
- **takeaway**: here, even though the variables are correlated, they are not redundant. in fact, their combination is more powerful in distinguishing between classes.

**intra-class covariance**: the spread of the class data within each class (how the data points are distributed within the same class). when this spread (covariance) is in a direction that doesn’t align with the class center line, combining variables can help improve the model’s performance.

**correlation doesn’t always mean redundancy**: just because two variables are correlated doesn’t mean they are redundant. if the correlation is aligned in a way that doesn’t contribute to class separation, then they’re redundant. if the correlation exists but in a way that provides new information (like in figure 2.b), then they’re not redundant and can be very useful when combined.

**what this means for variable selection**

- **be careful with simple correlation-based selection**: if you only look at correlation when selecting variables, you might mistakenly throw away variables that, while correlated, could provide complementary information.

- **use models that consider interactions**: when you have correlated variables, use models that can capture interactions (like decision trees, random forests, or even neural networks) to see if those correlations actually contribute useful information.

- **test for redundancy by combining variables**: before dropping correlated variables, try combining them in ways that capture potential complementary information. for instance, consider their sum, difference, or even more complex transformations.

- **note**: correlation alone isn’t a clear indicator of redundancy. correlated variables might seem redundant, but depending on how the correlation aligns with the class structure, they could still provide valuable information. always check how variables interact with each other in the context of your specific problem before deciding to remove them based on correlation alone.
  - "Therefore, in the limit of perfect variable correlation (zero variance in the direction perpendicular to the class center line), single variables provide the same separation as the sum of the two variables. Perfectly correlated variables are truly redundant in the sense that no additional information is gained by adding them."
  - "One notices that in spite of their great complementarity (in the sense that a perfect separation can be achieved in the two-dimensional space spanned by the two variables), the two variables are (anti-)correlated. More anti-correlation is obtained by making the class centers closer and increasing the ratio of the variances of the class conditional distributions. Very high variable correlation (or anti-correlation) does not mean absence of variable complementarity."


## Can a Variable that is Useless by Itself be Useful with Others

"One concern about multivariate methods is that they are prone to overfitting. The problem is aggravated when the number of variables to select from is large compared to the number of examples. It is tempting to use a variable ranking method to filter out the least promising variables before using a multivariate method. Still one may wonder whether one could potentially lose some valuable variables through that filtering process"

**Can a variable that is useless by itself be useful with others?**

- **overfitting concern**: multivariate methods can overfit the data, especially if there are many variables compared to the number of examples. filtering out variables before applying these methods might be tempting but can lead to losing potentially valuable variables.

- **Example 1: useless by itself but useful together**:
  - **scenario**: imagine two variables with identical covariance matrices, where each variable alone doesn’t provide useful separation between classes. however, when both variables are used together, they improve class separability.
  - **figure 3.a**: demonstrates that even though each variable might be “useless” by itself, combining them can lead to better performance.

- **Example 2: both variables useless by themselves but useful together**:
  - **scenario**: consider a situation inspired by the xor problem where four gaussian clusters are placed at the corners of a square, and class labels are assigned based on the xor function. here, projections on individual axes offer no class separation, but the classes can be separated effectively in the two-dimensional space.
  - **figure 3.b**: shows that even if two variables don’t provide separation individually (e.g., due to overlapping class densities), using them together can reveal class separability.

**Takeaways**:
- **combination of variables**: variables that seem useless individually can provide significant performance improvements when used in combination with others.
- **complex data structures**: some problems (like xor) require multiple variables to capture complex relationships and separations that single variables cannot reveal.

this highlights the importance of considering the interaction between variables rather than relying solely on individual variable performance.

## Wrappers, filters, embedded

### wrappers
- wrappers treat feature selection as a search problem by evaluating multiple combinations of features, "wrapping" the selection process around a machine learning model to find the best subset of features for optimal model performance
- wrappers evaluate and select features externally, repeatedly training the model on different subsets of features and comparing performance
- imagine cooking a meal (the model) and trying different combinations of ingredients (features), cooking and tasting (training and evaluating) multiple times until finding the perfect recipe
- wrappers offer high accuracy since they are tailored to the specific model, but they are computationally expensive because the model is trained multiple times

### embeded
- embedded methods integrate feature selection into the training process of the model itself, selecting the most important features as part of the model’s optimization process without needing to train multiple times with different feature sets
- embedded methods select features internally as part of the model’s learning process, with the model deciding which features to keep or discard during training
- imagine baking a cake (the model) and the recipe (the model’s algorithm) automatically tells which ingredients are essential and which can be left out, adjusting in real-time without needing to bake multiple cakes
- embedded methods are more efficient since feature selection happens during model training, but they are sometimes less flexible, as the selection process is tied to the specific model

### essentially:
- wrappers: external, exhaustive, model-agnostic feature selection through multiple training rounds
- embedded: internal, efficient, model-specific feature selection done within the training process
- both methods aim to select the best features, but wrappers use brute force (external experimentation) while embedded methods make internal decisions


# 5. Feature contruction and space dimensionality reduction

## Feature contruction/representation

- specific: if domain knowledge available
- not a one-size-fits-all
- generic/thinking about before customizing: some examples below
- can be both supervised and unsupervised (less prone to overfitting)

| **feature contruction method**                    | **what**                                                                 | **example**                                                                                   |
|-------------------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Clustering**                | Grouping similar data points into clusters; use cluster labels or centroids as features. | Customer segmentation: k-means clustering to group customers based on ALL features and distance metrics (scaling matters here - for categorical non-ordinal features: onehot, target encoding,...), create cluster label, assess the quality of the cluster with some criteria/scores, then push to a recommendation system. |
| **Basic Linear Transforms**   | Transform data into a new coordinate system for dimensionality reduction or class separation. | PCA+SVD: reduce the number of features by getting linearly uncorrelated variables (principal comps) while maintaining variance; these variables belong the left singular matrix of the SVD matrix while the variances are the middle matrix or diagonal matrix. <br> LDA: find the topic/theme/summary, usually in NLP, assume each document has smaller, intertwined topics, use word frequency and probability to assess the popularity/appearances of these smaller/dense topics. A bit new on LDA: [more](https://towardsdatascience.com/latent-dirichlet-allocation-lda-9d1cd064ffa2). |
| **Advanced Linear Transforms: Spectral Transforms**       | Transforming data into frequency or spectral domains (?).                           | **Fourier Transform:** Extracting frequency components from time series finance data, sound data, image compression/filtering, medical imaging (from MRI scans which are composed of strong magentic fields and radiofrequency (frequency domain) to become readable spatial image domain),.... <br> **Hadamard Transform:** Creating features for error correction in data transmission. |
| **Wavelet Transforms / Convolutions of kernels** | Analyzing data at multiple scales or applying convolutional filters.                | **Wavelet Transform:** Extracting features in audio signals for speech recognition. <br> **Convolutional Kernels:** Detecting edges and textures in images using CNNs. |
| **Simple Functions (monomials)** | Creating new features by applying mathematical functions like products or powers. | **Polynomial Regression:** Generating interaction terms by multiplying pairs of variables (e.g., \(x_1 \times x_2\)). |

Goals:
- (1) how to achieve "best" reconstruction of data
- (2) efficient prediction process

Discussed methods in this paper:

1. Clustering (2 papers)

- Popular algos: K-means and hierarchial clustering
- Introduce a "new" constructed feature (centroid) as discriminant feature
- Combined with a supervision idea: distributional clustering
  - metric used: information bottleneck
  - $X_1$ is constructed feature => (largest possible compression) minimize $I(\vec{X}, X_1)$ and "maximize" $I(Y, X_1)$ (retaining most variance with target variable)
  - min obj:= $J = I(\vec{X}, X_1) - \beta(Y, X_1)$
- Example application: in text processing, suppose supervision aspect is that we know document categories, we can replace variable vectors containing document frequency counts with shorter one containing frequncy counts of document category (i.e. words are represented as distributions over document categories) OR group words into clusters rather than cluster documents. The goal is to discover patterns or themes in the text data by grouping related words together, reducing the dimensionality. Instead of representing a document by how many times the word "cat" appears, represent it by how frequently "cat" appears in documents of each category i.e. how often "cat" appears in pet vs. entertainment documents. So, if in a prediction task, a document containing a lot of "cat" words, it's likely to be in pet category.
  - essentially, first there are no labels, use clustering to find category/topic (narrow down) the documents; then, this becomes a classification problem (i.e. if given new document, how to put this document in categories defined by clustering or even using word clouds/frequency overall), to do the training process, use the category frequency count overall instead of word count per document, this algorithm applies to new document to make classification prediction

2. Matrix factorization (1 paper)

- SVD: unsupervised, linear combination of the original variables (best in terms of least square optimization)
  - least sq optimization: given a matrix A (full of numerical values, scaled), SVD decomposes into 3 matrices such that min sq differences between A and the UZV matrix of lower rank approximation
    - then, calculate sq difference between A and A_lower_rank_approx with Frobenius norm squared, try to minimize this
    - in other word, SVD tries to find the 3 decomposed components such that the difference sq is minimized
  - what if a mix of categorical and numerical variables? need to process the categorical variables with typical ones like onehot, ordinal, non-ordinal, then scaling with normalization or standardization! beware of this categorical stuff
- SDR: (new) sufficient dimensionality reduction (information theoretic unsup feature construction) ~ information bottle neck in clustering situation
  - $U$ is the left orthogonal singular vectors, represent original rows in a transformed space
  - $\Sigma$  is diagonal singular values, variances => decide how many dimensions/components to keep
  - $V^T$ is orthogonal right singular vectors, represent original columns in a transformed space => focus on this matrix to reduce dimensions/variables/features
    - contains the principal components (eigenvectors of the covariance matrix). Each row in this matrix corresponds to a principal component and tells us how each original feature contributes to this component (for each row, columns are the features ordered like the original data)
    - for example, row_1 = [0.1, 0.5, 0.92], then feature_1 contributes 0.1, feature_2 contributes 0.5, feature_3 contributes 0.92, all to the first principal component => feature_3 highly influences first pc AND we can then compare to other rows/pc: whichever pc has the larger absolutue coefficients indicating the features contribute the most to this component
    - each row is a pc, each col is original feature => each row contains the weights of the original features for the particular pc
    - calculation of $V^T$?
    - pick pc? other than simply looking at the coefficients, others method like average coefficients by row, plots, domain knowledge,...

3. Supervised feature selection (1 paper)

- Nested subset methods:
  - extract/construct feature during training/learning i.e. neural network with internal nodes or feature extractors
  - node pruning techniques such OBD LeCun
  - Gram-Schmidt orthogonalization (?) convert linearly independent vectors into orthogonal set to simplitfy linear algebra operations: i.e. QR decomposition, least sq problems, part of PCA+SVD
  - orthogonal matrices/vectors (?)

- Filters
  - uses mutual information criterion, helpful for non-linear
  - maximize mutual info between $I(\phi, y)$ between, for each data point, feature vector $\phi$ and target $y$
  - how features affect mutual info? use Parzen windows to model density function of features -> calculate derivatives of I wrt $\phi_i$ that are independent of the specific feature transform used
  - combine feature-independent derivatives with derivatives of transform $\phi_i$ wrt $w$ to adjust the transform param $w$ through gradient descent: $w_{t+1} = w_t + \eta \frac{\partial I}{\partial w} = w_t + \eta \frac{\partial I}{\partial \phi_i} \frac{\partial \phi_i}{\partial w}$

- Direct obj optimization
  - Weston et al. (2003): They introduce a method to select implicit features specifically for polynomial kernels. Their approach involves minimizing the $\ell_0$-norm, which helps in choosing a subset of features to improve the model's generalization while maintaining computational efficiency.

# 6. Validation methods

or metrics to evaluate significant variables. or related to problems involving generalization prediction and model selection, i.e.

- determine significant variables
- guide/halt search for good subset
- choose hyperparams
- evaluate final performance of the system

Splitting the data into training and validation is okay

Cross validation method:
- computationally expensive
- if not sufficiently many data points, statistical tests that evaluate signi of differences in validation errors may not be valid because cross-valid violates independence assumptions (since resuse! the same data points can be used in multiple training and validation splots) => hard to accurately estimate the significance of observed differences
- leave-one-out cross-valid?

Variable ranking or nested subset ranking methods:
- introduce a probe (random variable, threshold): variables that have a relevance (?) smaller or equal to that of the probe will be discarded
  - i.e. 3 additional "fake variables" drawn randomly from a normal dist, put along with the dataset
  - then, discard the variables that are less relevant (correlated?) than one of the 3 fake variables (accroding to weight magnitude criterion)
  - relevance: comparing the weight or influence of each variable to the fake variables
- sophisticated Gram-cshmidt forward selection and normal dist probe: compute rank of probe associated with a given risk of accepting an irrelevant variable
- non-parametric variant of probe involves shuffling real variable vectors
  - in a forward selection process, the introduction of fake variables does not disturb the selection because fake variables can be discarded when they are encountered.
  - a halting criterion: upper bound on the fraction of falsely relevant variables in the subset selected so far
- parametric version: use T-stat as ranking criterion is exactly the... T-test

# 7. Other problems

## Variance of variable subset selection

- some different subsets of variables can have identical predictive power (not necessarily a bad thing):
  - overfitting because model relies on different variable subsets that all perform, say, equally well -> might say that model has high variance
  - reproducibility concern: does this mean if model is used on a completely new data, what is the result?
  - incomplete: one subset can fail to be representative
- solution: boostrapping (using several samples of the data (can have replacements) with all features to see which features are consistently powerful across all features => final subset is the union of variables across samples) and Bayesian variable selection (ranked based on marginal distribution)

## Variable ranking in the context of others

- classic issues of multicollinearity (redundant) or weak but interactions/combinations are powerful
- solution: bootstrap and Bayesian methods, how? also, relief algorithms? and backward elimination

## Unsupervised variable selection

- no target $y$ selected at all, how to select the most signi subset?
- a few criteria:
  - salient: salient if it has a high variance or a large range, compared to others
  - entropy: high entropy if the distribution of examples is uniform
  - smooth: (i.e. time series) smooth if on average its local curvature is moderate or low (less bending sharply)
  - density: high-density region if it is highly correlated with many other variables
  - reliable: if the measurement error bars computed by repeating measurements are small (consistently) compared to the variability of the variable values (i.e. ANOVA) 
    - given a feature (milage) and some categories (car_1, car_2, car_2)
    - measurement error bars = standard error = standard devication divided by square of number of measurements (# of milages for each car)
    - variability = standard deviation of the feature

## Forward vs Backward selection

- Backward elimination (although less computationally efficient) offers a way to assess importance of variables based on the context of the other variables while removing really redundant variables (aviod multicollinearity)
  - evaluates each variable's importance by examining its impact on the model's performance when other variables are included. This ensures that the contribution of each variable is assessed in conjunction with the others
  - can work with different models either in reg or classification
- example:
  - start with all variables: fit a regression model using all five features x1, x2, x3, x4, and x5.
  - evaluate feature significance: check the p-values of each feature. suppose the p-values indicate that x5 (distance to the city center) has a high p-value, suggesting it has little impact on predicting house prices.
  - remove x5 from the model. refit the model using x1, x2, x3, and x4.
  - repeat evaluation: evaluate the new model and check the significance of the remaining features. suppose x3 (age of the house) now shows a high p-value.
  - remove x3 from the model. refit using x1, x2, and x4.
  - continue until optimal model is found: all remaining features are statistically significant and contribute meaningfully to the model’s performance.

## Multi-class problem

Multi-class variable ranking criteria:
- Fisher's criterion ~ F-stat using the ANOVA test (probe for multi-class case)
- Wrappers or embedded methods depend on the capability of the classifier used to handle the multi-class
- Examples: LDA (linear discriminant analysis), multi-class version of Fisher's linear discriminant, multi-class SVMs

Advantages of multi-class:
- the larger the number of classes, the less likely a "random" set of features provide a good separation
- why? example: if features are drawn from a random distribution, finding a feature that matches the target class becomes easier
- say, 10 different categories to distinguish. If randomly pick features, the chance that one of these random features is helpful for distinguishing between your 10 categories by chance is very low. So, if a feature does help, it’s likely to be meaningful.
- In Quiz Gameshow, topics: Math, History, Science, Geography, Literature, etc. (let's say 10 topics). If a player gets a question right, it's more likely that they are knowledgeable about that specific topic. With many topics, it's easier to tell if a player's correct answer is due to actual knowledge rather than random guessing.

Disadvantages:
- uneven distributions across classes
- multi-class methods may over-represent or easily separable classes (overfitting)
- solution: mix ranked lists of several two-class problems

## Selection of examples

Selection vs construction of feature/variables related -> pattern selection or selection of data points in kernel methods (identifying the most relevant examples or patterns in the data that help in making accurate predictions)

Example cases in paper:

- Mislabeled examples: If examples in the data are incorrectly labeled, it can lead to selecting irrelevant features because the model might wrongly associate the features with incorrect patterns.
- Reliable labeling: If labels are accurate, then focusing on informative patterns close to the decision boundary can help avoid selecting irrelevant features. This is because reliable labels ensure that the features selected are genuinely useful for distinguishing between classes.


# 8. Recommendation

it is important when starting with a new problem to have a few baseline performance values. recommend 
- using a linear predictor of choice (e.g. a linear SVM) and select variables in two alternate ways: 
  - (1) with a variable ranking method using a correlation coefficient or mutual information; or
  - (2) with a nested subset selection method performing forward or backward selection or with multiplicative updates. 
- Further down the road, connections need to be made between the problems of variable and feature selection and
those of experimental design and active learning, in an effort to move away from observational data
toward experimental data, and to address problems of causality inference