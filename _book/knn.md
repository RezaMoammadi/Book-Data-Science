# Classification using k-Nearest Neighbors {#chapter-knn}



The k-Nearest Neighbors (kNN) algorithm is a simple yet effective machine learning technique, widely used for solving classification problems. Its intuitive approach and ease of implementation make it a go-to choice for beginners and a reliable tool for experienced practitioners. In this chapter, we will delve into the details of the kNN algorithm, demonstrate its implementation in R, and discuss its practical applications. But before we focus on kNN, it’s essential to revisit the fundamental concept of classification, one of the cornerstone tasks in machine learning.

## Classification  

Have you ever wondered how your email app effortlessly filters spam, how your streaming service seems to know exactly what you want to watch next, or how banks detect fraudulent credit card transactions in real-time? These seemingly magical predictions are made possible by **classification**, a fundamental task in machine learning.  

At its core, classification involves assigning a label or category to an observation based on its features. For example, given customer data, classification can predict whether they are likely to churn or stay loyal. Unlike regression, which predicts continuous numerical values (e.g., house prices), classification deals with discrete outcomes. The target variable, often called the **class** or **label**, can either be: 

- **Binary**: Two possible categories (e.g., spam vs. not spam).  
- **Multi-class**: More than two categories (e.g., car, bicycle, or pedestrian in image recognition).  

From diagnosing diseases to identifying fraudulent activities, classification is a versatile tool used across countless domains to solve practical problems.

### Where Is Classification Used? {-}

Classification algorithms power many everyday applications and cutting-edge technologies. Here are some examples:  
- **Email filtering**: Sorting spam from non-spam messages.  
- **Fraud detection**: Identifying suspicious credit card transactions.  
- **Customer retention**: Predicting whether a customer will churn.  
- **Medical diagnosis**: Diagnosing diseases based on patient records.  
- **Object recognition**: Detecting pedestrians and vehicles in self-driving cars.  
- **Recommendation systems**: Suggesting movies, songs, or products based on user preferences.  

Every time you interact with technology that "predicts" something for you, chances are, a classification model is working behind the scenes.

### How Does Classification Work? {-}

Classification involves two critical phases:  

1. **Training Phase**: The algorithm learns patterns from a labeled dataset, which contains both predictor variables (features) and target class labels. For instance, in a fraud detection system, the algorithm might learn that transactions involving unusually high amounts and originating from foreign locations are more likely to be fraudulent.  
2. **Prediction Phase**: Once the model is trained, it applies these learned patterns to classify new, unseen data. For example, given a new transaction, the model predicts whether it is fraudulent or legitimate.

A good classification model does more than just memorize the training data—it **generalizes** well, meaning it performs accurately on new, unseen data. For example, a model trained on historical medical data should be able to correctly diagnose a new patient it has never seen before.

### Which Classification Algorithm Should You Use? {-}

Different classification algorithms are designed for different kinds of problems and datasets. Some commonly used algorithms include:  
- **k-Nearest Neighbors (kNN)**: A simple, distance-based algorithm (covered in this chapter).  
- **Logistic Regression**: Popular for binary classification tasks, such as predicting customer churn.  
- **Decision Trees and Random Forests**: Versatile, interpretable methods for complex problems.  
- **Naive Bayes**: Particularly useful for text classification, like spam filtering.  
- **Neural Networks**: Effective for handling high-dimensional and complex data, such as images or natural language.

The choice of algorithm depends on factors like the dataset size, feature relationships, and the desired trade-off between interpretability and performance. For example, if you’re working with a small dataset and need an easy-to-interpret solution, kNN or Decision Trees might be ideal. On the other hand, if you’re analyzing high-dimensional data like images, Neural Networks could be more suitable.

To see classification in action, imagine a **bank dataset** where the goal is to predict whether a customer will make a deposit (`deposit = yes`) or not (`deposit = no`). The features might include customer details like `age`, `education`, `job`, and `marital status`. By training a classification model on this data, the bank can identify and target potential customers who are likely to invest, improving their marketing strategy.

### Why Is Classification Important? {-}

Classification forms the backbone of countless machine learning applications that drive smarter decisions and actionable insights in industries like finance, healthcare, retail, and technology. Understanding how it works is a critical step in mastering machine learning and applying it to solve real-world problems.  

In the rest of this chapter, we’ll explore the **k-Nearest Neighbors (kNN)** algorithm, a straightforward yet powerful method for classification. Its simplicity and intuitive nature make it an excellent choice for beginners and a foundational building block for more advanced algorithms. Let’s dive in!  

## How k-Nearest Neighbors Works  

Have you ever tried to make a decision by asking a few trusted friends for their advice? The **k-Nearest Neighbors (kNN)** algorithm works in a similar way—it "asks" the nearest data points in its neighborhood to determine the category of a new observation. This simple yet powerful idea makes kNN one of the most intuitive methods in machine learning.

Unlike many algorithms that require a complex training phase, kNN is a **lazy learning** or **instance-based** method. It doesn't build an explicit model during training; instead, it stores the entire training dataset and makes predictions on-the-fly by finding the nearest neighbors of a given observation. The parameter \(k\) determines how many neighbors to consider, and the majority class among these neighbors becomes the prediction.

### How Does kNN Classify a New Observation? {-}

When a new observation needs to be classified, kNN calculates its **distance** to every data point in the training set using a specified distance metric, such as Euclidean distance. The algorithm identifies the \(k\)-nearest neighbors and predicts the class based on a **majority vote** among these neighbors.  

To better understand how this works, let’s look at Figure \@ref(fig:knn-image), which illustrates a simple example with two classes: <span style="color: red;">Class A (red circles)</span> and <span style="color: blue;">Class B (blue squares)</span>.  

A new data point, represented by a **dark star**, needs to be classified. The figure demonstrates the predictions for two different values of \(k\):  

1. **When \(k = 3\)**: The algorithm looks at the 3 closest neighbors to the dark star—two blue squares and one red circle. Since the majority of these neighbors belong to **Class B (blue squares)**, the new point is classified as Class B.  
2. **When \(k = 6\)**: The algorithm now considers a larger neighborhood of 6 neighbors. In this case, four red circles and two blue squares are the nearest neighbors. With the majority vote shifting to **Class A (red circles)**, the new point is classified as Class A.  

\begin{figure}

{\centering \includegraphics[width=0.75\linewidth]{images/knn} 

}

\caption{A two-dimensional toy dataset with two classes (Class A and Class B) and a new data point (dark star), illustrating the k-Nearest Neighbors algorithm with k = 3 and k = 6.}(\#fig:knn-image)
\end{figure}

**Key Takeaway from the Figure:**  

- Increasing \(k\) smooths predictions by incorporating more neighbors into the decision-making process. However, this may lead to less sensitivity to local patterns.  
- In this example, when \(k = 6\), the larger neighborhood includes more red circles, shifting the majority class to Class A. This demonstrates how majority voting in larger neighborhoods can significantly affect the outcome.  

### Strengths and Limitations of kNN {-}

The simplicity of kNN makes it an excellent starting point for understanding classification. By relying only on distance metrics and majority voting, it avoids the complexity of training explicit models. However, this simplicity comes with trade-offs:  

- **Strengths**:  
    - Easy to understand and implement.  
    - Effective for small datasets with clear patterns.  
    
- **Limitations**:  
    - Sensitive to irrelevant or noisy features, as distance calculations may become less meaningful.  
    - Computationally expensive for large datasets, since the algorithm must compute distances for all training points during prediction.  
    - Requires careful choice of \(k\) to balance sensitivity to local patterns and robustness to noise.  

### A Practical Example of kNN in Action {-}

To further illustrate kNN, consider a toy simulated example from a real-world scenario involving drug prescription classification. A dataset of 200 patients includes their **age**, **sodium-to-potassium (Na/K) ratio**, and the drug type they were prescribed. Figure \@ref(fig:scatter-plot-ex-drug) shows a scatter plot of this data, where the drug types are represented by:  
    
- **Red circles** for Drug A,  
- **Green triangles** for Drug B, and  
- **Blue squares** for Drug C.  

\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{knn_files/figure-latex/scatter-plot-ex-drug-1} 

}

\caption{Scatter plot of Age vs. Sodium/Potassium Ratio for 200 patients, with drug type indicated by color and shape.}(\#fig:scatter-plot-ex-drug)
\end{figure}

Suppose we now have three new patients whose drug classifications are unknown. Their details are as follows:
    
1. **Patient 1**: 40 years old with a Na/K ratio of 30.5,  
2. **Patient 2**: 28 years old with a Na/K ratio of 9.6, and  
3. **Patient 3**: 61 years old with a Na/K ratio of 10.5.  

These patients are represented as **orange circles** in Figure \@ref(fig:scatter-plot-ex-drug-2). Using kNN, we will classify the drug type for each patient.

\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{knn_files/figure-latex/scatter-plot-ex-drug-2-1} 

}

\caption{Scatter plot of Age vs. Sodium/Potassium Ratio for 200 patients, with drug type indicated by color and shape. The three new patients are represented by large orange circles.}(\#fig:scatter-plot-ex-drug-2)
\end{figure}

For **Patient 1**, who is located deep within a cluster of red-circle points (Drug A), the classification is straightforward: **Drug A**. All the nearest neighbors belong to Drug A, making it an easy decision.

For **Patient 2**, the situation is more nuanced. 

- **With \(k = 1\)**: The nearest neighbor is a blue square, so the classification is **Drug C**.  
- **With \(k = 2\)**: There is a tie between Drug B and Drug C, leaving no clear majority.  
- **With \(k = 3\)**: Two out of the three nearest neighbors are blue squares, resulting in a majority vote for **Drug C**.  

For **Patient 3**, the scenario becomes even more ambiguous:  

- **With \(k = 1\)**: The closest neighbor is a blue square, so the classification is **Drug C**.  
- **With \(k = 2 or 3\)**: The neighbors belong to multiple classes, resulting in ties or uncertainty.  

\begin{figure}
\includegraphics[width=0.33\linewidth]{knn_files/figure-latex/scatter-plot-ex-drug-3-1} \includegraphics[width=0.33\linewidth]{knn_files/figure-latex/scatter-plot-ex-drug-3-2} \includegraphics[width=0.33\linewidth]{knn_files/figure-latex/scatter-plot-ex-drug-3-3} \caption{Zoom-in plots for the three new patients and their nearest neighbors. The left plot is for Patient 1, the middle plot is for Patient 2, and the right plot is for Patient 3.}(\#fig:scatter-plot-ex-drug-3)
\end{figure}

These examples illustrate several key considerations for kNN:

- The value of \(k\) determines how sensitive the algorithm is to local patterns or noise.  
- Distance metrics, such as Euclidean distance, affect how neighbors are selected.  
- Proper feature scaling is essential to ensure that all variables contribute fairly to the distance calculation.  

To classify a new observation, kNN relies on measuring the similarity between data points. This brings us to the question: _how do we define and calculate this similarity?_

## Distance Metrics  

In the k-Nearest Neighbors (kNN) algorithm, the classification of a new data point is determined by identifying the most _similar_ records from the training dataset. But how do we define and measure _similarity_? While similarity might seem intuitive, applying it in machine learning requires precise **distance metrics**. These metrics quantify the "closeness" or "distance" between two data points in a multidimensional space, directly influencing how neighbors are selected for classification.

Imagine you’re shopping online and looking for recommendations. You’re a 50-year-old married female—who’s more "similar" to you: a 40-year-old single female or a 30-year-old married male? The answer depends on how we measure the distance between you and each person. In kNN, this distance is calculated based on features such as age and marital status. The smaller the distance, the more "similar" the two individuals are, and the more influence they have in determining the recommendation or classification.

The most widely used distance metric in kNN is **Euclidean distance**, which measures the straight-line distance between two points. Think of it as the "as-the-crow-flies" distance, similar to the shortest path between two locations on a map. This metric is intuitive and aligns with how we often perceive distance in the real world.

In mathematical terms, the Euclidean distance between two points, \(x\) and \(y\), in \(n\)-dimensional space is given by:

\[
\text{dist}(x, y) = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + \ldots + (x_n - y_n)^2} 
\]

Where:  

- \( x = (x_1, x_2, \ldots, x_n) \) and \( y = (y_1, y_2, \ldots, y_n) \) represent the feature vectors of the two points.  
- The differences between corresponding features (\( x_i - y_i \)) are squared, summed, and then square-rooted to calculate the distance.  

### Example: Calculating Euclidean Distance  

Let’s calculate the Euclidean distance between two patients based on their **age** and **sodium/potassium (Na/K) ratio**:  

- Patient 1: \( x = (40, 30.5) \)  
- Patient 2: \( y = (28, 9.6) \)  

Using the formula:  
\[
\text{dist}(x, y) = \sqrt{(40 - 28)^2 + (30.5 - 9.6)^2} = \sqrt{(12)^2 + (20.9)^2} = 24.11
\]

This result quantifies the dissimilarity between the two patients. In kNN, this distance will help determine how similar Patient 1 is to Patient 2 and whether Patient 1 should be classified into the same drug class as Patient 2.

### A Note on Choosing Distance Metrics {-}

While there are many distance metrics, such as Manhattan Distance, Hamming Distance, and Cosine Similarity, by default, **Euclidean distance** is commonly used in kNN. It works well in many scenarios, particularly when features are continuous and have been properly scaled. Choosing the right distance measure is somewhat beyond the scope of this book, but for most general purposes, Euclidean distance is a reliable choice. If your dataset has unique characteristics or categorical features, you might need to explore alternative metrics; For more details, refer to the `dist()` function in R.

## How to Choose an Optimal \( k \)  

How many opinions do you seek before making an important decision? Too few might lead to a biased perspective, while too many might dilute the relevance of the advice. Similarly, in the k-Nearest Neighbors (kNN) algorithm, the choice of \( k \)—the number of neighbors considered for classification—directly impacts the model's performance. But how do we find the right \( k \)?

There is no universally "correct" value for \( k \). The optimal choice depends on the specific dataset and problem at hand, requiring careful consideration of the trade-offs involved.

### Balancing Overfitting and Underfitting {-}

When \( k \) is set to a very small value, such as \( k = 1 \), the algorithm becomes highly sensitive to outliers in the training data. Each new observation is classified solely based on its single closest neighbor. This can lead to **overfitting**, where the model memorizes the training data but struggles to generalize to unseen data. For example, a small cluster of mislabeled data points could disproportionately influence predictions, reducing the model's reliability.

Conversely, as \( k \) increases, the algorithm incorporates more neighbors into the classification decision. Larger \( k \) values smooth the decision boundary, making the model less sensitive to noise and outliers. However, if \( k \) becomes too large, the model may over-simplify, averaging out meaningful patterns in the data. For instance, when \( k \) is comparable to the size of the training set, the majority class will dominate predictions, leading to **underfitting**.

Finding the right \( k \) involves striking a balance between these extremes. Smaller \( k \) values capture local patterns more effectively, while larger \( k \) values provide robustness at the expense of detail.  

### Choosing \( k \) Through Validation {-}

In practice, selecting \( k \) is an iterative process. A common approach is to evaluate the algorithm’s performance for multiple \( k \) values using a **validation set** or **cross-validation**. Performance metrics like accuracy, precision, recall, or F1-score guide the selection of the \( k \) that works best for the dataset.

To illustrate, let’s use the **churn** dataset and evaluate the accuracy of the kNN algorithm across \( k \) values ranging from 1 to 30. Figure \@ref(fig:kNN-plot) shows how accuracy fluctuates as \( k \) increases. This plot is generated using the `kNN.plot()` function from the **liver** package in R.


```
   Setting levels: reference = "yes", case = "no"
   Setting levels: reference = "yes", case = "no"
```

\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{knn_files/figure-latex/kNN-plot-1} 

}

\caption{Accuracy of the k-Nearest Neighbors algorithm for different values of k in the range from 1 to 30.}(\#fig:kNN-plot)
\end{figure}

From the plot, we observe that the accuracy of the kNN algorithm fluctuates as \( k \) increases. In this example, the highest accuracy is achieved when \( k = 5 \). At this value, the kNN algorithm balances sensitivity to local patterns with robustness to noise, delivering an accuracy of 0.932 and an error rate of 0.068.

Choosing the optimal \( k \) is as much an art as it is a science. While there’s no universal rule for selecting \( k \), experimentation and validation are key. Start with a range of plausible \( k \) values, test the model's performance, and select the one that provides the best results based on your chosen metric.

Keep in mind that the optimal \( k \) may vary across datasets, so it’s essential to repeat this process whenever applying kNN to a new problem. By carefully tuning \( k \), you can ensure that your kNN model is both accurate and generalizable, striking the perfect balance between overfitting and underfitting.

## Preparing Data for kNN  

The effectiveness of the k-Nearest Neighbors (kNN) algorithm relies heavily on how the dataset is prepared. Since kNN uses distance metrics to evaluate similarity between data points, proper preprocessing is crucial to ensure accurate and meaningful results. Two essential steps in this process are **feature scaling** and **one-hot encoding**, which enable the algorithm to handle numerical and categorical features effectively.

### Feature Scaling  

In most datasets, numerical features often have vastly different ranges. For instance, **age** may range from 20 to 70, while **income** could range from 20,000 to 150,000. Without proper scaling, features with larger ranges (like income) dominate distance calculations, leading to biased predictions. To address this, all numerical features must be transformed to comparable scales.

A widely used scaling method is **min-max scaling**, which transforms each feature to a specified range, typically [0, 1], using the formula:

\[
x_{\text{scaled}} = \frac{x - \min(x)}{\max(x) - \min(x)}
\]

Here, \(x\) represents the original feature value, and \(\min(x)\) and \(\max(x)\) are the feature's minimum and maximum values. This ensures that all features contribute equally to the distance metric. Another commonly used method is **z-score standardization**, which scales features to have a mean of 0 and a standard deviation of 1:

\[
x_{\text{scaled}} = \frac{x - \text{mean}(x)}{\text{sd}(x)}
\]

This method is particularly useful when features follow different distributions or have varying units. Both methods prevent any single feature from dominating distance calculations, ensuring fair treatment of all numerical variables.

> **Important:** Scaling must always be performed **after partitioning** the dataset into training and test sets. Scaling parameters (e.g., minimum, maximum, mean, standard deviation) must be calculated using only the training set and applied consistently to both training and test sets. This ensures that test data remains independent, avoiding information leakage that could bias the results.

### Scaling Training and Test Data the Same Way  

To illustrate the importance of consistent scaling, consider the **patient drug classification problem**, which involves two features: `age` and `sodium/potassium (Na/K) ratio`. Figure \@ref(fig:scatter-plot-ex-drug-2) shows a dataset of 200 patients as the training set, with three additional patients in the test set. Using the `minmax()` function from the **liver** package, we demonstrate both correct and incorrect ways to scale the data:


```r
# A proper way to scale the data
train_scaled = minmax(train_data, col = c("Age", "Ratio"))

test_scaled = minmax(test_data, col = c("Age", "Ratio"), 
                     min = c(min(train_data$Age), min(train_data$Ratio)), 
                     max = c(max(train_data$Age), max(train_data$Ratio)))

# An incorrect way to scale the data
train_scaled_wrongly = minmax(train_data, col = c("Age", "Ratio"))
test_scaled_wrongly  = minmax(test_data , col = c("Age", "Ratio"))
```

The difference is illustrated in Figure \@ref(fig:ex-proper-scaling). The middle panel shows the results of proper scaling, where the test set is scaled using the same parameters derived from the training set. This ensures consistency in distance calculations across both datasets. In contrast, the right panel shows improper scaling, where the test set is scaled independently. This leads to distorted relationships between the training and test data, which can cause unreliable predictions.

> **Key Insight:** Proper scaling ensures that distance metrics remain valid, while improper scaling creates inconsistencies that undermine the kNN algorithm’s performance. **Always derive scaling parameters from the training set and apply them consistently to the test set**.

### One-Hot Encoding  

Categorical features, such as **marital status** or **subscription type**, cannot be directly used in distance calculations because distance metrics like Euclidean distance only work with numerical data. To overcome this, we use **one-hot encoding**, which converts categorical variables into binary (dummy) variables. For example, the categorical variable **voice.plan**, with levels `yes` and `no`, can be encoded as:

\[
\text{voice.plan-yes} = 
\bigg\{
\begin{matrix}
1 \quad \text{if voice plan = yes}  \\
0 \quad \text{if voice plan = no} 
\end{matrix}
\]

Similarly, a variable like **marital status** with three levels (`single`, `married`, `divorced`) can be encoded into two binary features:

\[
\text{marital-single} = 
\bigg\{
\begin{matrix}
1 \quad \text{if marital status = single}  \\
0 \quad \text{otherwise}
\end{matrix}
\]

\[
\text{marital-married} = 
\bigg\{
\begin{matrix}
1 \quad \text{if marital status = married}  \\
0 \quad \text{otherwise}
\end{matrix}
\]

The absence of both `marital_single` and `marital_married` implies the third category (`divorced`). This approach ensures that the categorical variable is fully represented, while maintaining the same scale as numerical features. For a categorical variable with \(k\) levels, \(k-1\) binary features are created to avoid redundancy.

The **liver** package in R provides the `one.hot()` function to perform one-hot encoding automatically. It identifies categorical variables and encodes them into binary columns, leaving numerical features unchanged. For example, applying one-hot encoding to the **marital** variable in the *bank* dataset adds binary columns for the encoded categories:


```r
# To perform one-hot encoding on the "marital" variable
bank_encoded <- one.hot(bank, cols = c("marital"), dropCols = FALSE)

str(bank_encoded)
   'data.frame':	4521 obs. of  20 variables:
    $ age             : int  30 33 35 30 59 35 36 39 41 43 ...
    $ job             : Factor w/ 12 levels "admin.","blue-collar",..: 11 8 5 5 2 5 7 10 3 8 ...
    $ marital         : Factor w/ 3 levels "divorced","married",..: 2 2 3 2 2 3 2 2 2 2 ...
    $ marital_divorced: int  0 0 0 0 0 0 0 0 0 0 ...
    $ marital_married : int  1 1 0 1 1 0 1 1 1 1 ...
    $ marital_single  : int  0 0 1 0 0 1 0 0 0 0 ...
    $ education       : Factor w/ 4 levels "primary","secondary",..: 1 2 3 3 2 3 3 2 3 1 ...
    $ default         : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
    $ balance         : int  1787 4789 1350 1476 0 747 307 147 221 -88 ...
    $ housing         : Factor w/ 2 levels "no","yes": 1 2 2 2 2 1 2 2 2 2 ...
    $ loan            : Factor w/ 2 levels "no","yes": 1 2 1 2 1 1 1 1 1 2 ...
    $ contact         : Factor w/ 3 levels "cellular","telephone",..: 1 1 1 3 3 1 1 1 3 1 ...
    $ day             : int  19 11 16 3 5 23 14 6 14 17 ...
    $ month           : Factor w/ 12 levels "apr","aug","dec",..: 11 9 1 7 9 4 9 9 9 1 ...
    $ duration        : int  79 220 185 199 226 141 341 151 57 313 ...
    $ campaign        : int  1 1 1 4 1 2 1 2 2 1 ...
    $ pdays           : int  -1 339 330 -1 -1 176 330 -1 -1 147 ...
    $ previous        : int  0 4 1 0 0 3 2 0 0 2 ...
    $ poutcome        : Factor w/ 4 levels "failure","other",..: 4 1 1 4 4 1 2 4 4 1 ...
    $ deposit         : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
```

> **Note:** One-hot encoding is unnecessary for ordinal features, where the categories have a natural order (e.g., `low`, `medium`, `high`). Ordinal variables should instead be assigned numerical values that preserve their order (e.g., `low = 1`, `medium = 2`, `high = 3`), enabling the kNN algorithm to treat them as numerical features.

## Applying kNN Algorithm in Practice {#sec-kNN-churn}

Applying the kNN algorithm involves several key steps, from preparing the data to training the model, making predictions, and evaluating its performance. In this section, we demonstrate the entire workflow using the **churn** dataset from the **liver** package in R. The target variable, `churn`, indicates whether a customer has churned (`yes`) or not (`no`), while the predictors include customer characteristics like account length, international plan status, and call details. Here is the strcure of the dataset:


```r
str(churn)
   'data.frame':	5000 obs. of  20 variables:
    $ state         : Factor w/ 51 levels "AK","AL","AR",..: 17 36 32 36 37 2 20 25 19 50 ...
    $ area.code     : Factor w/ 3 levels "area_code_408",..: 2 2 2 1 2 3 3 2 1 2 ...
    $ account.length: int  128 107 137 84 75 118 121 147 117 141 ...
    $ voice.plan    : Factor w/ 2 levels "yes","no": 1 1 2 2 2 2 1 2 2 1 ...
    $ voice.messages: int  25 26 0 0 0 0 24 0 0 37 ...
    $ intl.plan     : Factor w/ 2 levels "yes","no": 2 2 2 1 1 1 2 1 2 1 ...
    $ intl.mins     : num  10 13.7 12.2 6.6 10.1 6.3 7.5 7.1 8.7 11.2 ...
    $ intl.calls    : int  3 3 5 7 3 6 7 6 4 5 ...
    $ intl.charge   : num  2.7 3.7 3.29 1.78 2.73 1.7 2.03 1.92 2.35 3.02 ...
    $ day.mins      : num  265 162 243 299 167 ...
    $ day.calls     : int  110 123 114 71 113 98 88 79 97 84 ...
    $ day.charge    : num  45.1 27.5 41.4 50.9 28.3 ...
    $ eve.mins      : num  197.4 195.5 121.2 61.9 148.3 ...
    $ eve.calls     : int  99 103 110 88 122 101 108 94 80 111 ...
    $ eve.charge    : num  16.78 16.62 10.3 5.26 12.61 ...
    $ night.mins    : num  245 254 163 197 187 ...
    $ night.calls   : int  91 103 104 89 121 118 118 96 90 97 ...
    $ night.charge  : num  11.01 11.45 7.32 8.86 8.41 ...
    $ customer.calls: int  1 1 0 2 3 0 3 0 1 0 ...
    $ churn         : Factor w/ 2 levels "yes","no": 2 2 2 2 2 2 2 2 2 2 ...
```

It shows that data are as a *data.frame* object in **R** with 5000 observations and 19 features along with the target binary variable (the last column) with name *churn* that indicates whether customers churned (left the company) or not. Our goal is to build a kNN model that accurately predicts customer churn based on these features.

In Chapter \@ref(chapter-EDA), we explored the **churn** dataset and identified key features that influence customer churn. Based on that results we will use the following features to build the kNN model:

`account.length`, `voice.plan`, `voice.messages`, `intl.plan`, `intl.mins`, `day.mins`, `eve.mins`, `night.mins`, and `customer.calls`.

Let's start by preparing the data for the kNN algorithm by performing feature scaling and one-hot encoding. We will then proceed with selecting an optimal \( k \), training the kNN model, and evaluating its performance.

### Step 1: Preparing the Data  

The first step in applying kNN is to partition the data into training and test sets, followed by preprocessing tasks like feature scaling and one-hot encoding. Since the dataset is already clean and contains no missing values, we can proceed directly to these steps.

We split the dataset into an 80% training set and a 20% test set using the `partition()` function from the **liver** package:


```r
set.seed(43)

data_sets = partition(data = churn, ratio = c(0.8, 0.2))

train_set = data_sets$part1
test_set  = data_sets$part2

actual_test  = test_set$churn
```

The `partition()` function ensures a randomized split, preserving the overall distribution of the target variable between the training and test sets. Note that before proceeding, we should validate the partitions. We skip this step here as we did it in Section \@ref(sec-validate-partition). 

#### One-Hot Encoding {-} 

Categorical variables, such as `voice.plan` and `intl.plan`, are converted into binary (dummy) variables using the `one.hot()` function. This ensures that the kNN algorithm can handle categorical data effectively:  


```r
categorical_vars = c("voice.plan", "intl.plan")

train_onehot = one.hot(train_set, cols = categorical_vars)
test_onehot  = one.hot(test_set,  cols = categorical_vars)

str(test_onehot)
   'data.frame':	1000 obs. of  22 variables:
    $ state         : Factor w/ 51 levels "AK","AL","AR",..: 2 50 14 46 10 4 25 15 11 32 ...
    $ area.code     : Factor w/ 3 levels "area_code_408",..: 3 2 1 3 2 2 2 2 2 1 ...
    $ account.length: int  118 141 85 76 147 130 20 142 72 149 ...
    $ voice.plan_yes: int  0 1 1 1 0 0 0 0 1 0 ...
    $ voice.plan_no : int  1 0 0 0 1 1 1 1 0 1 ...
    $ voice.messages: int  0 37 27 33 0 0 0 0 37 0 ...
    $ intl.plan_yes : int  1 1 0 0 0 0 0 0 0 0 ...
    $ intl.plan_no  : int  0 0 1 1 1 1 1 1 1 1 ...
    $ intl.mins     : num  6.3 11.2 13.8 10 10.6 9.5 6.3 14.2 14.7 11.1 ...
    $ intl.calls    : int  6 5 4 5 4 19 6 6 6 9 ...
    $ intl.charge   : num  1.7 3.02 3.73 2.7 2.86 2.57 1.7 3.83 3.97 3 ...
    $ day.mins      : num  223 259 196 190 155 ...
    $ day.calls     : int  98 84 139 66 117 112 109 95 80 94 ...
    $ day.charge    : num  38 44 33.4 32.2 26.4 ...
    $ eve.mins      : num  221 222 281 213 240 ...
    $ eve.calls     : int  101 111 90 65 93 99 84 63 102 92 ...
    $ eve.charge    : num  18.8 18.9 23.9 18.1 20.4 ...
    $ night.mins    : num  203.9 326.4 89.3 165.7 208.8 ...
    $ night.calls   : int  118 97 75 108 133 78 102 148 71 108 ...
    $ night.charge  : num  9.18 14.69 4.02 7.46 9.4 ...
    $ customer.calls: int  0 0 1 1 0 0 0 2 3 1 ...
    $ churn         : Factor w/ 2 levels "yes","no": 2 2 2 2 2 2 2 2 2 2 ...
```

For instance, the `voice.plan` variable is transformed into `voice.plan_yes` and `voice.plan_no`. However, since the presence of one category implies the absence of the other, we retain only one dummy variable (e.g., `voice.plan_yes`) for simplicity.  

#### Feature Scaling  {-} 

kNN relies on distance metrics, which are sensitive to the scale of the features. To ensure fair contributions from all features, we scale the numerical variables using min-max scaling. The `minmax()` function from the **liver** package is applied to both training and test sets, using scaling parameters derived from the training set:


```r
numeric_vars = c("account.length", "voice.messages", "intl.mins", "intl.calls", 
                 "day.mins", "day.calls", "eve.mins", "eve.calls", 
                 "night.mins", "night.calls", "customer.calls")

min_train = sapply(train_set[, numeric_vars], min)
max_train = sapply(train_set[, numeric_vars], max)

train_scaled = minmax(train_onehot, col = numeric_vars, min = min_train, max = max_train)
test_scaled  = minmax(test_onehot,  col = numeric_vars, min = min_train, max = max_train)
```

The `minmax()` function scales the features to the range [0, 1]. By deriving the scaling parameters (minimum and maximum) from the training set, we ensure consistency and avoid data leakage.

### Step 2: Choosing an Optimal \( k \)  

The choice of \( k \), the number of neighbors, significantly affects the performance of the kNN algorithm. To identify the optimal \( k \), we evaluate the model’s accuracy for different values of \( k \) using the `kNN.plot()` function:  


```r
formula = churn ~ account.length + voice.plan_yes + voice.messages + 
                  intl.plan_yes + intl.mins + intl.calls + 
                  day.mins + day.calls + eve.mins + eve.calls + 
                  night.mins + night.calls + customer.calls

kNN.plot(formula = formula, train = train_scaled, test = test_scaled, 
         k.max = 30, set.seed = 43)
   Setting levels: reference = "yes", case = "no"
```



\begin{center}\includegraphics{knn_files/figure-latex/unnamed-chunk-8-1} \end{center}

The `kNN.plot()` function generates a plot of accuracy versus \( k \) values, allowing us to visually identify the optimal \( k \). In this case, the highest accuracy is achieved at \( k = 5 \), striking a balance between sensitivity to local patterns (small \( k \)) and robustness to noise (large \( k \)).

### Step 3: Training the Model and Making Predictions  

Using the optimal \( k = 5 \), we train the kNN model and make predictions on the test set with the `kNN()` function:  


```r
kNN_predict = kNN(formula = formula, train = train_scaled, test = test_scaled, k = 5)
```

The `kNN()` function computes the distances between each test point and all training points, identifies the 5 nearest neighbors, and assigns the majority class among those neighbors as the predicted class for each test point.

### Step 4: Evaluating the Model  

Model evaluation is essential to assess how well the kNN algorithm performs on unseen data. Here, we display the confusion matrix for the test set predictions using the `conf.mat()` function:


```r
conf.mat(kNN_predict, actual_test)
   Setting levels: reference = "yes", case = "no"
          Actual
   Predict yes  no
       yes  54   7
       no   83 856
```


```
   Setting levels: reference = "yes", case = "no"
```

The confusion matrix summarizes the number of correct and incorrect predictions. In this case, the model achieves "54 + 856" correct predictions and "7 + 83" incorrect predictions. This provides insights into the model’s performance and highlights areas for improvement.

### Final Remarks {-}

Through this step-by-step implementation of the kNN algorithm, we demonstrated the importance of data preprocessing, parameter tuning, and proper evaluation. While kNN is simple and intuitive, its effectiveness relies heavily on these steps. For further evaluation metrics and performance analysis, we will explore these topics in the next chapter (Chapter \@ref(chapter-evaluation)).

## Summary  

In this chapter, we explored the k-Nearest Neighbors (kNN) algorithm, a simple yet effective method for solving classification problems. We began by revisiting the concept of classification and its real-world applications, highlighting the difference between binary and multi-class problems. We then delved into the mechanics of kNN, emphasizing its reliance on distance metrics to identify the most similar data points. Critical preprocessing steps, such as feature scaling and one-hot encoding, were discussed to ensure accurate and meaningful distance calculations. We also covered how to select an optimal \( k \) value and demonstrated the implementation of kNN using the **liver** package in R with the **churn** dataset. Through practical examples, we highlighted the importance of proper data preparation and parameter tuning for reliable and effective classification performance.

The simplicity and interpretability of kNN make it an excellent starting point for understanding classification and exploring dataset structure. However, the algorithm has notable limitations, including sensitivity to noise, computational inefficiency with large datasets, and the requirement for proper scaling and feature selection. These challenges make kNN less practical for large-scale applications, but it remains a valuable tool for small to medium-sized datasets and serves as a benchmark for evaluating more advanced algorithms.

While kNN is easy to understand and implement, its prediction speed and scalability constraints often make it unsuitable for modern, large-scale datasets. Nonetheless, it is a helpful baseline method and a stepping stone to more sophisticated techniques. In the upcoming chapters, we will explore advanced classification algorithms, such as Decision Trees, Random Forests, and Logistic Regression, which address the limitations of kNN and provide enhanced performance and scalability for a wide range of applications.

## Exercises

To do ...
