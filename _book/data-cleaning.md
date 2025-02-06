# Data Preparation {#data-preparation}



The quality of input data has a huge impact on the Data Science Methodology. 
For data analytics purposes, database values must undergo data cleaning and data transformation.
Basically, we want to minimize *garbage in garbage out*.    
Thus, it is important to prepare and learn more about the data. This requires cleaning so-called *messy* data, eliminating unnecessary data, and if it's needed recording the (part of) data for the modeling part.

In general, data cleaning is the most time consuming part of the date science methodology. 
Effort for data preparation and cleaning ranges around 10\%-60\% of data analysis process – depending on the dataset.

## `diamonds` dataset for data cleaning

We represent how to preform the steps one and two of the Data Science Methodology using the `diamonds` dataset. This dataset is available in [**ggplot2**](https://CRAN.R-project.org/package=ggplot2) package and contains information about \~$54,000$ diamonds, including the `price`, `carat`, `color`, `clarity`, `cut`, and dimensions of each diamond.  

In general, we could import the Dataset sheet from our personal computer or an online source into **R**, by using the `read.csv()` function. But, here, since the *diamonds* dataset is available in the **R** package "*ggplot2*", we import the *diamonds* dataset in **R** as follows:

```r
data(diamonds) # loads "diamonds" data in your RStudio environment
```

To see the overview of the dataset in **R**, we could use the following functions: 

* `str()`  to see a compact display of the structure of the data. 
* `View()` to see spreadsheet-style data. 
* `head()` to see the first part of the data (first 6-rows of data).
* `summary()` to see the summary of each variable.

Here we use the `str()` function to report the structure of the *diamonds* dataset as follows

```r
str(diamonds)   
   tibble [53,940 × 10] (S3: tbl_df/tbl/data.frame)
    $ carat  : num [1:53940] 0.23 0.21 0.23 0.29 0.31 0.24 0.24 0.26 0.22 0.23 ...
    $ cut    : Ord.factor w/ 5 levels "Fair"<"Good"<..: 5 4 2 4 2 3 3 3 1 3 ...
    $ color  : Ord.factor w/ 7 levels "D"<"E"<"F"<"G"<..: 2 2 2 6 7 7 6 5 2 5 ...
    $ clarity: Ord.factor w/ 8 levels "I1"<"SI2"<"SI1"<..: 2 3 5 4 2 6 7 3 4 5 ...
    $ depth  : num [1:53940] 61.5 59.8 56.9 62.4 63.3 62.8 62.3 61.9 65.1 59.4 ...
    $ table  : num [1:53940] 55 61 65 58 58 57 57 55 61 61 ...
    $ price  : int [1:53940] 326 326 327 334 335 336 336 337 337 338 ...
    $ x      : num [1:53940] 3.95 3.89 4.05 4.2 4.34 3.94 3.95 4.07 3.87 4 ...
    $ y      : num [1:53940] 3.98 3.84 4.07 4.23 4.35 3.96 3.98 4.11 3.78 4.05 ...
    $ z      : num [1:53940] 2.43 2.31 2.31 2.63 2.75 2.48 2.47 2.53 2.49 2.39 ...
```

It shows the dataset has 53940 observations and 10 variables where:

* `price`: price in US dollars (\$326–\$18,823).
* `carat`: weight of the diamond (0.2–5.01).
* `cut`: quality of the cut (Fair, Good, Very Good, Premium, Ideal).
* `color`: diamond color, from D (best) to J (worst).
* `clarity`: a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)).
* `x`: length in mm (0–10.74).
* `y`: width in mm (0–58.9).
* `z`: depth in mm (0–31.8).
* `depth`: total depth percentage = `z` / mean(`x`, `y`) = 2 * `z` / (`x` + `y`).

To see the first part of the data

```r
head(diamonds)   
   # A tibble: 6 × 10
     carat cut       color clarity depth table price     x     y     z
     <dbl> <ord>     <ord> <ord>   <dbl> <dbl> <int> <dbl> <dbl> <dbl>
   1  0.23 Ideal     E     SI2      61.5    55   326  3.95  3.98  2.43
   2  0.21 Premium   E     SI1      59.8    61   326  3.89  3.84  2.31
   3  0.23 Good      E     VS1      56.9    65   327  4.05  4.07  2.31
   4  0.29 Premium   I     VS2      62.4    58   334  4.2   4.23  2.63
   5  0.31 Good      J     SI2      63.3    58   335  4.34  4.35  2.75
   6  0.24 Very Good J     VVS2     62.8    57   336  3.94  3.96  2.48
```

After loading dataset in the R, we need to know which type of features (variables, attributes, or fields) we have in our dataset. 

In general, the type of variables are:

1. Quantitative (or numerical) variables: which are represented by numbers, and are divided into three broad divisions.
    + Continuous variables: which are represented by entities get a distinct score. For example, in the diamonds dataset, the length of diamonds (variable `x`) as well as variables `y`, `z`, and `depth` are continuous.
    + Discrete (or count) variables: are countable in a finite amount of time. Some examples of such variables are age (in years) and number of customer who churn the company.

2. Qualitative (or categorical) variables: which describe data that fits into categories and they are not numerical. They may be divided into three broad divisions. 
    + Ordinal variables: are ordered categories and the distances between the categories are not known. For example, cloth sizes S, M, L, and XL. As an another example, in the diamonds dataset variable `clarity` is ordinal.
    + Binary variables: are variables which only take two values. For example, having a tattoo (yes/no)
    + Nominal variables: are categorical variables which do not have intrinsic ordering to the categories. For example, gender (female, male, LGBT) or color of diamonds in the diamonds dataset.

## Outliers

Outlines are unusual/extreme values that significantly differ from other observations.
Outliers may be due to variability in the measurement or may be due to data entry errors.
Outliers can potentially have serious problems in statistical analyses.
Thus, it’s important to detect potential outliers in the dataset and deal with them in an appropriate manner.
One easy way to detect outliers is by data visualization methods.

### Identify outliers by boxplot

We can detect outliers by using Boxplot which represents the distribution of the feature. Boxplot is suitable for the numerical features. 

<div class="figure" style="text-align: center">
<img src="images/boxplot.png" alt="Box plot is useful to report the summay for the numerical features/variables in the dataset." width="50%" />
<p class="caption">(\#fig:boxplot)Box plot is useful to report the summay for the numerical features/variables in the dataset.</p>
</div>

For example, the Boxplot for the variable `y` (width of diamonds) from the `diamonds` dataset is

```r
ggplot(data = diamonds) +
    geom_boxplot(mapping = aes(y = y))
```

<img src="data-cleaning_files/figure-html/unnamed-chunk-5-1.png" width="672" style="display: block; margin: auto;" />

The `y` variable measures one of the three dimensions of these diamonds, in mm. We know that diamonds can’t have a width of 0mm, so these values must be incorrect. We might also suspect that measurements of 32mm and 59mm are implausible.

### Identify outliers by histogram

Another way to detect outliers is by using histogram which represents the distribution of the feature. 
For example, the histogram for the variable `y` (width of diamonds) from the `diamonds` dataset is

```r
ggplot(data = diamonds) +
  geom_histogram(aes(x = y), binwidth = 0.5, color = 'blue', 
                 fill = "lightblue")
```

<img src="data-cleaning_files/figure-html/unnamed-chunk-6-1.png" width="672" style="display: block; margin: auto;" />

There are so many observations in the common bins that the rare bins are so short that you cann't see them (although maybe if you stare intently at 0 you'll spot something). To make it easy to see the unusual values, we need to zoom in to small values of the y-axis:


```r
ggplot(data = diamonds) +
  geom_histogram(mapping = aes(x = y), binwidth = 0.5, 
                 color = 'blue', fill = "lightblue") + 
    coord_cartesian(ylim = c(0, 30))
```

<img src="data-cleaning_files/figure-html/unnamed-chunk-7-1.png" width="672" style="display: block; margin: auto;" />

### Identify outliers by scator plot

Another way to detect outliers is by using scatter plot which represents the point distribution between two numerical features. 
For example, the scatter plot for variable `y` vs `price` is

```r
ggplot(data = diamonds, mapping = aes(x = y, y = price)) + 
    geom_point(colour = 'blue')
```

<img src="data-cleaning_files/figure-html/unnamed-chunk-8-1.png" width="672" style="display: block; margin: auto;" />

We might also suspect that measurements of 32mm and 59mm are implausible: those diamonds are over an inch long, but don’t cost hundreds of thousands of dollars!

## Handling Outliers

After detecting outliers, we should decide what to do with them. In this case we have two options: 

1. Treat outliers as missing values, which we recommend;
2. Remove outliers from the dataset, which we do *not* recommend it.

For replacing outliers with missing values, one easy way to do it is to use `mutate()` function which is from the **plyr** package; With this function, we can replace the variable with a modified copy. You can use the `ifelse()` function to replace unusual values with `NA`:


```r
diamonds_2 = mutate(diamonds, y = ifelse(y ==  0 | y > 30, NA, y)) 
```

[`ifelse()`](https://rdrr.io/r/base/ifelse.html) has three arguments. The first argument test should be a logical vector. The result will contain the value of the second argument, `yes`, when test is `TRUE`, and the value of the third argument, `no`, when it is `FALSE`. 
Note, in R, we show missing values with NA (Not Available).

To see the summary of the variable `y` with missing values:

```r
summary(diamonds_2$y)
      Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
     3.680   4.720   5.710   5.734   6.540  10.540       9
```
which show that we have 9 unusual values which are replaced by `NA`.


For the case of removing outliers for the dataset, we can do it by using `filter()` function from **dplyr** package

```r
diamonds_2 = filter(diamonds, between(y, 3, 20))
```


```r
ggplot(data = diamonds_2, mapping = aes(x = y, y = price)) + 
    geom_point(colour = 'blue')
```

<img src="data-cleaning_files/figure-html/unnamed-chunk-12-1.png" width="672" style="display: block; margin: auto;" />

It’s good practice to repeat your analysis with and without the outliers. If they have minimal effect on the results, and you can’t figure out why they’re there, it’s reasonable to replace them with missing values, and move on. However, if they have a substantial effect on your results, you shouldn’t drop them without justification. You’ll need to figure out what caused them (e.g. a data entry error) and disclose that you removed them in your write-up.

## Missing Values

Missing values pose problems to data analysis methods. We have two option to deal with the missing values:

1. Impute the missing values which we recommend.
2. Delete Records Containing Missing Values which we do *not* recommend; this option is similar to the precious section for removing outliers.

To impute the missing values with *random* values (which is proportional to categories' records) by using the function `impute()` from R package **Hmisc** 

```r
diamonds_2$y = impute(diamonds_2$y, 'random')
```

To see the summary of the variable `y`:

```r
summary(diamonds_2$y)
      Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
     3.680   4.720   5.710   5.734   6.540  10.540
```
  
## Data Transformation

In Data Science Methodology, before applying any machine learning algorithms, sometimes, we need to convert the raw data into a format or structure that would be more suitable for modeling. Fer example, in “diamonds” dataset, variable `carat` range between (0.2, 5) and variable `price` range between (326, 18823). Some some machine learning algorithms (i.e. the k-nearest neighbor algorithm) are adversely affected by differences in variable ranges. Variables with greater ranges tend to have larger influence on the data model’s results. Thus, numeric field values should be normalized.
Two of the prevalent methods will be reviewed

* min-max transformation.
* Z-score transformation also know as an Z-score Standardization

## How to Reexpress Categorical Field Values 



