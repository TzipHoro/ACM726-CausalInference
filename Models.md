Data Summary
================

- [Preprocess Data](#preprocess-data)
- [Logistic Regression](#logistic-regression)
- [Random Forests](#random-forests)
- [BART](#bart)

``` r
pacman::p_load(dplyr,
               fastDummies,
               caret,
               bartMachine,
               stargazer
)

source('ROC.R')
source('RFTree.R')

as_latex <- FALSE
set.seed(10)
```

# Preprocess Data

``` r
df <- read.csv(file = file.path('data', 'heart.csv'))

# create dummy variables
df$thal <- as.factor(df$thal)
df$cp <- as.factor(df$cp)
df$restecg <- as.factor(df$restecg)
df$slope <- as.factor(df$slope)
df$ca <- as.factor(df$ca)
df <- dummy_cols(df) %>%
  select('age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'cp_1', 'cp_2', 'cp_3', 'restecg_1',
         'restecg_2', 'slope_1', 'slope_2', 'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_1', 'thal_2', 'thal_3', 'target')

# rename columns
colnames(df) <- c('age', 'sex', 'resting blood pressure', 'serum cholesterol', 'fasting blood sugar > 120 mg/dl',
                 'maximum heart rate achieved', 'exercise induced angina', 'oldpeak', 'chest pain = 1',
                 'chest pain = 2', 'chest pain = 3', 'resting electrocardiograph = 1', 'resting electrocardiograph = 2',
                 'slope = 1', 'slope = 2', 'major vessels colored = 1', 'major vessels colored = 2',
                 'major vessels colored = 3', 'major vessels colored = 4', 'thalassemia = 1', 'thalassemia = 2',
                 'thalassemia = 3', 'target')

# train/test split
smp_size <- floor(0.6 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

df_train <- df[train_ind, ]
df_test <- df[-train_ind, ]
X_test <- select(df_test, -target)
y_test <- df_test$target

ctrl <- trainControl(method = 'cv', number = 5)
```

# Logistic Regression

``` r
lr_mod <- train(factor(target) ~ ., data = df_train, method = 'glm', family = binomial(link = "logit"), trControl = ctrl)

if (as_latex == TRUE) {
  summary(lr_mod)$coefficients %>%
          stargazer()
} else {
  data.frame(summary(lr_mod)$coefficients)
}
```

    ##                                             Estimate  Std..Error    z.value
    ## (Intercept)                             -0.641409161 2.510612102 -0.2554792
    ## age                                      0.022202857 0.018603866  1.1934539
    ## sex                                     -1.748565190 0.397958373 -4.3938394
    ## `\\`resting blood pressure\\``          -0.019840038 0.008549714 -2.3205500
    ## `\\`serum cholesterol\\``               -0.005259987 0.002888466 -1.8210316
    ## `\\`fasting blood sugar > 120 mg/dl\\``  0.296628496 0.409241424  0.7248252
    ## `\\`maximum heart rate achieved\\``      0.022726196 0.009032733  2.5159822
    ## `\\`exercise induced angina\\``         -0.795493587 0.329638746 -2.4132284
    ## oldpeak                                 -0.348018562 0.180994245 -1.9228156
    ## `\\`chest pain = 1\\``                   0.961956156 0.395997342  2.4291985
    ## `\\`chest pain = 2\\``                   2.028971416 0.365379944  5.5530454
    ## `\\`chest pain = 3\\``                   2.558827238 0.579411262  4.4162539
    ## `\\`resting electrocardiograph = 1\\``   0.150976047 0.281240594  0.5368217
    ## `\\`resting electrocardiograph = 2\\``  -1.160973411 2.173117766 -0.5342432
    ## `\\`slope = 1\\``                       -0.619582236 0.567821889 -1.0911560
    ## `\\`slope = 2\\``                        0.673433250 0.601171582  1.1202014
    ## `\\`major vessels colored = 1\\``       -2.443325796 0.359945406 -6.7880455
    ## `\\`major vessels colored = 2\\``       -3.271500644 0.566718354 -5.7727099
    ## `\\`major vessels colored = 3\\``       -2.245028195 0.810760680 -2.7690393
    ## `\\`major vessels colored = 4\\``        1.022481415 1.355826139  0.7541390
    ## `\\`thalassemia = 1\\``                  2.244612331 1.744048597  1.2870125
    ## `\\`thalassemia = 2\\``                  2.269069001 1.682617393  1.3485353
    ## `\\`thalassemia = 3\\``                  0.811895356 1.682795785  0.4824681
    ##                                             Pr...z..
    ## (Intercept)                             7.983530e-01
    ## age                                     2.326916e-01
    ## sex                                     1.113661e-05
    ## `\\`resting blood pressure\\``          2.031114e-02
    ## `\\`serum cholesterol\\``               6.860206e-02
    ## `\\`fasting blood sugar > 120 mg/dl\\`` 4.685593e-01
    ## `\\`maximum heart rate achieved\\``     1.187012e-02
    ## `\\`exercise induced angina\\``         1.581191e-02
    ## oldpeak                                 5.450321e-02
    ## `\\`chest pain = 1\\``                  1.513224e-02
    ## `\\`chest pain = 2\\``                  2.807351e-08
    ## `\\`chest pain = 3\\``                  1.004261e-05
    ## `\\`resting electrocardiograph = 1\\``  5.913908e-01
    ## `\\`resting electrocardiograph = 2\\``  5.931733e-01
    ## `\\`slope = 1\\``                       2.752043e-01
    ## `\\`slope = 2\\``                       2.626279e-01
    ## `\\`major vessels colored = 1\\``       1.136628e-11
    ## `\\`major vessels colored = 2\\``       7.800669e-09
    ## `\\`major vessels colored = 3\\``       5.622185e-03
    ## `\\`major vessels colored = 4\\``       4.507657e-01
    ## `\\`thalassemia = 1\\``                 1.980899e-01
    ## `\\`thalassemia = 2\\``                 1.774863e-01
    ## `\\`thalassemia = 3\\``                 6.294734e-01

``` r
p_pred <- predict(lr_mod, X_test, type = 'prob')
roc <- ROCMetrics$new(y_true = y_test, p_pred = p_pred$`1`)
thresholds <- roc$threshold.matrix(0.001)
roc$roc_plot(thresholds, title_prefix = 'Logistic Regression', file_name = 'plots/lr_roc.png')
```

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
p_th <- row.names(thresholds)[which(thresholds$`Balanced Accuracy` == max(thresholds$`Balanced Accuracy`))]
paste0('Optimal threshold: ', max(p_th))
```

    ## [1] "Optimal threshold: 0.621"

# Random Forests

``` r
rf_mod <- train(factor(target) ~ ., data = df_train, method = 'rf', trControl = ctrl)
rf_mod$finalModel
```

    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 12
    ## 
    ##         OOB estimate of  error rate: 2.76%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 297   9  0.02941176
    ## 1   8 301  0.02588997

``` r
tree_df <- randomForest::getTree(rf_mod$finalModel, labelVar=TRUE) %>%
        head(15)
if (as_latex == TRUE) {
  stargazer(tree_df)
} else {
  tree_df
}
```

    ##    left daughter right daughter                     split var split point
    ## 1              2              3             `thalassemia = 3`         0.5
    ## 2              4              5 `maximum heart rate achieved`       113.5
    ## 3              6              7      `resting blood pressure`       109.0
    ## 4              0              0                          <NA>         0.0
    ## 5              8              9                       oldpeak         2.5
    ## 6             10             11 `maximum heart rate achieved`       155.0
    ## 7             12             13              `chest pain = 3`         0.5
    ## 8             14             15                           age        55.5
    ## 9             16             17                           age        40.5
    ## 10             0              0                          <NA>         0.0
    ## 11            18             19              `chest pain = 1`         0.5
    ## 12            20             21              `chest pain = 2`         0.5
    ## 13             0              0                          <NA>         0.0
    ## 14            22             23   `major vessels colored = 3`         0.5
    ## 15            24             25           `serum cholesterol`       248.5
    ##    status prediction
    ## 1       1       <NA>
    ## 2       1       <NA>
    ## 3       1       <NA>
    ## 4      -1          0
    ## 5       1       <NA>
    ## 6       1       <NA>
    ## 7       1       <NA>
    ## 8       1       <NA>
    ## 9       1       <NA>
    ## 10     -1          1
    ## 11      1       <NA>
    ## 12      1       <NA>
    ## 13     -1          1
    ## 14      1       <NA>
    ## 15      1       <NA>

``` r
tree_func(rf_mod$finalModel, 1, file_name = 'plots/rf_tree.png')
```

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
p_pred <- predict(rf_mod, X_test, type = 'prob')
roc <- ROCMetrics$new(y_true = y_test, p_pred = p_pred$`1`)
thresholds <- roc$threshold.matrix(0.001)
roc$roc_plot(thresholds, title_prefix = 'Random Forests', file_name = 'plots/rf_roc.png')
```

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

``` r
p_th <- row.names(thresholds)[which(thresholds$`Balanced Accuracy` == max(thresholds$`Balanced Accuracy`))]
paste0('Optimal threshold: ', max(p_th))
```

    ## [1] "Optimal threshold: 0.44"

# BART

``` r
set_bart_machine_num_cores(4)
```

    ## bartMachine now using 4 cores.

``` r
X_train <- select(df_train, -target)
y_train <- df_train$target

bart_mod <- bartMachine(X = X_train, y = factor(y_train))
```

    ## bartMachine initializing with 50 trees...
    ## bartMachine vars checked...
    ## bartMachine java init...
    ## bartMachine factors created...
    ## bartMachine before preprocess...
    ## bartMachine after preprocess... 22 total features...
    ## bartMachine training data finalized...
    ## Now building bartMachine for classification where "0" is considered the target level...
    ## evaluating in sample data...done

``` r
bart_mod
```

    ## bartMachine v1.3.3.1 for classification
    ## 
    ## training data size: n = 615 and p = 22 
    ## built in 1.6 secs on 4 cores, 50 trees, 250 burn-in and 1000 post. samples
    ## 
    ## confusion matrix:
    ## 
    ##            predicted 0 predicted 1 model errors
    ## actual 0       278.000      28.000        0.092
    ## actual 1        16.000     293.000        0.052
    ## use errors       0.054       0.087        0.072

``` r
# partial dependency plots
for (i in 1:ncol(X_train)) {
  pd_plot(bart_mod, i)
}
```

    ## ...........

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-2.png)<!-- -->

    ## ...........

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-3.png)<!-- -->

    ## ...........

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-4.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-5.png)<!-- -->

    ## ...........

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-6.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-7.png)<!-- -->

    ## ........

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-8.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-9.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-10.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-11.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-12.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-13.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-14.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-15.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-16.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-17.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-18.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-19.png)<!-- -->

    ## ..

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-8-20.png)<!-- -->

``` r
p_pred <- 1 - predict(bart_mod, X_test, type = 'prob')
```

    ## predicting probabilities where "0" is considered the target level...

``` r
roc <- ROCMetrics$new(y_true = y_test, p_pred = p_pred)
thresholds <- roc$threshold.matrix(0.001)
roc$roc_plot(thresholds, title_prefix = 'BART', file_name = 'plots/bart_roc.png')
```

![](C:\Users\tzipo\DOCUME~1\JHU\THEORY~2\ACM726~1\MODELS~1/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
p_th <- row.names(thresholds)[which(thresholds$`Balanced Accuracy` == max(thresholds$`Balanced Accuracy`))]
paste0('Optimal threshold: ', max(p_th))
```

    ## [1] "Optimal threshold: 0.507"
