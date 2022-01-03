### Tree-Based Models

# import libraries
library(tidyverse)
library(DescTools)
library(glmnet)
library(earth)
library(InformationValue)
library(rpart)
library(randomForest)
library(ROCR)
library(xgboost)

# set the working directory
setwd("C:/Users/alexr/OneDrive/Documents/NCSU/MSA Program/Fall 2021/AA502/Machine Learning/Homework2_ML")

# read in the data set
ins_t = read.csv("insurance_t.csv")

##################################################################################################################

## Part 0: Data Cleaning

# Check for NAs
ins_nas = ins_t[colSums(is.na(ins_t)) > 0]

# Create missing category categorical
ins_t$CC[is.na(ins_t$CC)] = "M"
ins_t$INV[is.na(ins_t$INV)] = "M"
ins_t$CCPURC[is.na(ins_t$CCPURC)] = "M"

#Creating new NA indicator column for all 
ins_t$ACCTAGE_NA = (ifelse(is.na(ins_t$ACCTAGE), 1, 0))
ins_t$PHONE_NA = (ifelse(is.na(ins_t$PHONE), 1, 0))
ins_t$POS_NA = (ifelse(is.na(ins_t$POS), 1, 0))
ins_t$POSAMT_NA = (ifelse(is.na(ins_t$POSAMT), 1, 0))
ins_t$INVBAL_NA = (ifelse(is.na(ins_t$INVBAL), 1, 0))
ins_t$CCBAL_NA = (ifelse(is.na(ins_t$CCBAL), 1, 0))
ins_t$INCOME_NA = (ifelse(is.na(ins_t$INCOME), 1, 0))
ins_t$LORES_NA = (ifelse(is.na(ins_t$LORES), 1, 0))
ins_t$HMVAL_NA = (ifelse(is.na(ins_t$HMVAL), 1, 0))
ins_t$AGE_NA = (ifelse(is.na(ins_t$AGE), 1, 0))
ins_t$CRSCORE_NA = (ifelse(is.na(ins_t$CRSCORE), 1, 0))

# Using median imputation only for continuous
# Replace missing values in each column (now all continuous) with column median
for(i in 1:ncol(ins_t)) {
  ins_t[ , i][is.na(ins_t[ , i])] <- median(ins_t[ , i], na.rm=TRUE)
}

##################################################################################################################

## Part 1: Build a Random Forest model and report details

# examine relative counts
table(ins_t$INS)

# ensure we have a data frame
ins_t <- as.data.frame(ins_t)

# ensure INS is a factor variable
ins_t$INS <- as.factor(ins_t$INS)

# select variables to include in the random forest model
#rf_vars <- ins_t %>%
#  select(SAVBAL, CDBAL,
#         DDA, DDABAL,
#         MMBAL, ACCTAGE,
#         CHECKS, TELLER,
#         ATMAMT, INV,
#         CC, CCBAL,
#         BRANCH, IRABAL,
#         DEP, INS)

# set the seed for reproducibility
set.seed(12345)
# build the random forest model
rf_ins <- randomForest(INS ~ ., data = ins_t, ntree = 500, importance = T)

# examine error vs. number of trees
plot(rf_ins, main = "Number of Trees Compared to MSE")
# examine variable importance
varImpPlot(rf_ins, sort = T, n.var = 20, main = "Top 20 Variable Importance")
# display a list of important variables
importance(rf_ins)

# tune the random forest mtry value (number of variables to consider at each split)
set.seed(12345)
tuneRF(x = ins_t[,-37], y = ins_t[,37],
       plot = T, ntreeTry = 500, stepFactor = 0.5) # tune with 500 trees

# 6 variables appear to be the optimum number of variables to split on

# set the seed for reproducibility
set.seed(12345)
# tune the model parameters and fit another Random Forest model
rf_ins2 <- randomForest(INS ~ ., data = ins_t, ntree = 200, mtry = 6, importance = T)

# examine variable importance
varImpPlot(rf_ins2, sort = T, n.var = 20, main = "Top 20 Variable Importance")
# display a list of important variables
importance(rf_ins2)

# include a random variable to determine variable selection
ins_t$random <- rnorm(nrow(ins_t))
# build another Random Forest that includes the random variable
set.seed(12345)
rf_ins_random_var <- randomForest(INS ~ ., data = ins_t, ntree = 200, mtry = 7, importance = T)

# look for variables with lower importance than the random variable
varImpPlot(rf_ins_random_var, sort = T,
           n.var = 40, main = "Look for Variables with Lower Importance than Random Variable")

# grab the predictions
preds <- predict(rf_ins2, type = "prob")
ins_t$p_hat = preds[,2]

# plot the ROC curve
plotROC(ins_t$INS, ins_t$p_hat)

# plot another ROC curve
pred_rf = prediction(ins_t$p_hat, ins_t$INS)
perf_rf = performance(pred_rf, measure = "tpr", x.measure = "fpr")
plot(perf_rf, lwd = 1,  main = 'ROC Curve for RF Model 1')
abline(a = 0, b = 1, lty = 3)


##################################################################################################################

## Part 2: Build an XGBoost model and report details

# separate training and test data
train_x <- model.matrix(INS ~ ., data = ins_t)[,-1]
train_y <- ins_t$INS

## Step 0: Build an initial model

# build an XGBoost model with all variables
set.seed(12345)
xgb_ins <- xgboost(data = train_x, label = train_y, objective = "binary:logistic", subsample = 0.5, nrounds = 100)

# grab the predictions
preds = predict(xgb_ins, xgb.DMatrix(train_x, label=ins_t$INS))
ins_t$p_hat = preds

# report the area under the ROC curve
plotROC(ins_t$INS, ins_t$p_hat)


## Step 1: Tune the model parameters

# use cross validation to determine number of rounds to go
set.seed(12345)
xgbcv_ins <- xgb.cv(data = train_x, label = train_y, objective = "binary:logistic", subsample = 0.5, nrounds = 100, nfold = 10)

# error is minimized when nrounds = 12 (1 model and 11 error models)

# use tune_grid function to determine eta and max_depth
tune_grid_ins <- expand.grid(
  nrounds = 12,
  eta = c(0.1, 0.15, 0.2, 0.25, 0.3), # weights of error models
  max_depth = c(1:10),
  gamma = c(0),
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = c(0.25, 0.5, 0.75, 1) # what proportion of the data to give each tree
)

# grid search for optimal parameters
xgb_ins_caret <- train(x = train_x, y = train_y,
                       method = "xgbTree",
                       tuneGrid = tune_grid_ins,
                       trControl = trainControl(method = 'cv', number = 10))

# plot the error
plot(xgb_ins_caret)

# optimal parameters appear to be: eta = 0.3, depth = 5, subsample = 1.00, nrounds = 11


## Step 2: Perform variable selection

# fit a new model with optimal parameters
xgb_ins_best <- xgboost(data = train_x, label = train_y, subsample = 1, nrounds = 11, eta = 0.3, max_depth = 5) 

# examine variable importance
xgb.ggplot.importance(xgb.importance(feature_names = colnames(train_x), model = xgb_ins_best))

# Add a random variable
ins_t$random = rnorm(nrow(ins_t))

# create new training data that includes random variable
train_x <- model.matrix(INS ~ ., data = ins_t)[,-1]
train_y <- ins_t$INS

# fit a model that includes random variable
set.seed(12345)
xgb_ins_rand <- xgboost(data = train_x, label = train_y, subsample = 1, nrounds = 11, eta = 0.3, max_depth = 5)

# examine variable importance
xgb.ggplot.importance(xgb.importance(feature_names = colnames(train_x), model = xgb_ins_rand))

# variables flagged as more important than random are: SAVBAL, DDABAL, CDBAL, MM, DDA, ACCTAGE, MMBAL,
#                                                    : CHECKS, DEPAMT, AGE, ATMAMT, CCBAL

# select final variables
ins_t_final <- ins_t %>%
  dplyr::select(INS, SAVBAL, DDABAL, CDBAL, MM, DDA, ACCTAGE,
         MMBAL, CHECKS, ATMAMT, DEPAMT, CCBAL, AGE)

# create new training data that includes random variable
train_x_final <- model.matrix(INS ~ ., data = ins_t_final)[,-1]
train_y_final <- ins_t_final$INS

# exclude variables of lesser importance from final model
xgb_ins_final <- xgboost(data = train_x_final, label = train_y_final, subsample = 1, nrounds = 11, eta = 0.3, max_depth = 5)

# grab the predictions
preds_final = predict(xgb_ins_final, xgb.DMatrix(train_x_final, label=ins_t_final$INS))
ins_t_final$p_hat = preds_final

# report the area under the ROC curve
plotROC(ins_t_final$INS, ins_t_final$p_hat)
