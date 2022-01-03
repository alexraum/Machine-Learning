### Homework 3: Naive Bayes and Neural Network Models

# import libraries
library(tidyverse)
library(DescTools)
library(glmnet)
library(InformationValue)
library(rpart)
library(ROCR)
library(e1071)
library(caret)
library(xgboost)
library(nnet)
library(NeuralNetTools)
library(iml)

# set the working directory
setwd("C:/Users/alexr/OneDrive/Documents/NCSU/MSA Program/Fall 2021/AA502/Machine Learning/Homework3_ML")

# read in the data sets
ins_t = read.csv("data/insurance_t.csv")
ins_v = read.csv("data/insurance_v.csv")

##################################################################################################################

# Part 0: Data Posturing, Cleaning, and Imputation

## Training Data

# Check for NAs
ins_t_nas = ins_t[colSums(is.na(ins_t)) > 0]

# Create missing categories
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

# Replace missing values in each column (now all continuous) with column median
for(i in 1:ncol(ins_t)) {
  ins_t[,i][is.na(ins_t[,i])] <- median(ins_t[,i], na.rm=TRUE)
}

## Validation Data

# Check for NAs
ins_v_nas = ins_v[colSums(is.na(ins_v)) > 0]

# Create missing categories
ins_v$CC[is.na(ins_v$CC)] = "M"
ins_v$INV[is.na(ins_v$INV)] = "M"
ins_v$CCPURC[is.na(ins_v$CCPURC)] = "M"

#Creating new NA indicator column for all 
ins_v$ACCTAGE_NA = (ifelse(is.na(ins_v$ACCTAGE), 1, 0))
ins_v$PHONE_NA = (ifelse(is.na(ins_v$PHONE), 1, 0))
ins_v$POS_NA = (ifelse(is.na(ins_v$POS), 1, 0))
ins_v$POSAMT_NA = (ifelse(is.na(ins_v$POSAMT), 1, 0))
ins_v$INVBAL_NA = (ifelse(is.na(ins_v$INVBAL), 1, 0))
ins_v$CCBAL_NA = (ifelse(is.na(ins_v$CCBAL), 1, 0))
ins_v$INCOME_NA = (ifelse(is.na(ins_v$INCOME), 1, 0))
ins_v$LORES_NA = (ifelse(is.na(ins_v$LORES), 1, 0))
ins_v$HMVAL_NA = (ifelse(is.na(ins_v$HMVAL), 1, 0))
ins_v$AGE_NA = (ifelse(is.na(ins_v$AGE), 1, 0))
ins_v$CRSCORE_NA = (ifelse(is.na(ins_v$CRSCORE), 1, 0))

# Replace missing values in each column (now all continuous) with column median from training data
ins_v$ACCTAGE[is.na(ins_v$ACCTAGE)] = median(ins_t$ACCTAGE)
ins_v$PHONE[is.na(ins_v$PHONE)] = median(ins_t$PHONE)
ins_v$POS[is.na(ins_v$POS)] = median(ins_t$POS)
ins_v$POSAMT[is.na(ins_v$POSAMT)] = median(ins_t$POSAMT)
ins_v$INVBAL[is.na(ins_v$INVBAL)] = median(ins_t$INVBAL)
ins_v$CCBAL[is.na(ins_v$CCBAL)] = median(ins_t$CCBAL)
ins_v$INCOME[is.na(ins_v$INCOME)] = median(ins_t$INCOME)
ins_v$LORES[is.na(ins_v$LORES)] = median(ins_t$LORES)
ins_v$HMVAL[is.na(ins_v$HMVAL)] = median(ins_t$HMVAL)
ins_v$AGE[is.na(ins_v$AGE)] = median(ins_t$AGE)
ins_v$CRSCORE[is.na(ins_v$CRSCORE)] = median(ins_t$CRSCORE)


##################################################################################################################

## Part 1: Build a Naive Bayes model and report details

tune_grid_nb <- expand.grid(
  usekernel = c(TRUE, FALSE),
  fL = c(0, 0.5, 1),
  adjust = c(0) # not sure about this
)

set.seed(12345)
nb_ins_caret <- caret::train(as.factor(INS) ~ ., data = ins_t, method = "nb", tuneGrid = tune_grid_nb, trControl = trainControl(method = 'cv', number = 10))

nb_ins <- naiveBayes(INS ~ ., data = ins_t, laplace = 0, usekernel = T)

# grab the predictions
preds <- predict(nb_ins, ins_t, type = "raw", threshold = 0.001)
ins_t$p_hat = preds[,2]

# plot the ROC curve
plotROC(ins_t$INS, ins_t$p_hat)

# select variables to include in the model
ins_t_subset <- ins_t %>%
  select()


##################################################################################################################

## Part 2: Build a Neural Network model and report details

# cast INS to a factor variable
ins_t$INS <- as.factor(ins_t$INS)
# store indices of remaining categorical variables
idx_cats <- c(2, 7, 8, 12, 14, 18, 20, 22, 24, 26, 27, 29, 30, 36, 38:49)
# cast categorical variables to factor variables
for (i in 1:length(idx_cats)) {
  ins_t[,idx_cats[i]] <- as.factor(ins_t[,idx_cats[i]])
}


# store indices of continuous variables
idx_conts <- c(1, 3, 5, 9, 13, 15, 17, 19, 21, 23, 25, 28, 31, 33, 35)
# check for normal distribution of continuous data
for (i in 1:length(idx_conts)) {
  qqnorm(ins_t[,idx_conts[i]])
  qqline(ins_t[,idx_conts[i]])
}

# scale continuous data where necessary (narrow range around 0) (use the scale function)

## Model 1: Untuned with all variables

# build the neural network (be sure to account for categorical response)
set.seed(12345)
# specify 5 hidden nodes, linout = F since we're doing classification
nnet_ins1 <- nnet(INS ~ ., data = ins_t, size = 5, linout = F, trace = F)

# grab the predictions
p_hat1 <- predict(nnet_ins1, ins_t, type = "raw")
# plot the ROC curve
plotROC(ins_t$INS, p_hat1)

## Model 2: Tune size and decay parameters

# create a grid to search
tune_grid_nnet <- expand.grid(
  .size = c(2, 3, 4, 5, 6, 7, 8), # number of hidden nodes
  .decay = c(0, 0.5, 1) # regularization parameter to prevent overfitting
)
# determine the optimum parameters
set.seed(12345)
nnet_ins_caret2 <- train(INS ~ ., data = ins_t, method = "nnet", tuneGrid = tune_grid_nnet,
                                 trControl = trainControl(method = 'cv', number = 10), trace = F)
# display the optimal parameters
nnet_ins_caret2$bestTune

# build the neural network with optimal parameters
set.seed(12345)
nnet_ins2 <- nnet(INS ~ ., data = ins_t, size = 5, decay = 0.5, linout = F, trace = F)

# grab the predictions
p_hat2 <- predict(nnet_ins2, ins_t, type = "raw")
# plot the ROC curve
plotROC(ins_t$INS, p_hat2)

## Model 3: Include variables selected by the best XGBoost model

# determine the optimum parameters
set.seed(12345)
nnet_ins_caret3 <- train(INS ~ scale(SAVBAL) + scale(DDABAL) + scale(CDBAL) +
                               as.factor(DDA) + as.factor(MM) + scale(MMBAL) +
                               scale(ACCTAGE) + scale(CHECKS) + scale(CCBAL) +
                               scale(IRABAL) + scale(ATMAMT) + scale(TELLER) +
                               as.factor(INV) + scale(INCOME) + scale(DEPAMT) +
                               scale(HMVAL), data = ins_t, method = "nnet", tuneGrid = tune_grid_nnet,
                         trControl = trainControl(method = 'cv', number = 10), trace = F)
# display the optimal parameters
nnet_ins_caret3$bestTune

# build the neural network with optimal parameters
set.seed(12345)
nnet_ins3 <- nnet(INS ~ scale(SAVBAL) + scale(DDABAL) + scale(CDBAL) +
                        as.factor(DDA) + as.factor(MM) + scale(MMBAL) +
                        scale(ACCTAGE) + scale(CHECKS) + scale(CCBAL) +
                        scale(IRABAL) + scale(ATMAMT) + scale(TELLER) +
                        as.factor(INV) + scale(INCOME) + scale(DEPAMT) +
                        scale(HMVAL), data = ins_t, size = 5, decay = 0.0, linout = F, trace = F)

# grab the predictions
p_hat3 <- predict(nnet_ins3, ins_t, type = "raw")
# plot the ROC curve
plotROC(ins_t$INS, p_hat3)

## Model 4: Include all variables (tuning parameters and scaling where appropriate)

# determine the optimum parameters
set.seed(12345)
nnet_ins_caret4 <- train(INS ~ scale(ACCTAGE) + as.factor(DDA) + scale(DDABAL) + scale(DEP) +
                               scale(DEPAMT) +  scale(CHECKS) + as.factor(DIRDEP) + as.factor(NSF) +
                               scale(NSFAMT) + scale(PHONE) + scale(TELLER) + as.factor(SAV) + 
                               scale(SAVBAL) + as.factor(ATM) + scale(ATMAMT) + scale(POS) + 
                               scale(POSAMT) + as.factor(CD) + scale(CDBAL) + as.factor(IRA) +
                               scale(IRABAL) + as.factor(INV) + scale(INVBAL) + as.factor(MM) +
                               scale(MMBAL) + as.factor(MMCRED) + as.factor(CC) + scale(CCBAL) + 
                               as.factor(CCPURC) + as.factor(SDB) + scale(INCOME) + scale(LORES) + 
                               scale(HMVAL) + scale(AGE) + scale(CRSCORE) + as.factor(INAREA) + 
                               as.factor(BRANCH) + as.factor(ACCTAGE_NA) + as.factor(PHONE_NA) + as.factor(POS_NA) + 
                               as.factor(POSAMT_NA) + as.factor(INVBAL_NA) + as.factor(CCBAL_NA) + as.factor(INCOME_NA) + 
                               as.factor(LORES_NA) + as.factor(HMVAL_NA) + as.factor(AGE_NA) + as.factor(CRSCORE_NA), 
                         data = ins_t, method = "nnet", tuneGrid = tune_grid_nnet,
                         trControl = trainControl(method = 'cv', number = 10), trace = F)
# display the optimal parameters
nnet_ins_caret4$bestTune

# build the neural network with optimal parameters
set.seed(12345)
nnet_ins4 <- nnet(INS ~ scale(ACCTAGE) + as.factor(DDA) + scale(DDABAL) + scale(DEP) +
                        scale(DEPAMT) +  scale(CHECKS) + as.factor(DIRDEP) + as.factor(NSF) +
                        scale(NSFAMT) + scale(PHONE) + scale(TELLER) + as.factor(SAV) + 
                        scale(SAVBAL) + as.factor(ATM) + scale(ATMAMT) + scale(POS) + 
                        scale(POSAMT) + as.factor(CD) + scale(CDBAL) + as.factor(IRA) +
                        scale(IRABAL) + as.factor(INV) + scale(INVBAL) + as.factor(MM) +
                        scale(MMBAL) + as.factor(MMCRED) + as.factor(CC) + scale(CCBAL) + 
                        as.factor(CCPURC) + as.factor(SDB) + scale(INCOME) + scale(LORES) + 
                        scale(HMVAL) + scale(AGE) + scale(CRSCORE) + as.factor(INAREA) + 
                        as.factor(BRANCH) + as.factor(ACCTAGE_NA) + as.factor(PHONE_NA) + as.factor(POS_NA) + 
                        as.factor(POSAMT_NA) + as.factor(INVBAL_NA) + as.factor(CCBAL_NA) + as.factor(INCOME_NA) + 
                        as.factor(LORES_NA) + as.factor(HMVAL_NA) + as.factor(AGE_NA) + as.factor(CRSCORE_NA), 
                  data = ins_t, size = 6, decay = 1.0, linout = F, trace = F)

# grab the predictions
p_hat4 <- predict(nnet_ins4, ins_t, type = "raw")
# plot the ROC curve
plotROC(ins_t$INS, p_hat4)

## Model 5 (Final Model): Include all scaled variables, 13 nodes, and weight of 0.5

# build the neural network with optimal parameters
set.seed(12345)
nnet_final <- nnet(INS ~ scale(ACCTAGE) + as.factor(DDA) + scale(DDABAL) + scale(DEP) +
                    scale(DEPAMT) +  scale(CHECKS) + as.factor(DIRDEP) + as.factor(NSF) +
                    scale(NSFAMT) + scale(PHONE) + scale(TELLER) + as.factor(SAV) + 
                    scale(SAVBAL) + as.factor(ATM) + scale(ATMAMT) + scale(POS) + 
                    scale(POSAMT) + as.factor(CD) + scale(CDBAL) + as.factor(IRA) +
                    scale(IRABAL) + as.factor(INV) + scale(INVBAL) + as.factor(MM) +
                    scale(MMBAL) + as.factor(MMCRED) + as.factor(CC) + scale(CCBAL) + 
                    as.factor(CCPURC) + as.factor(SDB) + scale(INCOME) + scale(LORES) + 
                    scale(HMVAL) + scale(AGE) + scale(CRSCORE) + as.factor(INAREA) + 
                    as.factor(BRANCH) + as.factor(ACCTAGE_NA) + as.factor(PHONE_NA) + as.factor(POS_NA) + 
                    as.factor(POSAMT_NA) + as.factor(INVBAL_NA) + as.factor(CCBAL_NA) + as.factor(INCOME_NA) + 
                    as.factor(LORES_NA) + as.factor(HMVAL_NA) + as.factor(AGE_NA) + as.factor(CRSCORE_NA), 
                  data = ins_t, size = 13, decay = 0.5, linout = F, trace = F)

# grab the predictions
p_hat_final <- predict(nnet_final, ins_t, type = "raw")
# plot the ROC curve
plotROC(ins_t$INS, p_hat_final)


##################################################################################################################

## Part 3: Build an XGBoost model and report details

# separate predictors and response
train_x = model.matrix(INS ~., data=ins_t)[,-1]
train_y = ins_t$INS

# final model: all variables, nround=11, eta=0.25, max_depth=5, subsample=1
set.seed(12345)
# fit model to the training data
xgb_final = xgboost(data=train_x, label=train_y, subsample=1, nrounds=11, nfold=10, eta=0.25, max_depth=5)

# grab the predictions
preds_final = predict(xgb_final, xgb.DMatrix(train_x, label=ins_t$INS))
# report the area under the ROC curve
plotROC(ins_t$INS, preds_final)


##################################################################################################################

## Part 4: Run Final Model on validation data and report the accuracy

# Neural Network Model

# grab the predictions and compare to validation data
preds_nnet <- predict(nnet_final, ins_v, type = "raw")
# plot the ROC curve from validation data
plotROC(ins_v$INS, preds_nnet)

# XGBoost Model

# separate predictors and response
train_x_val = model.matrix(INS ~., data=ins_v)[,-1]
train_y_val = ins_v$INS

# grab the predictions
preds_xgb = predict(xgb_final, xgb.DMatrix(train_x_val, label=ins_v$INS))
# report the area under the ROC curve
plotROC(ins_v$INS, preds_xgb)

# generate a partial dependence plot
partial(xgb_final, pred.var = "ACCTAGE", plot = TRUE, rug = TRUE, alpha = 0.1,
        plot.engine = "lattice", train = train_x_val)

##################################################################################################################

## Part 5: Global Interpretation (XGBoost Model)

# partial dependence plot
set.seed(12345)
pred_xgb <- Predictor$new(xgb_final, data = as.data.frame(train_x), y = train_y, class = "classification",
                          type = "prob", predict.fun=function(model, newdata){
                            newData_x=xgb.DMatrix(as.matrix(newdata))
                            results=predict(model, newData_x)
                            return(results)
                          })

pdp_plot <- FeatureEffects$new(pred_xgb, method = "pdp")
pdp_plot$plot(c("ACCTAGE"))
