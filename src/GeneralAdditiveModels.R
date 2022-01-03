### Homework 1: Machine Learning (Alex Raum)

# import libraries
library(tidyverse)
library(DescTools)
library(glmnet)
library(earth)
library(InformationValue)
library(rpart)

# set the working directory
setwd("C:/Users/alexr/OneDrive/Documents/NCSU/MSA Program/Fall 2021/AA502/Machine Learning/Homework1_ML")

# read in the data set
ins_t = read.csv("insurance_t.csv")

##################################################################################################################

## Part 0: Imputing missing values

# distinguish continuous from categorical by number of levels
names = colnames(ins_t)
for (i in 1:length(names)) {
  print(names[i])
  print(length(unique(ins_t[,names[i]])))
}

# determine number of missing values by column
colSums(is.na(ins_t))

# store names of continuous columns
cont_vars = c("ACCTAGE", "DDABAL", "DEPAMT", "CHECKS", "NSFAMT", "TELLER",
              "SAVBAL", "ATMAMT", "POS", "POSAMT", "CDBAL", "IRABAL",
              "INVBAL", "MMBAL", "CCBAL", "INCOME", "LORES", "HMVAL",
              "AGE", "CRSCORE")

# store names of continuous NA indicators
cont_vars_na = c("ACCTAGE_NA", "DDABAL_NA", "DEPAMT_NA", "CHECKS_NA", "NSFAMT_NA", "TELLER_NA",
                 "SAVBAL_NA", "ATMAMT_NA", "POS_NA", "POSAMT_NA", "CDBAL_NA", "IRABAL_NA",
                 "INVBAL_NA", "MMBAL_NA", "CCBAL_NA", "INCOME_NA", "LORES_NA", "HMVAL_NA",
                 "AGE_NA", "CRSCORE_NA")

# store names of categorical variables
cat_vars = colnames(ins_t)[!(colnames(ins_t) %in% cont_vars)]

lapply(ins_t[cat_vars], as.factor)

# for each continuous variable in data frame
for (i in 1:length(cont_vars)) {
  # make an indicator variable to represent missing values
  ins_t[,cont_vars_na[i]] = as.integer(is.na(ins_t[,cont_vars[i]]))
}

# for each continuous variable with missing values
for (i in 1:length(cont_vars)) {
  # impute missing value with median of the variable
  ins_t[,cont_vars[i]][is.na(ins_t[,cont_vars[i]])] = median(ins_t[,cont_vars[i]], na.rm = T)
}

# for each categorical variable with missing values
for (i in 1:length(cat_vars)) {
  # impute missing value with mode of the variable
  ins_t[,cat_vars[i]][is.na(ins_t[,cat_vars[i]])] = as.factor(Mode(ins_t[,cat_vars[i]], na.rm = T))
} 


##################################################################################################################

# Part 1: Build a model using the MARS algorithm and report details

# pass a logistic regression model to the earth function and use generalized cross-validation
fit_mars = earth(INS ~ ., data = ins_t, glm = list(family = binomial(link = "logit")))

# set the seed
set.seed(54321)
# pass a logistic regression model to the earth function and use ten fold cross-validation
fit_mars_cv = earth(INS ~ ., data = ins_t, glm = list(family = binomial(link = "logit")), pmethod = "cv", nfold = 10)

# report variable importance
summary(fit_mars)
evimp(fit_mars)

# report variable importance on cv model
summary(fit_mars_cv)
evimp(fit_mars_cv)

# SAVBAL appeared in 20 of the 20 subsets, CDBAL appeared in 18 of 20 subsets, ...
# all variables showed up at least once

# consistent variable importance between cv and gcv

# plot the ROC curve
plotROC(ins_t$INS, fit_mars$fitted.values)

# area under ROC curve (%Conc + (1/2)%tied) = 0.7995

# plot the ROC curve
plotROC(ins_t$INS, fit_mars_cv$fitted.values)

# area under ROC curve (%Conc + (1/2)%tied) = 0.7982


##################################################################################################################

# Part 2: Build General Additive Model and report details

# use a decision tree for initial variable selection
ins_tree = rpart(INS ~ ., data = ins_t, method = 'class', parms = list(split = 'entropy'))

# make a data frame to hold variable importance values
vimp_ins_tree = data.frame(ins_tree$variable.importance)
vimp_ins_tree$names = as.character(rownames(vimp_ins_tree))

# use a bar chart to visualize variable importance
ggplot(data = vimp_ins_tree, aes(x = names, y = ins_tree.variable.importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(x = "Variable Name", y = "Variable Importance")

# fit a gam model on variables selected using the decision tree
fit_gam <- mgcv::gam(INS ~ s(SAVBAL) + factor(SAV) + factor(POSAMT_NA) + s(POSAMT) +  factor(POS_NA) + factor(MMCRED) +
                           s(MMBAL) + factor(MM) + s(IRABAL) + factor(INVBAL_NA) + s(INVBAL) + factor(INV) + 
                           s(INCOME) + s(DDABAL) + factor(DDA) + s(CDBAL) + factor(CCPURC) + factor(CCBAL_NA) +
                           factor(BRANCH) + s(ATMAMT) + s(ACCTAGE),
                 method = 'REML', family = binomial(link = 'logit'), data = ins_t)
# summarize model output
summary(fit_gam)

# plot the ROC curve
plotROC(ins_t$INS, fit_gam$fitted.values)

# examining output, keep: MM, INV, DDA, BRANCH, SAVBAL, DDABAL, CDBAL, ATMAMT, ACCTAGE (based on p-value)
# area under ROC curve (%Conc + (1/2)%tied) = 0.7947

# fit a gam model with spline variable selection on variables selected using the decision tree
fit_gam2 <- mgcv::gam(INS ~ s(SAVBAL) + factor(SAV) + factor(POSAMT_NA) + s(POSAMT) +  factor(POS_NA) + factor(MMCRED) +
                       s(MMBAL) + factor(MM) + s(IRABAL) + factor(INVBAL_NA) + s(INVBAL) + factor(INV) + 
                       s(INCOME) + s(DDABAL) + factor(DDA) + s(CDBAL) + factor(CCPURC) + factor(CCBAL_NA) +
                       factor(BRANCH) + s(ATMAMT) + s(ACCTAGE),
                     method = 'REML', select = T, family = binomial(link = 'logit'), data = ins_t)
# summarize model output
summary(fit_gam2)

# plot the ROC curve
plotROC(ins_t$INS, fit_gam2$fitted.values)

# examining output, keep: MM, INV, DDA, BRANCH, SAVBAL, DDABAL, CDBAL, ATMAMT, ACCTAGE (based on p-value)
# area under ROC curve (%Conc + (1/2)%tied) = 0.7924

# fit a gam model on variables selected using MARS algorithm
fit_gam3 <- mgcv::gam(INS ~ s(SAVBAL) + s(CDBAL) + factor(DDA) + s(DDABAL) + s(MMBAL) + factor(POS_NA) + 
                            s(ACCTAGE) + s(CHECKS) + s(TELLER) + s(ATMAMT) + factor(INV) + factor(CC) + 
                            s(CCBAL) + factor(BRANCH) + s(IRABAL) + factor(DEP),
                     method = 'REML', family = binomial(link = 'logit'), data = ins_t)
# summarize model output
summary(fit_gam3)

# plot the ROC curve
plotROC(ins_t$INS, fit_gam3$fitted.values)

# examining output, keep: DDA, POS_NA, INV, CC, BRANCH, SAVBAL, CDBAL, DDABAL, MMBAL, ACCTAGE, CHECKS, TELLER, ATMAMT (based on p-value)
# area under ROC curve (%Conc + (1/2)%tied) = 0.801
