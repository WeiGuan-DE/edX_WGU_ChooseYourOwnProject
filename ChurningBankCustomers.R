#######################################################################
# Data Initiation
#######################################################################

# Install and load the necessary packages.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org", dependencies = TRUE)
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org", dependencies = TRUE)
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org", dependencies = TRUE)
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org", dependencies = TRUE)
if(!require(DataExplorer)) install.packages("DataExplorer", repos = "http://cran.us.r-project.org", dependencies = TRUE)
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org", dependencies = TRUE)
if(!require(curl)) install.packages("curl", repos = "http://cran.us.r-project.org", dependencies = TRUE)
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org", dependencies = TRUE)

library(tidyverse)
library(ggplot2)
library(caret)
library(curl)
library(Rborist)
library(rpart)
library(DataExplorer)
library(data.table)

# Download the data set from the csv file 
Customer <- read.csv(curl("https://raw.githubusercontent.com/WeiGuan-DE/edX_WGU_ChooseYourOwnProject/main/ChurningBankCustomers.csv"), header = TRUE, stringsAsFactors =  TRUE)

# remove not relevant columns from the data set
Customer_clean <- Customer %>% select(-Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1 
                                      & -Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2)
Customer_clean <- Customer_clean %>% select(-CLIENTNUM)
rm(Customer)




#######################################################################
# Data Cleaning, Exploration and Visualization
#######################################################################

# Numbers of rows and columns in the data set customer_clean
dim(Customer_clean)

# data structure
str(Customer_clean)

# plot the overview of the data set
plot_intro(Customer_clean, title = "Data Exploration - Overview")

# plot missing values
plot_missing(Customer_clean, title = "Data Exploration - Missing Values")

# plot histograms
plot_histogram(Customer_clean, title = "Data Exploration - Feature Value Distribution")

# plot bar charts for discrete features
plot_bar(Customer_clean, title = "Data Exploration - Discrete Features")

# calculate the percentage of attrited customers
mean(Customer_clean$Attrition_Flag == "Attrited Customer")

# create the train set (80%) and the test set (20%). 
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(Customer_clean$Attrition_Flag,
                                  times = 1, 
                                  p = 0.2, 
                                  list = FALSE)
train_set <- Customer_clean[-test_index,]
test_set <- Customer_clean[test_index,]




#######################################################################
# Model Design
#######################################################################

# explore classification models in the caret package
all_models <- modelLookup()
all_models <- all_models %>% filter(forClass == TRUE)
head(all_models)
dim(all_models)

# select popular algorithms/models for further analysis
models <- c("adaboost", "bayesglm", "knn", "naive_bayes", "Rborist",
            "rf", "rpart", "svmLinear", "svmRadial")

# choose to use 10-fold cross validation
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
trainControl <- trainControl(method = "cv", number = 10, p = 0.9)

# train the models, make predictions on the test set and transform predictions into a data frame
# it could take quite a long time.
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
predictions <- sapply(models, function(model) {
  print(model)
  fit <- train(Attrition_Flag ~ .,
               method = model,
               data = train_set,
               trControl = trainControl)
  prediction <- predict(fit, test_set)
  data.frame(model = prediction)
})
predictions <- as.data.frame(predictions)




#######################################################################
# Model Evaluation
#######################################################################

# calculate the accuracies of the selected models
accuracies <- sapply(predictions, function(x) {
  confusionMatrix(data=x, reference=test_set$Attrition_Flag)$overall["Accuracy"]
})

# print the accuracies
print(accuracies)

# print the maximal accuracy (it is from "AdaBoost")
print(accuracies[which.max(accuracies)])

# calculate the F-measures of the selected models
f_measures <- sapply(predictions, function(x) {
  F_meas(data=x, reference=test_set$Attrition_Flag)
})

# print the F-measures
print(f_measures)

# print the maximal F-measure (it is from "AdaBoost")
print(f_measures[which.max(f_measures)])

# calculate the majority vote for Attrited Customer
votes <- rowMeans(predictions == "Attrited Customer")

# calculate prediction of the ensemble model
predictionEnsemble <- as.factor(ifelse(votes > 0.5, "Attrited Customer", "Existing Customer"))

# calculate the accuracy of the ensemble model
accuracyEnsemble <- 
  confusionMatrix(data=predictionEnsemble, reference=test_set$Attrition_Flag)$overall["Accuracy"]

# print the accuracy of the ensemble model
print(accuracyEnsemble)

# calculate the F-measure of the ensemble model
fMeasureEnsemble <- F_meas(data=predictionEnsemble, reference=test_set$Attrition_Flag)

# print the F-measure of the ensemble model
print(fMeasureEnsemble)

# train the optimized "AdaBoost" model with nIter = 400 and method = "Adaboost.M1"
fit_Adaboost <- train(Attrition_Flag ~ .,
                      method = "adaboost",
                      data = train_set,
                      trControl = trainControl,
                      tuneGrid = data.frame(nIter = 400, method = "Adaboost.M1"), 
)

# inspect of the overview of the final model parameters
fit_Adaboost

# calculate prediction of the optimized "Adaboost" model
predictionAdaboost <- predict(fit_Adaboost, test_set)

# calculate and print the accuracy of the optimized "Adaboost" model
print(confusionMatrix(data=predictionAdaboost,
                      reference=test_set$Attrition_Flag)$overall["Accuracy"])

# calculate and print the F-measure of the optimized "Adaboost" model
print(F_meas(data=predictionAdaboost, reference=test_set$Attrition_Flag))

# check variable importance
varImp(fit_Adaboost)

