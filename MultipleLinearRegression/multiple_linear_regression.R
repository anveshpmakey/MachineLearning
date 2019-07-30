#Data Preprocessing

#Importing dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'Florida', 'California'),
                         labels = c(1, 2, 3))

#Splitting the dataset into Training Set and Test Set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting Multiple Linear Regression to the training set
regressor = lm(formula = Profit ~ .,
               data = training_set)

y_pred = predict(regressor, newdata = test_set)

#using only one value as dependant variable
regressor = lm(formula = Profit ~ R.D.Spend,
               data = training_set)
y_pred = predict(regressor, newdata = test_set)


