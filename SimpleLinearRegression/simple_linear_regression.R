#Data Preprocessing

#Importing dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[, 2:3]

#Splitting the data into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

#Fitting simple linear regression to the training set
regressor = lm(formula = Salary ~ YearsExperience, 
               data = training_set)

#Predicting the test set results
y_pred = predict(regressor, newdata = test_set)

#Visualising the training set results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'green') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab('Years of Expereince') +
  ylab('Salary')

#Visualising the test set results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'green') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Test Set)') +
  xlab('Years of Expereince') +
  ylab('Salary')