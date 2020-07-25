library(caret)
library(tidyverse)
library(cluster) 
library(MASS)
library(corrplot)
# polynomial regression, clustering, KNN
redDat = read.table("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";", header = T)
str(redDat)
head(redDat)
summary(redDat)
colnames(redDat)
wine_corr <- cor(redDat)
corrplot(wine_corr,method="number")

# split the training data and testing data
set.seed(563)
train_idx <- sample(1:nrow(redDat), 1200, replace = FALSE)

preproc1 <- preProcess(redDat, method=c("center", "scale"))
norm1 <- predict(preproc1, redDat)
norm1[,'quality'] <- redDat[,'quality']

train_data <- norm1[train_idx, ]
test_data <- norm1[-train_idx, ]

hist(norm1$quality)

# variable selection-stepwise

# Fit the full model
full.model <- lm(quality ~., data = train_data)
# Stepwise regression model
step.model <- stepAIC(full.model, direction = "both", trace = FALSE)
summary(step.model)

test_predicted<-predict(step.model, test_data)

train_MSE = mean(step.model$residuals^2)
test_MSE <- mean((test_predicted - test_data[,"quality"])^2)


# KNN for the quality with converting the variable to be a level factor
norm1[,'quality'] <- as.factor(norm1[,'quality'])
pr <- knn(norm1[train_idx,],norm1[-train_idx,],cl=norm1[train_idx,'quality'],k=5)

tab <- table(pr,norm1[-train_idx,'quality'])
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)
