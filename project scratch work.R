setwd("C://Users//nwiec//OneDrive//Desktop//Academic//ST 563")
library(tidyverse)
library(caret)

data <- read.table("winequality-red.csv", sep=";", header=TRUE)
set.seed(563)
traindata <- sample(1:nrow(data), 1200)

ridge1 <- train(quality ~ ., method="ridge", subset=traindata, data=data)

ridge.pred <- predict(ridge1, newdata=data[-traindata,])
mean((ridge.pred - data[-traindata,]$quality)^2)