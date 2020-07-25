setwd("C://Users//nwiec//OneDrive//Desktop//Academic//ST 563")
library(tidyverse)
library(leaps)
library(caret)
library(glmnet)
library(e1071)
library(ggplot2)
library(reshape2)


data <- read.table("winequality-red.csv", sep=";", header=TRUE)
set.seed(563)
traindata <- sample(1:nrow(data), 1200)

#5 fold CV for all relevant models
control_cv <- trainControl(method = 'cv', number = 5)
lambda_grid <- expand.grid(lambda = 10 ^ seq(-2, 3, .05))


##########################   Ridge    ###############################
ridge <- train(quality ~ ., method="ridge", subset=traindata, data=data,
               trControl = control_cv,
               tuneGrid = lambda_grid)
ridge$bestTune
#need to figure out how to get model coeffs. Just fit new ridge with besttune? Probably?
ridge.pred <- predict(ridge, newdata=data[-traindata,])
mean((ridge.pred - data[-traindata,]$quality)^2)
#41.8

############################### Lasso  #######################
#I was not able to figure out lasso using caret, just used glmnet directly as in the lab
grid <- 10^seq(10, -2, length=100)

x <- model.matrix(quality ~ ., data)[,-1]
y <- data$quality

lasso.mod <- glmnet(x[traindata,], y[traindata], alpha=1, lambda=grid)
cv.out <- cv.glmnet(x[traindata,], y[traindata], alpha=1)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam
lasso.pred <- predict(lasso.mod, s=bestlam, newx=x[-traindata,])
mean((lasso.pred-y[-traindata])^2)
#41.4
out <- glmnet(x, y, alpha=1, lambda=grid)
lasso.coef <- predict(out, type="coefficients", s=bestlam)[1:12,]
lasso.coef

########################### Linear Regression #########################

linear <- lm(quality ~ ., data=data, subset=traindata)
summary(linear)
linear.pred <- predict(linear, newdata=data[-traindata,])
mean((linear.pred - data[-traindata,]$quality)^2)
#42.0

########################### Subsets ##################################
data.train <- data[traindata,]
data.test <- data[-traindata,]

regfit.best <- regsubsets(quality ~ ., data=data, subset=traindata, nvmax=11)
test.mat <- model.matrix(quality~., data=data[-traindata,])

predict.regsubsets <- function(object, newdata, id, ...){
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object,id=id)
  xvars <- names(coefi)
  mat[,xvars] %*% coefi
}

k <- 5
folds <- sample(1:k,nrow(data.train), replace=TRUE)
cv.errors <- matrix(NA, k, 11, dimnames=list(NULL, paste(1:11)))

for(j in 1:k){
  best.fit =regsubsets(quality~.,data=data.train[folds!=j,],nvmax=11)
  for(i in 1:11){
    pred <- predict(best.fit, data.train[folds==j,],id=i)
    cv.errors[j,i] <- mean((data.train$quality[folds==j]-pred)^2)
  }
}


mean.cv.errors <- apply(cv.errors, 2, mean)
mean.cv.errors
par(mfrow=c(1,1))
plot(mean.cv.errors, type='b')
bestsize <- which.min(mean.cv.errors)

reg.best <- regsubsets(quality~.,data=data.train,nvmax=11)
coef(reg.best,bestsize)

best.lm <- lm(quality ~ volatile.acidity + citric.acid + chlorides +free.sulfur.dioxide+ total.sulfur.dioxide + pH + sulphates + alcohol, data=data.train)

subset.pred <- predict(best.lm, newdata=data.test)
mean((data.test$quality - subset.pred)^2)
#41.7
##################### Logistic Regression ###########################

data.logistic <- data
data.logistic$quality <- as.factor(as.integer(data.logistic$quality > 5))

logistic.fit <- glm(quality ~ ., data=data.logistic, subset=traindata, family=binomial)

summary(logistic.fit)
logistic.pred <- predict(logistic.fit, newdata=data[-traindata,], type="response")
logistic.pred <- as.integer(logistic.pred > .5)

conf.matrix <- table(pred=logistic.pred, true=data.logistic[-traindata, "quality"])
conf.matrix
(conf.matrix[1,2] + conf.matrix[2,1])/sum(conf.matrix)
#prediction error of 24.8% on test data


############################### SVM  ######################################

#Linear
trControl <- trainControl(method="CV",number=5)

svm.fit.lin <- train(quality ~ ., method="svmLinear2", trControl = trControl, tuneGrid=data.frame(cost=c(100, 50, 25, 12, 6, 3, 1, .5, .1)),
                 data=data.logistic, subset=traindata)
svm.fit.lin
svm.fit.lin$bestTune
best.linear.svm <- svm.fit.lin$finalModel

names.pred <- best.linear.svm$xNames
svm.linear.pred <- predict(best.linear.svm, newdata=data.logistic[-traindata,names.pred])
conf.matrix.svm.lin <- table(pred = svm.linear.pred, true=data.logistic[-traindata,"quality"])
(conf.matrix.svm.lin[1,2] + conf.matrix.svm.lin[2,1]) / sum(conf.matrix.svm.lin) 
#25.3 % test error 

#polynomial
tuneGrid.poly <- data.frame(expand.grid(c(1,2,3,4,5), c(3,1,.5,.1,.01)), rep(1,25))
names(tuneGrid.poly) <- c("degree", "C", "scale")


svm.fit.poly <- train(quality ~ ., method="svmPoly", trControl=trControl, 
                      tuneGrid=tuneGrid.poly,
                      data=data.logistic, subset=traindata)

svm.fit.poly
best.poly.svm <- svm.fit.poly$finalModel
names.pred.poly <- svm.fit.poly$coefnames
svm.poly.pred <- predict(svm.fit.poly, newdata=data.logistic[-traindata,names.pred.poly])
conf.matrix.svm.poly <- table(pred = svm.poly.pred, true=data.logistic[-traindata,"quality"])
(conf.matrix.svm.poly[1,2] + conf.matrix.svm.poly[2,1]) / sum(conf.matrix.svm.poly) 
#24.06 % test error

#radial kernel
#using e1071 package from lab since don't know how parameter sigma for kernlab/caret works

tune.out <- tune(svm, quality~., data=data.logistic[traindata,], kernel="radial", ranges=list(cost=c(.1,1,10,100,1000), gamma=c(.5,1,2,3,4)))
summary(tune.out)

conf.matrix.radial <- table(true=data.logistic[-traindata,"quality"], pred=predict(tune.out$best.model, newdata=data.logistic[-traindata,]))
(conf.matrix.radial[1,2] + conf.matrix.radial[2,1]) / sum(conf.matrix.radial) 
#wow! 22.8% test error

#################################  PCR/PLS  #################################################

#Principle component regression
tuneGrid.pcr <- data.frame(ncomp = seq(1,11))
pcr.fit <- train(quality ~ ., data=data, method="pcr", trControl=trControl, tuneGrid=tuneGrid.pcr, subset=traindata)
pcr.fit$bestTune
#ie, perform least squares regression

#Partial least squares

pls.fit <- train(quality ~ ., data=data, method="pls", trControl=trControl, tuneGrid=tuneGrid.pcr,subset=traindata)
pls.fit$bestTune
ncomp.best <- pls.fit$bestTune
pls.fit$finalModel
pls.pred <- predict(pls.fit$finalModel, newdata=data[-traindata,])
mean((data[-traindata,]$quality - pls.pred)^2) #pretty bad
#somehow worse than least squares?

#######################  EDA  ############################################
cols <- names(data)
for(i in 1:11){
  plot(data[,i], data$quality)
  title(cols[i])
}
#correlation heatmap http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization
cors <- cor(data[,-12])
melted_cormat <- melt(cors)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile()
#not much corrleation between predictors. good. means lasso/ridge a bit less useful potentially
