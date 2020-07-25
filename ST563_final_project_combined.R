
#######################################################################################
##############################Project Work Wiecha (Nate Wiecha).R######################
#######################################################################################

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





#######################################################################################
##############################Final Project EDA (David Weck).r#########################
#######################################################################################


library(tidyverse)
library(ggcorrplot)
library(knitr)

wine <- read_delim('winequality-red.csv', delim = ';')

colnames(wine)

#Creating data format suited for multiple histograms
hist_data <- wine %>% gather()

#Creating histograms for each variable
ggplot(hist_data, aes(x = value)) + 
  geom_histogram(fill = 'darkred', bins = 20) +
  facet_wrap(~key, scales = c('free_x'))

#Creating data format suited for multiple scatterplots
scatter_data <- wine %>% gather(key = 'variable', value = 'value', -quality)

#Creating scatterplots of each variable
ggplot(scatter_data, aes(x = value, y = quality)) +
  geom_jitter(color = 'darkred', alpha = .15) +
  facet_wrap(~variable, scales = 'free_x')

#Creating correlation matrix of variables
ggcorrplot(cor(wine), type= 'upper', title = 'Correlation Matrix of Wine Data', legend.title = 'Correlation', 
           lab = TRUE, lab_size = 3, outline.color = 'black')




#######################################################################################
##############################Final Project Model Fitting (David Weck)#################
#######################################################################################
library(tidyverse)
library(caret)
library(randomForest)
library(gbm)

wine <- read_delim('winequality-red.csv', delim = ';')

#Creating binary response for classification methods
wine_class <- wine
wine_class$above_avg <- as.factor(ifelse(wine$quality > mean(wine$quality), 1, 0))
wine_class <- select(wine_class, -quality)

#Setting seed for reproducibility
set.seed(563)

train_idx <- sample(1:nrow(wine), 1200, replace = FALSE)

train_data <- wine[train_idx, ]
test_data <- wine[-train_idx, ]

train_class <- wine_class[train_idx, ]
test_class <- wine_class[-train_idx, ]

#5 fold cross validation
control <- trainControl(method = 'cv', number = 5)

## ------------------------------------------
#Bagged models

bag_grid <- expand.grid(mtry = c(11))

bag_mod <- train(quality ~ ., data = train_data, method = 'rf',
                 trControl = control,
                 tuneGrid = bag_grid)

bag_mod

bag_preds <- predict(bag_mod, newdata = test_data)

(bag_mse <- mean((test_data$quality - bag_preds) ^ 2))

bag_imp <- varImp(bag_mod)
plot(bag_imp)

## Classification
bag_class_mod <- train(above_avg ~ ., data = train_class, method = 'rf',
                       trControl = control,
                       tuneGrid = bag_grid)

bag_class_mod

bag_class_preds <- predict(bag_class_mod, newdata = test_class)

confusionMatrix(bag_class_preds, test_class$above_avg)

bag_class_imp <- varImp(bag_class_mod)
plot(bag_class_imp)

## ------------------------------------------
#Random forest models
rf_grid <- expand.grid(predFixed = c(4,6, 11), minNode = c(3,5,7))

rf_mod <- train(quality ~ ., data = train_data, method = 'Rborist',
                trControl = control,
                tuneGrid = rf_grid)

rf_mod$bestTune

rf_preds <- predict(rf_mod, newdata = test_data)

(rf_mse <- mean((test_data$quality - rf_preds) ^ 2))

rf_imp <- varImp(rf_mod)
plot(rf_imp)

## Classification

rf_class_mod <- train(above_avg ~ ., data = train_class, method = 'Rborist',
                      trControl = control,
                      tuneGrid = rf_grid)

rf_class_mod$bestTune

rf_class_preds <- predict(rf_class_mod, newdata = test_class)

confusionMatrix(rf_class_preds, test_class$above_avg)

rf_imp_class <- varImp(rf_class_mod)
plot(rf_imp_class)

## -------------------------------------------
#GBM model

#setting up random search
gbm_control <- trainControl(method = 'cv', number = 5, search = 'random')

gbm_grid <- expand.grid(n.trees = c(500, 1000, 1200),
                        interaction.depth = c(2, 5, 7, 10),
                        shrinkage = c(.01, .05, .1),
                        n.minobsinnode = c(3, 5, 7))

gbm_mod <- train(quality ~ ., data = train_data, method = 'gbm',
                 trControl = gbm_control,
                 tuneGrid = gbm_grid,
                 verbose = FALSE)

gbm_mod$bestTune

gbm_preds <- predict(gbm_mod, newdata = test_data)

(gbm_mse <- mean((test_data$quality - gbm_preds) ^ 2))

gbm_imp <- varImp(gbm_mod)
plot(gbm_imp)

## Classification

gbm_class_mod <- train(above_avg ~ ., data = train_class, method = 'gbm',
                       trControl = gbm_control,
                       tuneGrid = gbm_grid,
                       verbose = FALSE)

gbm_class_mod$bestTune

gbm_class_preds <- predict(gbm_class_mod, newdata = test_class)

confusionMatrix(gbm_class_preds, test_class$above_avg)

gbm_imp_class <- varImp(gbm_class_mod)

plot(gbm_imp_class)

## ------------------------------------------
#KNN 

knn_grid <- expand.grid(k = 1:20)

knn_class_mod <- train(above_avg ~ ., data = train_class, method = 'knn',
                       trControl = control,
                       tuneGrid = knn_grid,
                       preProcess = c('center', 'scale'))

knn_class_mod$bestTune

knn_class_preds <- predict(knn_class_mod, newdata = test_class)

confusionMatrix(knn_class_preds, test_class$above_avg)
## ------------------------------------------
#Polynomial Regression

poly_grid <- expand.grid(nvmax = 1:34)

forward_mod <- train(quality ~ poly(`fixed acidity`, 3) + poly(`volatile acidity`, 3) + poly(`citric acid`, 3) +
                       poly(`residual sugar`, 3) + poly(chlorides, 3) + poly(`free sulfur dioxide`, 3) +
                       poly(`total sulfur dioxide`, 3) + poly(density, 3) + poly(pH, 3) + poly(sulphates, 3) +
                       poly(alcohol, 3),
                     data = train_data, method = 'leapForward',
                     trControl = control,
                     tuneGrid = poly_grid)

forward_mod$bestTune

summary(forward_mod)

forward_mod_preds <- predict(forward_mod, newdata = test_data)

(forward_mod_mse <- mean((test_data$quality - forward_mod_preds) ^ 2))

plot(forward_mod)

# Backward selection
backward_mod <- train(quality ~ poly(`fixed acidity`, 3) + poly(`volatile acidity`, 3) + poly(`citric acid`, 3) +
                        poly(`residual sugar`, 3) + poly(chlorides, 3) + poly(`free sulfur dioxide`, 3) +
                        poly(`total sulfur dioxide`, 3) + poly(density, 3) + poly(pH, 3) + poly(sulphates, 3) +
                        poly(alcohol, 3),
                      data = train_data, method = 'leapBackward',
                      trControl = control,
                      tuneGrid = poly_grid)

backward_mod$bestTune

summary(backward_mod)

backward_mod_preds <- predict(backward_mod, newdata = test_data)

(backward_mod_mse <- mean((test_data$quality - backward_mod_preds) ^ 2))

plot(backward_mod)

##Best subset

# Backward selection
best_subsets_mod <- train(quality ~ poly(`fixed acidity`, 3) + poly(`volatile acidity`, 3) + poly(`citric acid`, 3) +
                            poly(`residual sugar`, 3) + poly(chlorides, 3) + poly(`free sulfur dioxide`, 3) +
                            poly(`total sulfur dioxide`, 3) + poly(density, 3) + poly(pH, 3) + poly(sulphates, 3) +
                            poly(alcohol, 3),
                          data = train_data, method = 'leapSeq',
                          trControl = control,
                          tuneGrid = poly_grid)

best_subsets_mod$bestTune

summary(best_subsets_mod)

best_subsets_mod_preds <- predict(best_subsets_mod, newdata = test_data)

(best_subsets_mod_mse <- mean((test_data$quality - best_subsets_mod_preds) ^ 2))

plot(best_subsets_mod)     




#######################################################################################
##############################Final project MIN HE.r###################################
#######################################################################################

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
