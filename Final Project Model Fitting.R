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
