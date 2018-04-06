# Read in the data
library(readr)
data <- read_csv("data/connect-4.dat", 
                 col_names = FALSE, skip = 47)

# Pre-processing
table(data$X43)
data.new <- apply(data,2, as.factor)

data.new <- as.data.frame(data.new)
data.use <- model.matrix(~ . + 0, data=data.new, contrasts.arg = lapply(data.new, contrasts, contrasts=FALSE))
ncol(data.use)

stopifnot(require(caret))

in_train <- createDataPartition(data.new$X43, p = 0.75, list = FALSE)
training <- data.use[in_train, ]
testing <- data.use[-in_train, ]

testingdata <- data.new[-in_train, ]
x_trainrf <- data.new[in_train, -43]
y_trainrf <- data.new[in_train, 43]
x_testrf <- data.new[-in_train, -43]
y_testrf <- data.new[-in_train, 43]

x_train <- as.matrix(training[,-c(127:129)])
y_train <- as.matrix(training[,c(127:129)])
x_test <- as.matrix(testing[,-c(127:129)])
y_test <- as.matrix(testing[,c(127:129)])


# Initialize NN
library(keras)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 600, activation = "relu", input_shape = c(126)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 300, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 150, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 3, activation = "softmax")

summary(model)

# Define loss function and optimizer

model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999),
  metric_categorical_accuracy
)

# Fit the NN

history <- model %>% fit(
  x_train, y_train, 
  epochs = 50, batch_size = 256, 
  validation_split = 0.1
)

plot(history)

# Score the model
p_correct_nn <- model %>% evaluate(x_test, y_test, batch_size = 256, verbose = 1)
paste("The accuracy of the neural network is:", round(p_correct_nn$categorical_accuracy, 3))

raw_pred_nn <- model %>% predict(x_test)
pred_nn <- model %>% predict_classes(x_test)
table_nn <- table(testingdata$X43, pred_nn)
table_nn


# Benchmark Random Forest model

library(randomForest)

rforest <- randomForest(x_trainrf, y_trainrf)
pred_rforest <- predict(rforest, newdata = x_testrf, type = "response")
rforest_table <- table(testingdata$X43, pred_rforest)

p_correct_rforest <- (rforest_table[1,1] + rforest_table[2,2] + rforest_table[3,3]) / nrow(testingdata)
paste("The accuracy of the random forest model is:", round(p_correct_rforest, 3))

rforest_table