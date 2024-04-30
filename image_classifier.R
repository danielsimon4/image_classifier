# Convolutional Neural Networks



############################ INSTALL & LOAD PACKAGES ###########################

#install.packages("BiocManager")
#BiocManager::install("EBImage")


#install.packages("remotes")
#remotes::install_github("rstudio/tensorflow")
#reticulate::install_python()
#install_tensorflow(envname = "r-tensorflow")
#install.packages("keras")
#install_keras()

library(keras)
library(EBImage)



################################## READ IMAGES #################################

# Set working directory
setwd('C:/.../images')


# Define a vector containing filenames for training images
train_filenames <- c('p1.jpg', 'p2.jpg', 'p3.jpg', 'p4.jpg', 'p5.jpg',
          'c1.jpg', 'c2.jpg', 'c3.jpg', 'c4.jpg', 'c5.jpg',
          'b1.jpg', 'b2.jpg', 'b3.jpg', 'b4.jpg', 'b5.jpg')

# Initialize an empty list to store training images
train <- list()

# Loop through each filename in train_filenames, read the corresponding image, 
# and store it in the train list
for (i in 1:15) {
  train[[i]] <- readImage(train_filenames[i])
}



# Define a vector containing filenames for testing images
test_filenames <- c('p6.jpg', 'c6.jpg', 'b6.jpg')

# Initialize an empty list to store testing images
test <- list()

# Loop through each filename in test_filenames, read the corresponding image, 
# and store it in the test list
for (i in 1:3) {
  test[[i]] <- readImage(test_filenames[i])
}



################################ EXPLORE IMAGES ################################

display(train[[12]])

par(mfrow = c(3,5))
for (i in 1:15) plot(train[[i]])
par(mfrow = c(1,1))



############################ RESIZE & COMBINE IMAGES ###########################

for (i in 1:15) {train[[i]] <- resize(train[[i]], 100, 100)}
for (i in 1:3) {test[[i]] <- resize(test[[i]], 100, 100)}


train <- combine(train)
x <- tile(train, 5)
display(x, title='Train Images')

test <- combine(test)
y <- tile(test, 3)
display(y, title = 'Test Images')

str(train)



############################### REORDER DIMENSION ##############################

train <- aperm(train, c(4, 1, 2, 3))
test <- aperm(test, c(4, 1, 2, 3))

str(train)



##################################### LABELS ###################################

trainy <- c(0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2)
testy <- c(0, 1, 2)

trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)

trainLabels



################################## DEFINE MODEL ################################

model <- keras_model_sequential()

model %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu',
                input_shape = c(100,100, 3)) %>%
  layer_conv_2d(filters = 32,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_conv_2d(filters = 64,
                kernel_size = c(3,3),
                activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate=0.25) %>%
  layer_dense(units = 3, activation = 'softmax') %>%
  
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_sgd(learning_rate = 0.01,
                                    weight_decay = 1e-6,
                                    momentum = 0.9,
                                    nesterov = T),
          metrics = c('accuracy'))

summary(model)



################################## TRAIN MODEL #################################

history <- model %>%
  fit(train,
      trainLabels,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2
      #, validation_data = list(test, testLabels)
  )



################################ EVALUATE MODEL ################################

model %>% evaluate(train, trainLabels)



prob <- model %>% predict(train)
prob

predicted_classes <- max.col(prob)
predicted_classes



prob <- model %>% predict(test)
prob

predicted_classes <- max.col(prob)
predicted_classes
