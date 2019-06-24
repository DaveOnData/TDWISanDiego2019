#=======================================================================================
#
# File:        MLMadeEasy.R
# Authors:     Caitlin Johnson & Dave Langer
# Description: This code was written as part of the 1-day course "Hands-on: Machine 
#              Learning Made Easy - No, Really!" delivered at the August 2019 TDWI 
#              San Diego Conference. Additional materials for the course can
#              be found at:
#
#                 https://github.com/DaveOnData/TDWISanDiego2019
#
#              The code in this file leverages a cleaned, subsetted version of the 
#              UCI ML Repository's "Adult" dataset. The original data is available 
#              at:
#
#                 http://archive.ics.uci.edu/ml/datasets/Adult
#
# NOTE - This file is provided "As-Is" and no warranty regardings its contents are
#        offered nor implied. USE AT YOUR OWN RISK!
#
#=======================================================================================



#=======================================================================================
#
# Section 2 - Introduction to decision trees
#
#=======================================================================================

# Load the libraries we will use for this section of the course
library(rpart)
library(rpart.plot)


# NOTE - Set your working directory before running the following.
# Re-load adult training data.
adult.train <- read.csv("AdultTrain.csv")


# Let's get familiar with our data.
str(adult.train)


# Here's how we can check what values a categorical variable has.
levels(adult.train$MaritalStatus)


# Let's train our first decision tree using 4 features and visualize it.
decision.tree.1 <- rpart(Label ~ MaritalStatus + EducationNum + HoursPerWeek + Age, data = adult.train)
prp(decision.tree.1, varlen = 14, faclen = 14, tweak = 1.2)


# Let's train another tree using all the features. Notice how the tree
# chooses only the features that are most powerful for prediction.
decision.tree.2 <- rpart(Label ~ ., data = adult.train)
prp(decision.tree.2, varlen = 0, faclen = 0, tweak = 1.6)



#=======================================================================================
#
# Section 3 - Introduction to random forests
#
#=======================================================================================

# Load the libraries we will use for this section of the course.
library(randomForest)


# Let's take a look at the help file.
?randomForest


# Time to train our first random forest. First, we want to ensure that we
# all see the same randomness. Use set.seed() to tell R to start the 
# ranomness from a common point.
set.seed(329324)
# Ask the random forest algorithm to predict our label using all the 
# available features using R's formula syntax.
random.forest.1 <- randomForest(Label ~ ., data = adult.train)


# The randomForest function gives us a lot of info back.
random.forest.1



#=======================================================================================
#
# Section 4 - Data analysis & feature engineering with random forests
#
#=======================================================================================


# The mighty random forest can tell us what features the algorithm found to be
# important for predicting the class label. To do this we need to call the 
# randomForest() function and tell it to keep track of feature importance.
set.seed(329324)
random.forest.1 <- randomForest(Label ~ ., data = adult.train, importance = TRUE)


# The random forest tracks feature importance by randomly shuffling each feature
# and seeing how much accuracy is impacted. The more important a feature is, the
# more shuffling the values will negatively impact the accuracy.
varImpPlot(random.forest.1, type = 1, scale = FALSE)


# A hallmark of machine learning is what is known as "feature engineering". 
# Feature engineering is the process for presenting your data to machine 
# learning algorithms in an optimal way. One of the most important forms of
# feature engineering is the creation of new features from your data that
# allows the ML algorithm to learn more.

# We can see in the variable importance plot that the "Fnlwgt" feature is
# the weakest feature. We also know that the Fnlwgt feature represents 
# demographic information (e.g., age, race, education, occupation, etc.).
# We can leverage this information as data scientists to engineer a new
# feature by dividing Fnlwgt by Age and EducationNum to extract additional 
# information from which the mighty random forest can learn.

# Call our new feature "OurFeature" (aka new hotness).
adult.train$OurFeature <- adult.train$Fnlwgt / (adult.train$Age * adult.train$EducationNum)


# We no longer need the Fnlwgt feature now that we have our new hotness!
adult.train$Fnlwgt <- NULL


# Train a new random forest with our new hotness.
set.seed(329324)
random.forest.2 <- randomForest(Label ~ ., data = adult.train, importance = TRUE)


# How important is our new feature (aka new hotness)?
varImpPlot(random.forest.2, type = 1, scale = FALSE)


# OK, let's compare before/after random forests.
random.forest.1
random.forest.2



#=======================================================================================
#
# Section 5 - Introduction to predictive model testing
#
#=======================================================================================

# Load the libraries we will use for this section of the course.
library(caret)


# NOTE - Set your working directory before runing the following.
# Load adult test data.
adult.test <- read.csv("AdultTest.csv")


# Houston, we have a problem!
length(unique(adult.train$NativeCountry))
length(unique(adult.test$NativeCountry))


# We have one more NativeCountry category (R calls these "levels") in the
# training data than we do in the testing data. Fix this up.
levels(adult.test$NativeCountry) <- levels(adult.train$NativeCountry)


# Our first random forest estimated the error rate at 16.92%. Reflexively,
# the estimate is 83.08% accuracy. Let's check the estimate using the
# test dataset. First, we need to make some predictions.
rf.preds.1 <- predict(random.forest.1, adult.test)
rf.preds.1[1:20]


# Use the confusionMatrix() function to analyze our predictions.
confusionMatrix(rf.preds.1, adult.test$Label)


# In the previous section we engineered a new feature named "OurFeature"
# (aka new hotness). We then trained a new mighty random forest to use this
# feature. If we want to make predictions we need to add this OurFeature to
# the test set.
adult.test$OurFeature <- adult.test$Fnlwgt / (adult.test$Age * adult.test$EducationNum)


# Make some predictions using our 2nd mighty random forest and check out 
# the confusion matrix.
rf.preds.2 <- predict(random.forest.2, adult.test)
confusionMatrix(rf.preds.2, adult.test$Label)


# Using our test set the 1st mighty random forest performed thusly:
#      Overall accuracy: 81.99%
#      <=50k accuracy:   91.51%
#      >50k accuracy:    76.02%


# Using our enhanced test set the 2nd mighty random forest perfomed thusly:
#      Overall accuracy: 82.24%
#      <=50k accuracy:   91.95%
#      >50k accuracy:    76.19%

