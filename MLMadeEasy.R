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
#              The code in this file leverages a cleaned version of the UCI ML 
#              Repository's "Adult" dataset. The original data is available at:
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


# NOTE - Set your working directory before runing the following.
# Re-load adult training data.
adult.train <- read.csv("AdultTrain.csv")


# Let's get familiar with our data.
str(adult.train)


# Here's how we can check what values a categorical variable has.
levels(adult.train$MaritalStatus)


# Let's train our first decision tree using 3 features and visualize it.
decision.tree.1 <- rpart(Label ~ MaritalStatus + EducationNum + HoursPerWeek + Age, data = adult.train)
rpart.plot(decision.tree.1, type = 5, extra = 104, tweak = 1.4)


# Let's train another tree using all the features. Notice how the tree
# chooses only the features that are most powerful for prediction.
decision.tree.2 <- rpart(Label ~ ., data = adult.train)
rpart.plot(decision.tree.2, type = 5, extra = 104, tweak = 1.4)




#=======================================================================================
#
# Section 3 - Introduction to random forests
#
#=======================================================================================

# Load the libraries we will use for this section of the course
library(randomForest)


# NOTE - Set your working directory before runing the following.
# Re-load adult training data.
adult.train <- read.csv("AdultTrain.csv")


# Let's take a look at the help file.
?randomForest


# Time to train our first random forest. Ask the algorithm to predict our
# label using all the available features usign R's formula syntax.
first.rf <- randomForest(Label ~ ., data = adult.train)


# The randomForest function gives us a lot of info back.
first.rf



#=======================================================================================
#
# Section 4 - Data analysis & feature engineering with random forests
#
#=======================================================================================


# Caitlin to put content here.




#=======================================================================================
#
# Section 5 - Introduction to predictive model testing
#
#=======================================================================================

# Load the libraries we will use for this section of the course
library(randomForest)


# NOTE - Set your working directory before runing the following.
# Re-load adult training data.
adult.train <- read.csv("AdultTrain.csv")

# Load adult test data.
adult.test <- read.csv("AdultTest.csv")


# Houston, we have a problem!
length(unique(adult.train$NativeCountry))
length(unique(adult.test$NativeCountry))


# We have one more NativeCountry category (R calls these "levels") in the
# training data than we do in the testing data. Fix this up.
levels(adult.test$NativeCountry) <- levels(adult.train$NativeCountry)



