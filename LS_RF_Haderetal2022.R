

##################################################.
## Project: Landslide prediction using machine learning supervised classification techniques
## Script purpose: Random forest machine (RF) in CARET package
## Author: Paulo Hader
## Article: Landslide risk assessment considering socionatural factors: methodology and application to Cubatão municipality, São Paulo, Brazil
## url: https://doi.org/10.1007/s11069-021-04991-4
##################################################.


# References

# https://www.guru99.com/r-random-forest-tutorial.html


####
## 1 Data preparation ---------------------------------------------------------
####

# Go to URL of local folder and select and Copy.(C:/LS_RF)
path=readClipboard()
setwd("C:/LS_RF")

getwd() # for checking
.libPaths()
# .libPaths("C:/LS_RF/LS_Library")

# Install packages
packages <- c("rgdal","raster","plyr","dplyr","RStoolbox","RColorBrewer",
              "RColorBrewer","ggplot2", "sp", "caret", "doParallel", "randomForest", "Information")
install.packages(packages)
lapply(packages, require, character.only = T) # read the packages and check if installed 
rm(list=ls()) # clear the environment

# # Packages description
# install.packages("RStoolbox")  # Image analysis & plotting spatial data 
# library(rgdal)        # spatial data processing
# library(raster)       # raster processing
# library(plyr)         # data manipulation 
# library(dplyr)        # data manipulation 
# library(RStoolbox)    # Image analysis & plotting spatial data 
# library(RColorBrewer) # color
# library(ggplot2)      # plotting
# library(sp)           # spatial data
# library(caret)        # machine learning
# library(doParallel)   # Parallel processing
# # library(e1071)        # Naive Bayes

# 2 Import training and testing data ----
list.files(pattern = "csv$", full.names = TRUE)

# 2.1 Training Data -------------------------------------------------

# Importing training data
data_train <-  read.csv("C:/LS_RF/Data/Training.csv", header = TRUE, sep = ";" )
data_train <- (na.omit(data_train))
data_train <- data.frame(data_train)  # to remove the unwelcome attributes

# ETL Categorical Data
#data_train$TRAINING <- factor(data_train$TRAINING)

# Dealing with Categorical data (Converting numeric variable into groups in R)
#https://www.r-bloggers.com/from-continuous-to-categorical/

# Slope Aspect
ASPr <- cut(data_train$ASP, seq(0,361,45), right=FALSE, labels=c("a","b","c","d","e","f","g","h"))
table(ASPr)
ASPr <- factor(ASPr)
class(ASPr) # double check if not a factor


# Dealing with Categorical data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top

flags = data.frame(Reduce(cbind,lapply(levels(ASPr),function(x){(ASPr == x)*1})
))
names(flags) = levels(ASPr)
data_train = cbind(data_train, flags) # combine the ASPECTS with original data

# Remove the original Aspect data
data_train[ ,c('ASP')] <- list(NULL)


# Land use setting
LANDUSEr <- cut(data_train$LANDUSE, seq(1,11,1), right=FALSE, labels=c("l1","l2","l3","l4","l5","l6","l7","l8","L9","l10"))
table(LANDUSEr) 
class(LANDUSEr) # double check if not a factor

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top
flags = data.frame(Reduce(cbind,lapply(levels(LANDUSEr),function(x){(LANDUSEr == x)*1})
))
names(flags) = levels(LANDUSEr)
data_train = cbind(data_train, flags) # combine the landcover with original data



# Geology setting
GEOr<-cut(data_train$GEO, seq(2,16,1), right=FALSE, labels=c("g2","g3","g4","g5","g6","g7","g8","g9","g10","g11","g12","g13","g14","g15"))
table(GEOr) 
class(GEOr) # double check if not a factor

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top
flags = data.frame(Reduce(cbind,lapply(levels(GEOr),function(x){(GEOr == x)*1})
))
names(flags) = levels(GEOr)
data_train = cbind(data_train, flags) # combine the GEO with original data



# Geomorphology setting
GEOMr<-cut(data_train$GEOM, seq(1,8,1), right=FALSE, labels=c("gm1","gm2","gm3","gm4","gm5","gm6","gm7"))
table(GEOMr) 
class(GEOMr) # double check if not a factor

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top
flags = data.frame(Reduce(cbind,lapply(levels(GEOMr),function(x){(GEOMr == x)*1})
))
names(flags) = levels(GEOMr)
data_train = cbind(data_train, flags) # combine the GEOM with original data

#scaled_train = cbind(scaled_train, flags) # combine the GEOM with scaled_train
#scaled_t = cbind(scaled_t, flags) # combine the GEOM with scaled_t

table(data_train$gm)
# Remove the original landuse, geo, geom and related with missing data
data_train[,c('LANDUSE', 'GEO', 'GEOM', 'g15', 'gm2', 'l4', 'gm7', 'g14', 'gm5')] <- list(NULL)
# data_train <- data_train[,-7] # to remove landuse
# data_train <- data_train[,-8] # to remove GEO
# data_train <- data_train[,-7] # to remove Geom
# data_train <- data_train[,-39] # to remove g15
# data_train <- data_train[,-40] # to remove gm2
# data_train <- data_train[,-19] # to remove l4
# data_train <- data_train[,-43] # to remove gm7
# data_train <- data_train[,-37] # to remove g14
# 
# data_train <- data_train[,-39] # to remove gm2
# data_train <- data_train[,-43] # to remove gm7
# data_train <- data_train[,-37] # to remove g14
# data_train <- data_train[,-40] # to remove gm5
summary(data_train)

# Count the number of 1 and 0 elements with the values of dependent variable
as.data.frame(table(data_train$Training))

# Do Scale the data
# Standardization or Normalization, see this for more info
# https://sebastianraschka.com/Articles/2014_about_feature_scaling.html
# Z-score standardization or [ Min-Max scaling: typical neural network algorithm require data that on a 0-1 scale]
# https://vitalflux.com/data-science-scale-normalize-numeric-data-using-r/

# Original equation: n4 - unitization with zero minimum 
# >> ((x-min)/range))=y
# y: is the scaled value
# x: is the original value
# range: original range
# min: original minimum value

#  > ((y*range)+min)= x
# Ref (https://stackoverflow.com/questions/15215457/standardize-data-columns-in-r)

# Normalization
maxs <- apply(data_train, 2, max) 
mins <- apply(data_train, 2, min)
scaled_train <- as.data.frame(scale(data_train, center = mins, scale = maxs - mins))
scaled_t <- scaled_train
scaled_t$Training <- ifelse(scaled_t$Training == 1, "yes","no")

summary(scaled_train)
summary(scaled_t)


# 2.2 Testing Data --------------------------------------------------------

data_test <- read.csv("C:/LS_RF/Data/Testing.csv", header = T, sep = ";" )
data_test <- na.omit(data_test)
data_test <- data.frame(data_test)
str(data_test)
as.data.frame(table(data_test$Testing))


# Fix the categorial factor

# Slope Aspect
ASPTe <- cut(data_test$ASP, seq(0,361,45), right=FALSE, labels=c("a","b","c","d","e","f","g","h"))
table(ASPTe)

# Dealing with Categorial data
ASPTe <- factor(ASPTe)
flagse = data.frame(Reduce(cbind, 
                          lapply(levels(ASPTe), function(x){(ASPTe == x)*1})
))
names(flagse) = levels(ASPTe)
data_test = cbind(data_test, flagse) # combine the ASPECTS with original data
data_test[,c('ASP')] <- list(NULL) # remove original Aspect


# Landcover setting

LANDUSETe <- cut(data_test$LANDUSE, seq(1,11,1), right=FALSE, labels=c("l1","l2","l3","l4","l5","l6","l7","l8","L9","l10"))
table(LANDUSETe) 
class(LANDUSETe) # double check if not a factor

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top


flags = data.frame(Reduce(cbind,lapply(levels(LANDUSETe),function(x){(LANDUSETe == x)*1})
))
names(flags) = levels(LANDUSETe)
data_test = cbind(data_test, flags) # combine the landuse with original data


# GEO setting

GEOTe<-cut(data_test$GEO, seq(2,16,1), right=FALSE, labels=c("g2","g3","g4","g5","g6","g7","g8","g9","g10","g11","g12","g13","g14","g15"))
table(GEOTe) 
class(GEOTe) # double check if not a factor

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top


flags = data.frame(Reduce(cbind,lapply(levels(GEOTe),function(x){(GEOTe == x)*1})
))
names(flags) = levels(GEOTe)
data_test = cbind(data_test, flags) # combine the GEO with original data



# GEOM setting

GEOMTe<-cut(data_test$GEOM, seq(1,8,1), right=FALSE, labels=c("gm1","gm2","gm3","gm4","gm5","gm6","gm7"))
table(GEOMTe) 
class(GEOMTe) # double check if not a factor

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top


flags = data.frame(Reduce(cbind,lapply(levels(GEOMTe),function(x){(GEOMTe == x)*1})
))
names(flags) = levels(GEOMTe)
data_test = cbind(data_test, flags) # combine the GEOM with original data



summary(data_test)
# Remove the original landuse, geo, geom and related with missing data
data_test[,c('LANDUSE', 'GEO', 'GEOM', 'g15', 'gm2', 'l4', 'gm7', 'g14', 'gm5')] <- list(NULL)
# data_test <- data_test[ ,c(-7, -8, -7, -39, -40, -19)] <- list(NULL)
# data_test <- data_test[,-7] # to remove landuse
# data_test <- data_test[,-8] # to remove GEO
# data_test <- data_test[,-7] # to remove Geom
# data_test <- data_test[,-39] # to remove g15
# data_test <- data_test[,-40] # to remove gm2
# data_test <- data_test[,-19] # to remove l4
# data_test <- data_test[,-43] # to remove gm7
# data_test <- data_test[,-37] # to remove g14
# 
# data_test <- data_test[,-19] # to remove l4
# data_test <- data_test[,-37] # to remove g14
# data_test <- data_test[,-37] # to remove g15
# data_test <- data_test[,-38] # to remove gm2
# data_test <- data_test[,-40] # to remove gm5
# data_test <- data_test[,-41] # to remove gm7

summary(data_test)
data_test <- na.omit(data_test)
# Match the columns position for all the input data
#head(data_train,1)
#head(data_test,1)
#colnames(data_test)[3] <- "SlOPE" # Match the columns names

#data_test <- data_testN[,c(1,2,3,4,5,6,7,9,8)] Used to re-arrange the columns
## 2.3 Scale the data
maxs <- apply(data_test, 2, max) 
mins <- apply(data_test, 2, min)
scaled_test <- as.data.frame(scale(data_test, center = mins, scale = maxs - mins))
scaled_tst <-scaled_test
scaled_tst$Testing <- ifelse(scaled_tst$Testing == 1, "yes","no")

summary(scaled_tst)  # to check if you have any missings 

## 2.3 Merge all data ----
# Creating one data frame containing all data
names(scaled_tst)
scaled_tst$Slides=scaled_tst$Testing
names(scaled_t)
scaled_t$Slides=scaled_t$Training

All_incidents <- merge(scaled_tst[,-1], scaled_t[,-1], all=TRUE) #Full outer join: To keep all rows from both data frames, specify all=TRUE.  https://www.dummies.com/programming/r/how-to-use-the-merge-function-with-data-sets-in-r/
str(All_incidents)
All_incidents <- All_incidents[,c(40,1:39)] # re-order columns

scaled_tst$Slides= NULL  # remove Slide column
scaled_t$Slides=NULL  # remove Slide column



## 2.4 Data mining ----

# To predict which variable would be the best one for splitting the Decision Tree, plot a graph that represents the split for each of the variables
# Creating seperate dataframe for '"LevelsAve" features which is our target.
number.perfect.splits <- apply(X=All_incidents[,c(-1)], MARGIN = 2, FUN = function(col){
  t <- table(All_incidents$Slides,col)
  sum(t == 0)})

# Descending order of perfect splits
order <- order(number.perfect.splits,decreasing = TRUE)
number.perfect.splits <- number.perfect.splits[order]

# Plot graph
par(mar=c(10,2,2,2))
barplot(number.perfect.splits, main = "Number of perfect splits vs feature", xlab = "", ylab = "Feature", las=3, col="wheat") # Slope and TWI are the best classifiers


# Step 2: Data Visualization
data_train2=data_train
data_train2$Training <- ifelse(data_train2$Training == 1, "yes","no")
data_test2=data_test
data_test2$Testing <- ifelse(data_test2$Testing == 0, "no","yes")

# Create one file contain all data
data_test2$Slides=data_test2$Testing
data_train2$Slides=data_train2$Training

All_incidents_orginal <- merge(data_train2[,-1], data_test2[,-1], all=TRUE) #Full outer join: To keep all rows from both data frames, specify all=TRUE.  https://www.dummies.com/programming/r/how-to-use-the-merge-function-with-data-sets-in-r/
str(All_incidents_orginal)
#All_incidents_orginal <- All_incidents_orginal [,c(21,1:20)] # re-order columns


# Visual Elevation
ggplot(All_incidents_orginal, aes(ELEV, colour = Slides)) +
  geom_freqpoly(binwidth = 1) + labs(title= "Elevation Distribution by Landslides occurences")

# Visual Slope
ggplot(All_incidents_orginal, aes(SLOPE, colour = Slides)) +
  geom_freqpoly(binwidth = 1) + labs(title= "Slope Distribution by Landslides occurences", xlab = "Slope grade º", ylab = "Pixels (10)")


###
# Double check for na values
###
sum(is.na(scaled_t))
sum(is.na(scaled_tst))

# scaled_t <- na.omit(scaled_t)
# scaled_tst <- na.omit(scaled_tst)

# Information value to check predictor variable
Information::create_infotables(scaled_train, y = 'Training') # 3 variables with both positive and > 1 IV)
# the info tables show three potential predictable variables: SLOPE, l3 and TWI, which matches with the previous plots

#### 
## 3 Modeling ---------------------------------------------------------
####


# 3.1 Default settings ----

# Define the control
trControl <- trainControl(method='repeatedcv', 
                          repeats=3,
                          number = 10,
                          search = "grid")

# Building the model with the default values.

set.seed(1234)
# Run the model
rf_defaultN <- train(Training~., 
                     data=scaled_t,
                     method = "rf",
                     metric = "Accuracy",
                     trControl = trControl)
# Print the results
print(rf_defaultN)     
ggplot(rf_defaultN)
rf_defaultN$finalModel        # Results: mtry= 20, Number of trees: 500, OOB error rate: 3,03%
rf_defaultN$results 

# The algorithm uses 500 trees and tested three different values of mtry: 2, 20, 39.
# The final value used for the model was mtry = 20 with an accuracy of 0.9699824  Kappa = 0.7631195. Let's try to get a higher score.

# https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/randomForest

#  ntree
#       Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times.

#   mtry
#       Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3)

#   maxnodes
#        Maximum number of terminal nodes trees in the forest can have. 
#        If not given, trees are grown to the maximum possible (subject to limits by node size)


# 3.2 Search best mtry ----

set.seed(1234)
tuneGrid <- expand.grid(.mtry = c(1: 22))
rf_mtry <- train(Training~., 
                 data=scaled_t,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 500)
print(rf_mtry)
rf_mtry$bestTune$mtry # mtry = 21, 0.9700430 Accuracy, Kappa = 0.7627416

# We now can store it and use it when we need to tune the other parameters.
max(rf_mtry$results$Accuracy) # accuracy
best_mtry <- rf_mtry$bestTune$mtry 
best_mtry # mtry

mtryjune <- ggplot(data=rf_mtry, aes(x=mtry,y=Accuracy)) #ggplot2 better visualization
mtryjune + xlab ("#mtry") 


# 3.3 Search best maxnodes ----
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 30)) {
  set.seed(1234)
  rf_maxnode <- train(Training~., 
                      data=scaled_t,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 500)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
results_mtry
maxnodes
store_maxnode

maxnodeframe <- as.data.frame(store_maxnode)
rf_maxnode$results
maxnodesjune <- ggplot(data=rf_maxnode$finalModel) #ggplot2 better visualised
maxnodesjune + xlab ("#mtry")


# 3.4 Hyperparameterisation (search) ----
# We have our final model. Thus, we can train the random forest with the following parameters:

# ntree = default (500) trees will be trained
# mtry = 21: 21 features is chosen for each iteration

fit_rf_final <- train(Training~., 
                      data=scaled_t,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE
)

fit_rf_final # mtry = 22, 0.9703476 Accuracy
print(fit_rf_final) 
varImp(fit_rf_final) # l3 (land use 3) is the most important variable, following by SLOPE
ggplot(varImp(fit_rf_final), main="RF tuned model") # plot the variable importance


fit_rf_final$results # to check all the mtry and its accuracy

# Tune ntree 
# Here we obtain the best ntree vs accuracy of the model for OOB, non landslide and landslide
oob.error.data <- data.frame(
                  Trees=rep(1:nrow(fit_rf_final$finalModel$err.rate), times=3),
                  Type=rep(c("OOB", "no", "yes"), each=nrow(fit_rf_final$finalModel$err.rate)),
                  Error=c(fit_rf_final$finalModel$err.rate[,"OOB"],
                          fit_rf_final$finalModel$err.rate[,"no"],
                          fit_rf_final$finalModel$err.rate[,"yes"]))
# Plot of the OOB, non-landslide and landslide error vs ntree
ggplot(data=oob.error.data, aes(x=Trees, y=Error)) + 
  geom_line(aes(color=Type))


# Evaluate the model
p1_final<-predict(fit_rf_final, scaled_tst[,c(-1)], type = "raw")
confusionMatrix(p1_final, as.factor(scaled_tst$Testing))  # using more deep tree, the accuracy linearly increases 
# Accuracy : 0.9301

#Confusion Matrix and Statistics
             #Reference
#Prediction   no  yes
#       no  1819   37
#       yes  111  150

# 3.5 Hyperparameterisation (random) ----

control <- trainControl(method='repeatedcv', 
                        number=10, 
                        repeats=3,
                        search = 'random')    

# Random generate 15 mtry values with tune Length = 15
set.seed(1)
rf_random <- train(Training~., 
                   data=scaled_t,
                   method = 'rf',
                   metric = 'Accuracy',
                   trControl = control,
                   importance = TRUE)
print(rf_random)
varImp(rf_random)
plot(varImp(rf_random))
plot(rf_random)
plot(varImp(rf_random), main="RF random model")

# Evaluate the model
p1_random<-predict(rf_random, scaled_tst[,c(-1)], type = "raw")
confusionMatrix(p1_random, as.factor(scaled_tst$Testing))  # using more deep tree, the accuracy linearly increases! 
# Accuracy : 0.9225

# fit_rf_final is the best model


# To save any plot

# 1. Open jpeg file
#jpeg("varImportance RF.jpg", width = 800, height = 500)
# 2. Create the plot
#plot(X.rf,main="varImportanceAll RF" )
# 3. Close the file
#dev.off()



# 4 Validation - ROC curve ----
# https://stackoverflow.com/questions/46124424/how-can-i-draw-a-roc-curve-for-a-randomforest-model-with-three-classes-in-r
library(pROC)
#install.packages("pROC")
# the model is used to predict the test data. However, you should ask for type="prob" here
predictions1 <- as.data.frame(predict(fit_rf_final, scaled_test, type = "prob"))

##  Since you have probabilities, use them to get the most-likely class.
# predict class and then attach test class
predictions1$predict <- names(predictions1)[1:2][apply(predictions1[,1:2], 1, which.max)]
predictions1$observed <- as.factor(scaled_tst$Testing)
head(predictions1)

#    Now, let's see how to plot the ROC curves. For each class, convert the multi-class problem into a binary problem. Also, 
#    call the roc() function specifying 2 arguments: i) observed classes and ii) class probability (instead of predicted class).
# 1 ROC curve, Moderate, Good, UHeal vs non Moderate non Good non UHeal
roc.yes <- roc(ifelse(predictions1$observed=="yes","no-yes","yes"), as.numeric(predictions1$yes))
roc.no <- roc(ifelse(predictions1$observed=="no","no-no", "no"), as.numeric(predictions1$no))

plot(roc.no, col = "green", main="Random Forest best tune prediction ROC plot using testing data", xlim=c(0.44,0.1))
lines(roc.yes, col = "red")

par(pty = "s") #to remove the ugly's spaces at each side
plot(roc.yes, col = "red", plot=TRUE, legacy.axes=TRUE) #to remove the ugly's spaces at each side and put 1-specificity

# calculating the values of AUC for ROC curve
results= c("Yes AUC" = roc.yes$auc) #,"No AUC" = roc.no$auc)
print(results)
legend("right",c("AUC = 0.95 "),fill=c("red"),inset = (0.01))

#Important note: In previous course (Prediction using ANN "regression") prediction rate = 0.80 using .
summary(All_incidents)

All_incidents <-na.omit(All_incidents)

set.seed(849)
fit.rfAll<- train(Slides~., 
                  data=All_incidents,
                  method = "rf",
                  metric = "Accuracy",
                  trControl = trControl,
                  importance = TRUE)

X.rfAll = varImp(fit.rfAll)
ggplot(X.rfAll, top = 20) ###it can be worked
X.rfAll
# Plot graph
# 1. Open jpeg file
jpeg("varImportance All RF.jpg", width = 800, height = 500)
# 2. Create the plot
ggplot(X.rfAll,main="varImportanceAll RF" )
plot(fit.rfAll$results)

colnames(alljune)[colnames(alljune)=="l3"] <- "Population on slopes"

# 3. Close the file
dev.off()


# 6  Produce prediction map using Raster data ---------------------------


# 6-1 Import and process thematic maps ------------------------------------


#Produce LSM map using Training model results and Raster layers data

# Import Raster
install.packages("raster")
install.packages("rgdal")
library(raster)
library(rgdal)


# load all the data

# Load the Raster data
ELEV = raster("C:/LS_RF/Raster/MDE_10px_Corrigido.tif")  
SLOPE= raster("C:/LS_RF/Raster/Declividade_10px_Corrigido.tif") 
PROFC= raster("C:/LS_RF/Raster/ProfileCurvature_10px_Corrigido.tif") 
TWI= raster("C:/LS_RF/Raster/TWI_Corrigido.tif") 
PLANC=raster("C:/LS_RF/Raster/PlanCurvature_10px_Corrigido.tif")
ASP=raster("C:/LS_RF/Raster/ASP_10px.tif")
LANDUSE=raster("C:/LS_RF/Raster/LandUse_10px.tif")
NDVI=raster("C:/LS_RF/Raster/NDVI10px_Reprojetado.tif") 
GEO=raster("C:/LS_RF/Raster/Geologia10px_IPT1986b.tif") 
GEOM=raster("C:/LS_RF/Raster/Geomorfologia10px_RR.tif") 



# check attributes and projection and extent
extent(ELEV)
extent(SLOPE)
extent(TWI)
extent(PROFC)
extent(ASP)
extent(PLANC)
extent(LANDUSE)
extent(GEO)
extent(GEOM)
extent(NDVI)


# if you have different extent, then try to Resample them using the smallest area
ELEV_r <- resample(ELEV,GEO, resample='bilinear') 
SLOPE_r <- resample(SLOPE,GEO, resample='bilinear') 
TWI_r <- resample(TWI,GEO, resample='bilinear') 
PROFC_r <- resample(PROFC,GEO, resample='bilinear') 
PLANC_r <- resample(PLANC,GEO, resample='bilinear') 
LANDUSE_r <- resample(LANDUSE,GEO, resample='bilinear') 
GEOM_r <- resample(GEOM,GEO, resample='bilinear') 
NDVI_r <- resample(NDVI,GEO, resample='bilinear') 
ASP_r <- resample(ASP,GEO, resample='bilinear') 

extent(ASP_r) # check the new extent
extent(GEO)

# write to a new geotiff file
# Create new folder in WD using manually or in R studio (lower right pan)
writeRaster(ASP_r,filename="C:/LS_RF/resampled/ASP.tif", format="GTiff", overwrite=TRUE) 
writeRaster(PROFC_r,filename="C:/LS_RF/resampled/PROFC.tif", format="GTiff", overwrite=TRUE)
writeRaster(PLANC_r,filename="C:/LS_RF/resampled/PLANC.tif", format="GTiff", overwrite=TRUE)
writeRaster(TWI_r,filename="C:/LS_RF/resampled/TWI.tif", format="GTiff", overwrite=TRUE)
writeRaster(ELEV_r,filename="C:/LS_RF/resampled/ELEV.tif", format="GTiff", overwrite=TRUE)
writeRaster(SLOPE_r,filename="C:/LS_RF/resampled/SLOPE.tif", format="GTiff", overwrite=TRUE)
writeRaster(LANDUSE_r,filename="C:/LS_RF/resampled/LANDUSE.tif", format="GTiff", overwrite=TRUE)
writeRaster(GEO,filename="C:/LS_RF/resampled/GEO.tif", format="GTiff", overwrite=TRUE)
writeRaster(GEOM_r,filename="C:/LS_RF/resampled/GEOM.tif", format="GTiff", overwrite=TRUE)
writeRaster(NDVI_r,filename="C:/LS_RF/resampled/NDVI.tif", format="GTiff", overwrite=TRUE)


#Stack_List= stack(ASPECT_r,LS_r)#,pattern = "tif$", full.names = TRUE)
#names(Stack_List)
#Stack_List.df = as.data.frame(Stack_List, xy = TRUE, na.rm = TRUE)
#head(Stack_List.df,1)


## stack multiple raster files
Stack_List= list.files(path = "C:/LS_RF/resampled/",pattern = "tif$", full.names = TRUE) #mudar pra = se der errado
Rasters=stack(Stack_List) #mudar pra = se der errado

names(Rasters)


# 6-1-1 Convert rasters to dataframe with Long-Lat -----------------------
#Convert raster to dataframe with Long-Lat
Rasters.df = as.data.frame(Rasters, xy = TRUE, na.rm = TRUE)
head(Rasters.df,1)


# Now:Prediction using imported Rasters

# check the varaibles names to match with training data
#colnames(Rasters.df)[4] <- "ElEVATION"   # change columns names 
#colnames(Rasters.df)[6] <- "SlOPE"   # change columns names 

#head(Rasters.df[,c(-9,-10)],1)
#head(nn.ce$covariate,1)

Rasters.df_N <- Rasters.df[,c(-11,-12)] # remove x, y


# 6-1-2 Dealing with Categorial data --------------------------------------


# Dealing with Categorial data (Converting numeric variable into groups in R)
#https://www.r-bloggers.com/from-continuous-to-categorical/

# ASPECT
ASPras<-cut(Rasters.df_N$ASP, seq(0,361,45), right=FALSE, labels=c("a","b","c","d","e","f","g","h"))
table(ASPras)


# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top

ASPras <- factor(ASPras)
flagsras = data.frame(Reduce(cbind, 
                             lapply(levels(ASPras), function(x){(ASPras == x)*1})
))
names(flagsras) = levels(ASPras)
Rasters.df_N = cbind(Rasters.df_N, flagsras) # combine the ASPECTS with original data

# Remove the original aspect data
Rasters.df_N<- Rasters.df_N[,-1]
str(Rasters.df_N)


# LANDCOVER

# Dealing with Categorial data (Converting numeric variable into groups in R)
#https://www.r-bloggers.com/from-continuous-to-categorical/
LANDUSEras<-cut(Rasters.df_N$LANDUSE, seq(1,11,1), right=FALSE, labels=c("l1","l2","l3","l4","l5","l6","l7","l8","L9","l10"))
table(LANDUSEras)
class(LANDUSEras)

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top

flagsras = data.frame(Reduce(cbind, 
                             lapply(levels(LANDUSEras), function(x){(LANDUSEras == x)*1})
))
names(flagsras) = levels(LANDUSEras)
Rasters.df_N = cbind(Rasters.df_N, flagsras) # combine the LANDCOVER with original data


# GEO

# Dealing with Categorial data (Converting numeric variable into groups in R)
#https://www.r-bloggers.com/from-continuous-to-categorical/
GEOras<-cut(Rasters.df_N$GEO, seq(1,16,1), right=FALSE, labels=c("g1","g2","g3","g4","g5","g6","g7","g8","g9","g10","g11","g12","g13","g14","g15"))
table(GEOras)
class(GEOras)

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top

flagsras = data.frame(Reduce(cbind, 
                             lapply(levels(GEOras), function(x){(GEOras == x)*1})
))
names(flagsras) = levels(GEOras)
Rasters.df_N = cbind(Rasters.df_N, flagsras) # combine the GEO with original data


# GEOM

# Dealing with Categorial data (Converting numeric variable into groups in R)
#https://www.r-bloggers.com/from-continuous-to-categorical/
GEOMras<-cut(Rasters.df_N$GEOM, seq(1,8,1), right=FALSE, labels=c("gm1","gm2","gm3","gm4","gm5","gm6","gm7"))
table(GEOMras)
class(GEOMras)

# Dealing with Categorial data
#https://stackoverflow.com/questions/27183827/converting-categorical-variables-in-r-for-ann-neuralnet?answertab=oldest#tab-top

GEOMras <- factor(GEOMras)
flagsras = data.frame(Reduce(cbind, 
                             lapply(levels(GEOMras), function(x){(GEOMras == x)*1})
))
names(flagsras) = levels(GEOMras)
Rasters.df_N = cbind(Rasters.df_N, flagsras) # combine the GEOM with original data


# Remove the original LANDCOVER data
Rasters.df_N<- Rasters.df_N[,-4] #REMOVE LANDUSE
Rasters.df_N<- Rasters.df_N[,-2] #REMOVE GEO
Rasters.df_N<- Rasters.df_N[,-2] #REMOVE GEOM
str(Rasters.df_N)


# 6-1-3 Scale the numeric variables --------------------------------------

# Check the relationship between the numeric varaibles, Scale the numeric var first!
maxss <- apply(Rasters.df_N, 2, max) 
minss <- apply(Rasters.df_N, 2, min)
Rasters.df_N_scaled <- as.data.frame(scale(Rasters.df_N, center = minss, scale = maxss - minss)) # we removed the Aspect levels because it might be changed to NA!
colnames(Rasters.df_N_scaled)[colnames(Rasters.df_N_scaled)=="CURVATURE"] <- "CURVE"

rm(Rasters.df_N)
Rasters.df_N_scaled2 <-na.omit(Rasters.df_N_scaled) ###FOLLOW THIS PATH IF ERROR
Rasters.df <-na.omit(Rasters.df)

# PRODUCE PROBABILITY MAP
p3<-as.data.frame(predict(fit.rfAll, Rasters.df_N_scaled, type = "prob"))
summary(p3)
Rasters.df$Levels_yes<-p3$yes
Rasters.df$Levels_no<-p3$no

x<-SpatialPointsDataFrame(as.data.frame(Rasters.df)[, c("x", "y")], data = Rasters.df)
r_ave_yes <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_yes")])
proj4string(r_ave_yes)=CRS(projection(ELEV))

r_ave_no <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Levels_no")])
proj4string(r_ave_no)=CRS(projection(ELEV))


# Plot Maps
spplot(r_ave_yes, main="Landslides SM prob using RF")
writeRaster(r_ave_yes,filename="RunX Prediction_RF Tunned_LandslidesYes prob SMJune.tif", format="GTiff", overwrite=TRUE) 

spplot(r_ave_no, main="Non Slide prob RF")
writeRaster(r_ave_no,filename="RunX Prediction_RF Tunned_Non Slide prob SM.tif", format="GTiff", overwrite=TRUE) 


# PRODUCE CLASSIFICATION MAP
#Prediction at grid location
p3<-as.data.frame(predict(fit.rfAll, Rasters.df_N_scaled, type = "raw"))
summary(p3)
# Extract predicted levels class
head(Rasters.df, n=2)
Rasters.df$Levels_Slide_No_slide<-p3$`predict(fit.rfAll, Rasters.df_N_scaled, type = "raw")`
head(Rasters.df, n=2)

# Import levels ID file 
ID<-read.csv("C:/LS_RF/Excel/Levels_key.csv", header = TRUE, sep = ",")

# Join landuse ID
grid.new<-join(Rasters.df, ID, by="Levels_Slide_No_slide", type="inner") 
# Omit missing values
grid.new.na<-na.omit(grid.new)  
head(grid.new.na, n=2)

#Convert to raster
x<-SpatialPointsDataFrame(as.data.frame(grid.new.na)[, c("x", "y")], data = grid.new.na)
r_ave_Slide_No_slide <- rasterFromXYZ(as.data.frame(x)[, c("x", "y", "Level_ID")])

# coord. ref. : NA 
# Add coord. ref. system by using the original data info (Copy n Paste).
# borrow the projection from Raster data
proj4string(r_ave_Slide_No_slide)=CRS(projection(ELEV)) # set it to lat-long

# Export final prediction map as raster TIF ---------------------------
# write to a new geotiff file
writeRaster(r_ave_Slide_No_slide,filename="Run4 Classification_Map RF Tunned SLIDE_NO SLIDE.tif", format="GTiff", overwrite=TRUE) 


#Plot Landuse Map:
# Color Palette follow Air index color style
#https://bookdown.org/rdpeng/exdata/plotting-and-color-in-r.html

myPalette <- colorRampPalette(c("light green","red" ))

# Plot Map
LU_ave<-spplot(r_ave_Slide_No_slide,"Level_ID", main="Landslide prediction: RF tunned" , 
               colorkey = list(space="right",tick.number=1,height=1, width=1.5,
                               labels = list(at = seq(1,4.8,length=5),cex=1.0,
                                             lab = c("Yes" ,"No"))),
               col.regions=myPalette,cut=4)
LU_ave
jpeg("Prediction_Map RF_Landslide .jpg", width = 1000, height = 700)
LU_ave
dev.off()

##########DDDDDDDDDDDDOOOOOOOOOOOOOOOONNNNNNNNNNNNNNEEEEEEEEE :) :)






