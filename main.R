
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)
library(rlist)
library(doParallel)

# Load internal libreries
source('data_cleaning.R')
source('ModellingNum.R')
source('ModellingCat.R')

# Graphics sep up
trellis.par.set(caretTheme())

# Clean the data 
clean_data <- cleaning()


# Data regression
# DATA SPLITTING
num_target <- 'price'
fmla <- fmla <- as.formula(paste(num_target, '~.'))

dataset_name <- deparse(substitute(clean_data))
y <- eval(parse(text = paste(dataset_name, '$', num_target, sep='')))
trainIndex <- createDataPartition(y, p = .1, list = FALSE, times = 1)

dataTrain <- clean_data[trainIndex,]
dataTest <- clean_data[-trainIndex,]


# RESAMPLING: 10-fold CV repeated ten times
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

model.Results <- target.num.NoHyper(fmla, dataTrain, fitcontrol, parallel = FALSE, slow = FALSE)
model.Results <- target.num.Hyper(fmla, dataTrain, fitcontrol, parallel = FALSE, slow = FALSE)


# Data classification
# Create binary variable
a <- clean_data
a$binPrice <- ifelse(clean_data$price <= mean(clean_data$price), 'cheap', 'expensive')
a$binPrice[a$beds > 1] <- 'medium'
a$binPrice <- factor(a$binPrice)
a$price <- NULL

clean_data$binPrice <- factor(ifelse(clean_data$price <= mean(clean_data$price), 'cheap', 'expensive'))

clean_data$price <- NULL

# DATA SPLITTING
cat_target <- 'binPrice'
fmla <- fmla <- as.formula(paste(cat_target, '~.'))

dataset_name <- deparse(substitute(clean_data))
y <- eval(parse(text = paste(dataset_name, '$', cat_target, sep='')))
trainIndex <- createDataPartition(y, p = .8, list = FALSE, times = 1)

dataTrain <- clean_data[trainIndex,]
dataTest <- clean_data[-trainIndex,]

# RESAMPLING: 10-fold CV repeated ten times
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

model.Results <- target.catMult.NoHyper(fmla, dataTrain, fitControl, parallel = FALSE)
model.Results <- target.catBin.Hyper(fmla, dataTrain, fitControl, parallel = FALSE, slow = FALSE)
model.Results <- target.catMult.Hyper(fmla, dataTrain, fitControl, parallel = FALSE, slow = FALSE)

