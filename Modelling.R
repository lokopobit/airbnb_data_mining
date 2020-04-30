
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)

# DATA SPLITTING
num_target <- 'price'
cat_target <- 'host_response_time'
fmla <- fmla <- as.formula(paste(num_target, '~.'))

y <- eval(parse(text = paste('clean_data', '$', num_target, sep='')))
trainIndex <- createDataPartition(y, p = .8, list = FALSE, times = 1)

dataTrain <- clean_data[trainIndex,]
dataTest <- clean_data[-trainIndex,]

## 10-fold CV ## repeated ten times
fitControl <- trainControl(method = "repeatedcv", number = 30, repeats = 10)

gbmFit1 <- train(fmla, data = dataTrain, 
                 method = "gbm", 
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
gbmFit1
