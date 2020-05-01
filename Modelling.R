
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)

# Graphics sep up
trellis.par.set(caretTheme())

# DATA SPLITTING
num_target <- 'price'
cat_target <- 'host_response_time'
fmla <- fmla <- as.formula(paste(num_target, '~.'))

dataset_name <- deparse(substitute(clean_data))
y <- eval(parse(text = paste(dataset_name, '$', num_target, sep='')))
trainIndex <- createDataPartition(y, p = .8, list = FALSE, times = 1)

dataTrain <- clean_data[trainIndex,]
dataTest <- clean_data[-trainIndex,]


# RESAMPLING: 10-fold CV repeated ten times
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 2)

# GBM: STOCHASTIC GRADIENT BOOSTING
# https://cran.r-project.org/web/packages/gbm/index.html

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:10)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

gbmFit1 <- train(fmla, data = dataTrain, 
                 method = "gbm", 
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE,
                 tuneGrid = gbmGrid)
gbmFit1
plot(gbmFit1) 


# MONOMVN: BAYESIAN RIDGE REGRESSION
# https://cran.r-project.org/web/packages/monomvn/

MONOMVNFit1 <- train(fmla, data = dataTrain, 
                  method = "bridge", 
                  trControl = fitControl)
MONOMVNFit1

MONOMVNFit2 <- train(fmla, data = dataTrain, 
                     method = "blassoAveraged", 
                     trControl = fitControl)
MONOMVNFit2


# CUBIST: RULE AND INSTANCE BASED REGRESSION MODELLING
# https://cran.r-project.org/web/packages/Cubist/

CUBISTGrid <-  expand.grid(committees = c(1, 5, 9, 15), neighbors = 1:9)

CUBISTFit1 <- train(fmla, data = dataTrain, 
                 method = "cubist", 
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE,
                 tuneGrid = CUBISTGrid)
CUBISTFit1
plot(CUBISTFit1) 


# ELASTICNET: ELASTIC-NET FOR SPARSE ESTIMATION AND SPARSE PCA
# https://cran.r-project.org/web/packages/elasticnet/

ENETGrid <-  expand.grid(fraction = c(0.1, 0.5, 0.9), lambda = seq(0.1,0.9,0.1))

ENETFit1 <- train(fmla, data = dataTrain, 
                    method = "enet", 
                    trControl = fitControl,
                    ## This last option is actually one
                    ## for gbm() that passes through
                    verbose = FALSE,
                    tuneGrid = ENETGrid)
ENETFit1
plot(ENETFit1) 


# FASTICA: INDEPENDENT COMPONENT REGRESSION
# https://cran.r-project.org/web/packages/fastICA/

FASTICAGrid <-  expand.grid(n.comp = c(1, 5, 9))

FASTICAFit1 <- train(fmla, data = dataTrain, 
                  method = "icr", 
                  trControl = fitControl,
                  ## This last option is actually one
                  ## for gbm() that passes through
                  verbose = FALSE,
                  tuneGrid = FASTICAGrid)
FASTICAFit1
plot(FASTICAFit1) 










# FRBS: ADAPTIVE-NETWORK-BASED FUZZY INFERENCE SYSTEM
# https://cran.r-project.org/web/packages/frbs/

frbsGrid <-  expand.grid(num.labels = c(1, 2), max.iter = (1:5)*2)

frbsFit1 <- train(fmla, data = dataTrain, 
                  method = "ANFIS", 
                  trControl = fitControl,
                  tuneGrid = frbsGrid)
frsbFit1
plot(frbsFit1) 
# Include DYNAMIC EVOLVING NEURAL-FUZZY INFERENCE SYSTEM
# Include FUZZY INFERENCE RULES BY DESCENT METHOD
# Include FUZZY RULES VIA MOGUL
# Include FUZZY RULES VIA THRIFT
# Include GENETIC LATERAL TUNING AND RULES SELECTION OF LINGUISTIC FUZZY SYSTEMS 
# Include HYBRID NEURAL FUZZY INFERENCE SYSTEM

# BRNN: BAYESIAN REGULARIZED NEURAL NETWORKS
# https://cran.r-project.org/web/packages/brnn/

