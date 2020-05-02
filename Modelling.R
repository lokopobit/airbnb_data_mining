
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
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

# GBM: STOCHASTIC GRADIENT BOOSTING
# https://cran.r-project.org/web/packages/gbm/index.html

gbmGrid <-  expand.grid(interaction.depth = c(5, 9, 15), 
                        n.trees = (1:5)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

gbmFit1 <- train(fmla, data = dataTrain, 
                 method = "gbm", 
                 trControl = fitControl,
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

CUBISTGrid <-  expand.grid(committees = c(1, 5, 9), neighbors = 1:9)

CUBISTFit1 <- train(fmla, data = dataTrain, 
                 method = "cubist", 
                 trControl = fitControl,
                 verbose = FALSE,
                 tuneGrid = CUBISTGrid)
CUBISTFit1
plot(CUBISTFit1) 


# ELASTICNET: ELASTIC-NET FOR SPARSE ESTIMATION AND SPARSE PCA
# https://cran.r-project.org/web/packages/elasticnet/

ENETGrid <-  expand.grid(fraction = c(0.1, 0.5, 0.9), lambda = seq(0.01,0.09,0.01))

ENETFit1 <- train(fmla, data = dataTrain, 
                    method = "enet", 
                    trControl = fitControl,
                    verbose = FALSE,
                    tuneGrid = ENETGrid)
ENETFit1
plot(ENETFit1) 


# FASTICA: INDEPENDENT COMPONENT REGRESSION
# https://cran.r-project.org/web/packages/fastICA/

FASTICAGrid <-  expand.grid(n.comp = c(1:9))

FASTICAFit1 <- train(fmla, data = dataTrain, 
                  method = "icr", 
                  trControl = fitControl,
                  verbose = FALSE,
                  tuneGrid = FASTICAGrid)
FASTICAFit1
plot(FASTICAFit1) 


# LARS: LEAST ANGLE REGRESSION
# https://cran.r-project.org/web/packages/lars/

LARSGrid1 <-  expand.grid(fraction = seq(0.1,0.9,0.1))

LARSFit1 <- train(fmla, data = dataTrain, 
                     method = "lars", 
                     trControl = fitControl,
                     tuneGrid = LARSGrid1)
LARSFit1
plot(LARSFit1) 

LARSGrid2 <-  expand.grid(step = 1:10)

LARSFit2 <- train(fmla, data = dataTrain, 
                  method = "lars2", 
                  trControl = fitControl,
                  tuneGrid = LARSGrid2)
LARSFit2
plot(LARSFit2)


# LM: LINEAR REGRESSION
# BUILT IN

LMGrid <-  expand.grid(intercept = TRUE)

LMFit1 <- train(fmla, data = dataTrain, 
                     method = "lm", 
                     trControl = fitControl,
                     tuneGrid = LMGrid)
LMFit1


# LEAPBACKWARD: LINEAR REGRESSION WITH BACKWARDS SELECTION
# LEAPFORWARD: LINEAR REGRESION WITH FORWARD SELECTION
# LEAPSEQ: LINREAR REGRESSION WITH STEPWISE SELECTION
# https://cran.r-project.org/web/packages/leaps/

LEAPBGrid <-  expand.grid(nvmax = 16)

LEAPBFit1 <- train(fmla, data = dataTrain, 
                method = "leapBackward", 
                trControl = fitControl,
                tuneGrid = LEAPBGrid)
LEAPBFit1

LEAPFGrid <-  expand.grid(nvmax = 16)

LEAPFFit1 <- train(fmla, data = dataTrain, 
                   method = "leapForward", 
                   trControl = fitControl,
                   tuneGrid = LEAPFGrid)
LEAPFFit1

LEAPSGrid <-  expand.grid(nvmax = 16)

LEAPSFit1 <- train(fmla, data = dataTrain, 
                   method = "leapSeq", 
                   trControl = fitControl,
                   tuneGrid = LEAPSGrid)
LEAPSFit1


# LMSTEPAIC: LINEAR REGRESSION WITH STEPWISE SELECTION
# https://cran.r-project.org/web/packages/MASS/

LMSTEPAICFit1 <- train(fmla, data = dataTrain, 
                   method = "lmStepAIC", 
                   trControl = fitControl)
LMSTEPAICFit1


# NEGATIVE BINOMIAL GENERILIZED MODEL --- NEEDS TO BE CHECKED
# https://cran.r-project.org/web/packages/MASS/

NBGrid <-  expand.grid(link = 'identity')

NBFit1 <- train(fmla, data = dataTrain, 
                   method = "glm.nb", 
                   trControl = fitControl,
                   tuneGrid = NBGrid)
NBFit1


# M5RULES: MODEL RULES
# M5: MODEL TREE
# https://cran.r-project.org/web/packages/RWeka/  --- PROBLEM WITH JAVA ---

M5RULESGrid <-  expand.grid(pruned = TRUE, smoothed = TRUE)

M5RULESFit1 <- train(fmla, data = dataTrain, 
                method = "M5Rules", # 'M5'
                trControl = fitControl,
                tuneGrid = M5RULESGrid)
M5RULESFit1


# RQNC: NON-CONVEX PENALIZED QUANTILE REGRESSION  -- OBJETO deriv_func no encontrado
# https://cran.r-project.org/web/packages/rqPen/

RQNCGrid <-  expand.grid(lambda = seq(0.1,0.9,0.1), penalty = c(0.1,0.2))

RQNCFit1 <- train(fmla, data = dataTrain, 
                     method = "rqnc", 
                     trControl = fitControl,
                     tuneGrid = RQNCGrid)
RQNCFit1


# NNLS: NON-NEGATIVE LEAST SQUARES 
# https://cran.r-project.org/web/packages/nnls/

NNLSFit1 <- train(fmla, data = dataTrain, 
                  method = "nnls", 
                  trControl = fitControl)
NNLSFit1


# PENALIZED: PENALIZED LINEAR REGRESSION
# https://cran.r-project.org/web/packages/penalized/

PENALIZEDGrid <-  expand.grid(lambda1 = seq(0.1,0.9,0.1), lambda2 = c(0.1,0.2))

PENALIZEDFit1 <- train(fmla, data = dataTrain, 
                  method = "penalized", 
                  trControl = fitControl,
                  tuneGrid = PENALIZEDGrid)
PENALIZEDFit1
plot(PENALIZEDFit1)


# KRLS: POLYNOMIAL KERNEL REGULARIZED LEAST SQUERES
# https://cran.r-project.org/web/packages/KRLS/

KRLSGrid <-  expand.grid(lambda = seq(0.1,0.9,0.1), degree = c(1,2))

KRLSFit1 <- train(fmla, data = dataTrain, 
                       method = "krlsPoly", 
                       trControl = fitControl,
                       tuneGrid = KRLSGrid)
KRLSFit1
plot(KRLSFit1)


# PLS: PRINCIPAL COMPONENT ANALYSIS
# https://cran.r-project.org/web/packages/pls/





















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

# NEURALNET: NEURAL NETWORK
# https://cran.r-project.org/web/packages/neuralnet/

frbsGrid <-  expand.grid(layer1 = 10, layer2=10, layer3=10)

frbsFit1 <- train(fmla, data = dataTrain, 
                  method = "neuralnet", 
                  trControl = fitControl,
                  tuneGrid = frbsGrid)
frsbFit1
plot(frbsFit1) 

