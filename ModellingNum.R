
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)
library(doParallel)

# Use parallel computing
cl <- makePSOCKcluster(3)
registerDoParallel(cl) # stopCluster(cl)

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

############################################################################
####################### MODELS WITHOUT HYPERPARAMETERS #####################
############################################################################

# MONOMVN: BAYESIAN RIDGE REGRESSION
# https://cran.r-project.org/web/packages/monomvn/

MONOMVN.Fit1 <- train(fmla, data = dataTrain, 
                  method = "bridge", 
                  trControl = fitControl)
MONOMVN.Fit1

MONOMVN.Fit2 <- train(fmla, data = dataTrain, 
                     method = "blassoAveraged", 
                     trControl = fitControl)
MONOMVN.Fit2


# LMSTEPAIC: LINEAR REGRESSION WITH STEPWISE SELECTION
# https://cran.r-project.org/web/packages/MASS/

LMSTEPAIC.Fit1 <- train(fmla, data = dataTrain, 
                       method = "lmStepAIC", 
                       trControl = fitControl)
LMSTEPAIC.Fit1


# NNLS: NON-NEGATIVE LEAST SQUARES 
# https://cran.r-project.org/web/packages/nnls/

NNLSFit1 <- train(fmla, data = dataTrain, 
                  method = "nnls", 
                  trControl = fitControl)
NNLSFit1


# RVMLINEAR: RELEVANCE VECTOR MACHINES WITH LINEAR KERNEL
# https://cran.r-project.org/web/packages/kernlab/ 

RVMLINEAR.Fit1 <- train(fmla, data = dataTrain, 
                        method = "rvmLinear", 
                        trControl = fitControl)
RVMLINEAR.Fit1
plot(RVMLINEAR.Fit1)

############################################################################
####################### MODELS WITH HYPERPARAMETERS ########################
############################################################################

# LM: LINEAR REGRESSION
# BUILT IN

LMGrid <-  expand.grid(intercept = TRUE)

LMFit1 <- train(fmla, data = dataTrain, 
                method = "lm", 
                trControl = fitControl,
                tuneGrid = LMGrid)
LMFit1


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

PLS.Grid <-  expand.grid(ncomp = c(2,4,6,16))

PLS.Fit1 <- train(fmla, data = dataTrain, 
                  method = "pcr", 
                  trControl = fitControl,
                  tuneGrid = PLS.Grid)
PLS.Fit1
plot(PLS.Fit1)


# PPR: PROJECTION PURSUIT REGRESSION
# BUILT IN

PPR.Grid <-  expand.grid(nterms = c(2,4,6,16))

PPR.Fit1 <- train(fmla, data = dataTrain, 
                  method = "ppr", 
                  trControl = fitControl,
                  tuneGrid = PPR.Grid)
PPR.Fit1
plot(PPR.Fit1)


# QRF: QUANTILE RANDOM FOREST
# https://cran.r-project.org/web/packages/quantregForest/

QRF.Grid <-  expand.grid(mtry = 2)

QRF.Fit1 <- train(fmla, data = dataTrain, 
                  method = "qrf", 
                  trControl = fitControl,
                  tuneGrid = QRF.Grid)
QRF.Fit1
plot(QRF.Fit1)


# RQLASSO: QUANTILE REGRESSION WITH LASSO PENALTY
# https://cran.r-project.org/web/packages/rqPen/

RQLASSO.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.1))

RQLASSO.Fit1 <- train(fmla, data = dataTrain, 
                  method = "rqlasso", 
                  trControl = fitControl,
                  tuneGrid = RQLASSO.Grid)
RQLASSO.Fit1
plot(RQLASSO.Fit1)


# KRLSRADIAL: RADIAL BASIS FUNCTION KERNEL REGULARIZED LEAST SQUARES
# https://cran.r-project.org/web/packages/KRLS/
# https://cran.r-project.org/web/packages/kernlab/ 

KRLSRADIAL.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.1), sigma = 0.1)

KRLSRADIAL.Fit1 <- train(fmla, data = dataTrain, 
                      method = "krlsRadial", 
                      trControl = fitControl,
                      tuneGrid = KRLSRADIAL.Grid)
KRLSRADIAL.Fit1
plot(KRLSRADIAL.Fit1)


# RELAXO: RELAXED LASSO
# https://cran.r-project.org/web/packages/relaxo/
# https://cran.r-project.org/web/packages/plyr/

RELAXO.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.1), phi = 0.99)

RELAXO.Fit1 <- train(fmla, data = dataTrain, 
                         method = "relaxo", 
                         trControl = fitControl,
                         tuneGrid = RELAXO.Grid)
RELAXO.Fit1
plot(RELAXO.Fit1)


# RVMPOLY: RELEVANCE VECTOR MACHINES WITH POLYNOMIAL KERNEL
# RVMRADIAL: RELEVANCE VECTOR MACHINES WITH RADIAL BASIS FUNCTION KERNEL
# https://cran.r-project.org/web/packages/kernlab/ 

RVMPOLY.Grid <-  expand.grid(scale = seq(0.1,0.9,0.1), degree = 3)

RVMPOLY.Fit1 <- train(fmla, data = dataTrain, 
                     method = "rvmPoly", 
                     trControl = fitControl,
                     tuneGrid = RVMPOLY.Grid)
RVMPOLY.Fit1
plot(RVMPOLY.Fit1)


RVMRADIAL.Grid <-  expand.grid(SIGMA = seq(0.1,0.9,0.1))

RVMRADIAL.Fit1 <- train(fmla, data = dataTrain, 
                      method = "rvmRadial", 
                      trControl = fitControl,
                      tuneGrid = RVMRADIAL.Grid)
RVMRADIAL.Fit1
plot(RVMRADIAL.Fit1)


# RIDGE: RISGE REGRESSION
# https://cran.r-project.org/web/packages/elasticnet/

RIDGE.Grid <-  expand.grid(lambda = seq(0.01,0.9,0.01))

RIDGE.Fit1 <- train(fmla, data = dataTrain, 
                        method = "ridge", 
                        trControl = fitControl,
                        tuneGrid = RIDGE.Grid)
RIDGE.Fit1
plot(RIDGE.Fit1)


# FOBA: RIDGE REGRESSION WITH VARIABLE SELECTION
# https://cran.r-project.org/web/packages/foba/

FOBA.Grid <-  expand.grid(k = 1, lambda = seq(0.1,0.9,0.1))

FOBA.Fit1 <- train(fmla, data = dataTrain, 
                    method = "foba", 
                    trControl = fitControl,
                    tuneGrid = FOBA.Grid)
FOBA.Fit1
plot(FOBA.Fit1)


# RLM: ROBUST LINEAR MODEL
# https://cran.r-project.org/web/packages/MASS/

RLM.Grid <-  expand.grid(intercept = TRUE, psi = seq(0.1,0.9,0.1))

RLM.Fit1 <- train(fmla, data = dataTrain, 
                   method = "rlm", 
                   trControl = fitControl,
                   tuneGrid = RLM.Grid)
RLM.Fit1
plot(RLM.Fit1)


# SPIKESLAB: SPIKE AND SLAB REGRESSION
# https://cran.r-project.org/web/packages/spikeslab/
# https://cran.r-project.org/web/packages/plyr/

SPIKESLAB.Grid <-  expand.grid(vars = 1:10)

SPIKESLAB.Fit1 <- train(fmla, data = dataTrain, 
                  method = "spikeslab", 
                  trControl = fitControl,
                  tuneGrid = SPIKESLAB.Grid)
SPIKESLAB.Fit1
plot(SPIKESLAB.Fit1)


# SUPERPC: SUPERVISED PRINCIPAL COMPONENT ANALYSIS
# https://cran.r-project.org/web/packages/superpc/

SUPERPC.Grid <-  expand.grid(threshold = seq(0.1,0.5,0.1), n.components = 2:4)

SUPERPC.Fit1 <- train(fmla, data = dataTrain, 
                        method = "superpc", 
                        trControl = fitControl,
                        tuneGrid = SUPERPC.Grid)
SUPERPC.Fit1
plot(SUPERPC.Fit1)


# BLASSO: THE BAYESIAN LASSO
# https://cran.r-project.org/web/packages/monomvn/

BLASSO.Grid <-  expand.grid(sparsity = 0.1)

BLASSO.Fit1 <- train(fmla, data = dataTrain, 
                      method = "blasso", 
                      trControl = fitControl,
                      tuneGrid = BLASSO.Grid)
BLASSO.Fit1
plot(BLASSO.Fit1)


# LASSO: THE LASSO
# https://cran.r-project.org/web/packages/elasticnet/

LASSO.Grid <-  expand.grid(fraction = seq(0.1,0.9,0.1))

LASSO.Fit1 <- train(fmla, data = dataTrain, 
                     method = "lasso", 
                     trControl = fitControl,
                     tuneGrid = LASSO.Grid)
LASSO.Fit1
plot(LASSO.Fit1)















