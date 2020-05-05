
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)
library(doParallel)

# Graphics sep up
trellis.par.set(caretTheme())

# DATA SPLITTING
num_target <- 'price'
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

target.num.NoHyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # BRIDGE: BAYESIAN RIDGE REGRESSION: https://cran.r-project.org/web/packages/monomvn/
  # BLASSOAVERAGED: BAYESIAN RIDGE REGRESSION (MODEL AVERAGED): https://cran.r-project.org/web/packages/monomvn/
  # LMSTEPAIC: LINEAR REGRESSION WITH STEPWISE SELECTION: https://cran.r-project.org/web/packages/MASS/  
  # NNLS: NON-NEGATIVE LEAST SQUARES: https://cran.r-project.org/web/packages/nnls/
  fast.models <- c("bridge", "blassoAveraged", "lmStepAIC", "nnls")
  
  # RVMLINEAR: RELEVANCE VECTOR MACHINES WITH LINEAR KERNEL: https://cran.r-project.org/web/packages/kernlab/ 
  slow.models <- c("rvmLinear")
 
  if (parallel) {
    cl <- makePSOCKcluster(3)
    registerDoParallel(cl)
  }
  
  models.Results <- list()
  for (model in fast.models) {
    model.Fit <- train(fmla, data = dataTrain, 
                          method = model, 
                          trControl = fitControl)
    models.Results <- rbind(models.Results, model.Fit$results)
  }
  
  if (slow) {
    for (model in slow.models) {
      model.Fit <- train(fmla, data = dataTrain, 
                         method = model, 
                         trControl = fitControl)
      models.Results <- rbind(models.Results, model.Fit$results)
    }
  }
  
  if (parallel) {
    stopCluster(cl)
    registerDoSEQ()
  }
  
  return(models.Results)
}


############################################################################
####################### MODELS WITH HYPERPARAMETERS ########################
############################################################################

target.num.Hyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # LM: LINEAR REGRESSION: BUILT IN
  # CUBIST: RULE AND INSTANCE BASED REGRESSION MODELLING: https://cran.r-project.org/web/packages/Cubist/
  # ENET: ELASTIC-NET FOR SPARSE ESTIMATION AND SPARSE PCA: https://cran.r-project.org/web/packages/elasticnet/
  # ICR: INDEPENDENT COMPONENT REGRESSION: https://cran.r-project.org/web/packages/fastICA/
  # LARS,LARS2: LEAST ANGLE REGRESSION: https://cran.r-project.org/web/packages/lars/
  # LEAPBACKWARD: LINEAR REGRESSION WITH BACKWARDS SELECTION: https://cran.r-project.org/web/packages/leaps/
  # LEAPFORWARD: LINEAR REGRESION WITH FORWARD SELECTION: https://cran.r-project.org/web/packages/leaps/
  # LEAPSEQ: LINREAR REGRESSION WITH STEPWISE SELECTION: https://cran.r-project.org/web/packages/leaps/
  # PENALIZED: PENALIZED LINEAR REGRESSION: https://cran.r-project.org/web/packages/penalized/
  # PCR: PRINCIPAL COMPONENT ANALYSIS: https://cran.r-project.org/web/packages/pls/
  # PPR: PROJECTION PURSUIT REGRESSION: BUILT IN
  # QRF: QUANTILE RANDOM FOREST: https://cran.r-project.org/web/packages/quantregForest/
  # RQLASSO: QUANTILE REGRESSION WITH LASSO PENALTY: https://cran.r-project.org/web/packages/rqPen/
  # RIDGE: RISGE REGRESSION: https://cran.r-project.org/web/packages/elasticnet/
  # FOBA: RIDGE REGRESSION WITH VARIABLE SELECTION: https://cran.r-project.org/web/packages/foba/
  # RLM: ROBUST LINEAR MODEL: https://cran.r-project.org/web/packages/MASS/
  # SPIKESLAB: SPIKE AND SLAB REGRESSION: https://cran.r-project.org/web/packages/spikeslab/  : https://cran.r-project.org/web/packages/plyr/
  # BLASSO: THE BAYESIAN LASSO: https://cran.r-project.org/web/packages/monomvn/
  # LASSO: THE LASSO: https://cran.r-project.org/web/packages/elasticnet/
  fast.models <- c("lm", "cubist", "enet", "icr", "lars", "lars2", "leapBackward", "leapForward",
                   "leapSeq", "penalized", "pcr", "ppr", "qrf", "rqlasso", "ridge", "foba",
                   "rlm", "spikeslab", "blasso", "lasso")
  lm.Grid <-  expand.grid(intercept = TRUE)
  cubist.Grid <-  expand.grid(committees = c(1, 5, 9), neighbors = 1:9)
  enet.Grid <-  expand.grid(fraction = c(0.1, 0.5, 0.9), lambda = seq(0.01,0.09,0.01))
  icr.Grid <-  expand.grid(n.comp = c(1:9))
  lars.Grid <-  expand.grid(fraction = seq(0.1,0.9,0.1))
  lars2.Grid <-  expand.grid(step = 1:10)
  leapBackward.Grid <-  expand.grid(nvmax = 16)
  leapForward.Grid <-  expand.grid(nvmax = 16)
  leapSeq.Grid <-  expand.grid(nvmax = 16)
  penalized.Grid <-  expand.grid(lambda1 = seq(0.1,0.9,0.1), lambda2 = c(0.1,0.2))
  pcr.Grid <-  expand.grid(ncomp = c(2,4,6,16))
  ppr.Grid <-  expand.grid(nterms = c(2,4,6,16))
  qrf.Grid <-  expand.grid(mtry = 2)
  rqlasso.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.1))
  ridge.Grid <-  expand.grid(lambda = seq(0.01,0.9,0.01))
  foba.Grid <-  expand.grid(k = 1, lambda = seq(0.1,0.9,0.1))
  rlm.Grid <-  expand.grid(intercept = TRUE, psi = seq(0.1,0.9,0.1))
  spikeslab.Grid <-  expand.grid(vars = 1:10)
  blasso.Grid <-  expand.grid(sparsity = seq(0.1,0.9,0.1))
  lasso.Grid <-  expand.grid(fraction = seq(0.1,0.9,0.1))
  
  # KRLS: POLYNOMIAL KERNEL REGULARIZED LEAST SQUERES: https://cran.r-project.org/web/packages/KRLS/
  # KRLSRADIAL: RADIAL BASIS FUNCTION KERNEL REGULARIZED LEAST SQUARES: https://cran.r-project.org/web/packages/KRLS/ : ttps://cran.r-project.org/web/packages/kernlab/ 
  # RELAXO: RELAXED LASSO: https://cran.r-project.org/web/packages/relaxo/ : https://cran.r-project.org/web/packages/plyr/
  # RVMPOLY: RELEVANCE VECTOR MACHINES WITH POLYNOMIAL KERNEL: https://cran.r-project.org/web/packages/kernlab/ 
  # RVMRADIAL: RELEVANCE VECTOR MACHINES WITH RADIAL BASIS FUNCTION KERNEL: https://cran.r-project.org/web/packages/kernlab/ 
  slow.models <- c("krlsPoly", "krlsRadial", "relaxo", "rvmPoly", "rvmRadial")
  krls.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.1), degree = c(1,2))
  krlsRadial.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.1), sigma = 0.1)
  relaxo.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.1), phi = 0.99)
  rvmPoly.Grid <-  expand.grid(scale = seq(0.1,0.9,0.1), degree = 3)
  rvmRadial.Grid <-  expand.grid(sigma = seq(0.1,0.9,0.1))
  
  if (parallel) {
    cl <- makePSOCKcluster(3)
    registerDoParallel(cl)
  }
  
  models.Results <- list()
  for (model in fast.models) {
    model.Grid <- eval(parse(text = paste(model, '.Grid', sep='')))
    model.Fit <- train(fmla, data = dataTrain, 
                       method = model, 
                       trControl = fitControl,
                       tuneGrid = model.Grid)
    #models.Results <- rbind(models.Results, model.Fit$results)
    #plot(model.Fit)
    print(model.Fit$results)
  }
  
  if (slow) {
    for (model in slow.models) {
      model.Grid <- eval(parse(text = paste(model, '.Grid', sep='')))
      model.Fit <- train(fmla, data = dataTrain, 
                         method = model, 
                         trControl = fitControl,
                         tuneGrid = model.Grid)
      
      #models.Results <- rbind(models.Results, model.Fit$results)
      # plot(model.Fit)
    }
  }
  
  if (parallel) {
    stopCluster(cl)
    registerDoSEQ()
  }
  
  return(models.Results)
}


############################################################################
####################### MODELS NOT WORKING  ################################
############################################################################

# SUPERPC: SUPERVISED PRINCIPAL COMPONENT ANALYSIS --- ERROR: STOPING ---
# https://cran.r-project.org/web/packages/superpc/

SUPERPC.Grid <-  expand.grid(threshold = seq(0.1,0.5,0.1), n.components = 2:4)

SUPERPC.Fit1 <- train(fmla, data = dataTrain, 
                        method = "superpc", 
                        trControl = fitControl,
                        tuneGrid = SUPERPC.Grid)
SUPERPC.Fit1
plot(SUPERPC.Fit1)


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









