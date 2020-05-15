
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)
library(rlist)
library(doParallel)

# Graphics sep up
trellis.par.set(caretTheme())

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

############################################################################
####################### MODELS WITHOUT HYPERPARAMETERS #####################
############################################################################

model.Grid <-  expand.grid(diagonal = c(TRUE,FALSE), lambda = seq(0, 1, length = 0.3))
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "sda", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('sda')
plot(model.Fit)
model.Fit

###################### BINARY CLASSIFICATION ONLY ##########################
target.catBin.NoHyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # 
  fast.models <- c("")
  
  # 
  slow.models <- c("")
  
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

###################### MULTICLASS CLASSIFICATION #############################
target.catMult.NoHyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # lda: LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/MASS/
  # Mlda: MAXIMUN UNCERTAINTY LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/HiDimDA/
  # qda: QUADRATIC DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/MASS/
  # Linda: ROBUST LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/rrcov/
  # QdaCov: ROBUST QUADRATIC DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/rrcov/
  # RSimca: ROBUST SIMCA: https://cran.r-project.org/web/packages/rrcovHD/
  
  fast.models <- c("lda", "Mlda", "qda", "Linda", "QdaCov", "RSimca")
  
  # 
  slow.models <- c("")
  
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

###################### BINARY CLASSIFICATION ONLY ##########################
target.catBin.Hyper <- function(fmla, dataTrain, fitControl, parallel = TRUE, slow = FALSE) {
  # ADABOOST: ADABOOST CLASSIFICATION TREES: https://cran.r-project.org/web/packages/fastAdaboost/
  # ada: BOOSTED CLASSIFICATION TREES: https://cran.r-project.org/web/packages/ada/ : https://cran.r-project.org/web/packages/plyr/
  # C5.0Cost: COST-SENSITIVE C5.0: https://cran.r-project.org/web/packages/C50/ : https://cran.r-project.org/web/packages/plyr/
  # rpartCost: COST-SENSITIVE CART: https://cran.r-project.org/web/packages/rpart/ : https://cran.r-project.org/web/packages/plyr/
  # deepboost: DEEPBOOST: https://cran.r-project.org/web/packages/deepboost/
  # svmLinearWeights2: L2 REGULARIZED LINEAR SUPPORT VECTOR MACHINES WITH CLASS WEIGHTS: https://cran.r-project.org/web/packages/LiblineaR/
  # svmLinearWeights: LINEAR SUPPORT VECTOR MACHINES WITH CLASS WEIGHTS: https://cran.r-project.org/web/packages/e1071/
  # PRIM: PATIENT RULE INDUCTION METHOD: https://cran.r-project.org/web/packages/supervisedPRIM/
  # rotationForest: ROTATION FOREST: https://cran.r-project.org/web/packages/rotationForest/
  # rotationForestCp: ROTATION FOREST: https://cran.r-project.org/web/packages/rotationForest/
  
  fast.models <- c("adaboost", "ada", "C5.0Cost", "rpartCost", "deepboost", "svmLinearWeights2",
                   "svmLinearWeights", "PRIM", "rotationForest", "rotationForestCp")
  adaboost.Grid <-  expand.grid(nIter = 100, method = c("Adaboost.M1", "Real adaboost"))
  ada.Grid <-  expand.grid(iter = 100, maxdepth = c(4, 6), nu = 0.5)
  C5.0Cost.Grid <-  expand.grid(trials = seq(10,30,10), model = c("tree", "rules"), winnow = c(TRUE, FALSE), cost = 1:3)
  rpartCost.Grid <-  expand.grid(cp = 1:3, Cost = 1:3)
  deepboost.Grid <-  expand.grid(num_iter = seq(10,30,20), tree_depth = 5:6, beta = seq(0.2,0.3,0.1), lambda = 0.3, loss_type = "l")
  svmLinearWeights2.Grid <-  expand.grid(cost = c(0.1,0.9), Loss = c("L1","L2"), weight = c(5,15))
  svmLinearWeights.Grid <-  expand.grid(cost = seq(2,9,3), weight = c(2, 4))
  PRIM.Grid <-  expand.grid(peel.alpha = seq(0.01,0.25,0.09), paste.alpha = seq(0.01,0.25,0.2), mass.min = seq(0.01,0.25,0.2))
  rotationForest.Grid <-  expand.grid(K = seq(1,15,5), L = seq(1,15,5))
  rotationForestCp.Grid <-  expand.grid(K = seq(1,15,5), L = seq(1,15,5), cp = 0.1)
  
  # ORFlog: OBLIQUE RANDOM FOREST: https://cran.r-project.org/web/packages/obliqueRF/
  # ORFpls: OBLIQUE RANDOM FOREST: https://cran.r-project.org/web/packages/obliqueRF/
  # ORFridge: OBLIQUE RANDOM FOREST: https://cran.r-project.org/web/packages/obliqueRF/
  # ORFsvm: OBLIQUE RANDOM FOREST: https://cran.r-project.org/web/packages/obliqueRF/
  # plr: PENALIZED LOGISTIC REGRESSION: https://cran.r-project.org/web/packages/stepPlr/
  
  slow.models <- c("ORFlog", "ORFpls", "ORFridge", "ORFsvm", "plr")
  ORFlog.Grid <-  expand.grid(mtry = 2)
  ORFpls.Grid <-  expand.grid(mtry = 2)
  ORFridge.Grid <-  expand.grid(mtry = 2)
  ORFsvm.Grid <-  expand.grid(mtry = 2)
  plr.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.3), cp = c("aic", "bic"))
  
  if (parallel) {
    cl <- makePSOCKcluster(3)
    registerDoParallel(cl)
  }
  
  models.Results <- list()
  for (model in fast.models) {
    model.Grid <- eval(parse(text = paste(model, '.Grid', sep='')))
    model.Fit <- fitting(model.Grid, model, fmla, dataTrain, fitControl)
    browser
    if (!sum(is.na(model.Fit))) {models.Results[[model]] <- model.Fit}
    if (match(model,fast.models) != 1) {print(summary(resamples(models.Results)))}
  }
  
  if (slow) {
    for (model in slow.models) {
      model.Grid <- eval(parse(text = paste(model, '.Grid', sep='')))
      fitting(model.Grid, model, fmla, dataTrain, fitControl)
      
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

###################### MULTICLASS CLASSIFICATION #############################
target.catMult.Hyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # ADABAG: BAGGED ADABOOST: https://cran.r-project.org/web/packages/adabag/
  # ADABOOST.M1: ADABOOST.M1: https://cran.r-project.org/web/packages/adabag/
  # bagFDAGCV: BAGGED FDA USING gCV PRUNING: https://cran.r-project.org/web/packages/earth/
  # LogitBoost: BOOSTED LOGISTIC REGRESSION: https://cran.r-project.org/web/packages/caTools/
  # J48: C4.5-LIKE TREES: https://cran.r-project.org/web/packages/RWeka/
  # C5.0: C5.0: https://cran.r-project.org/web/packages/C50/
  # multinom: PENALIZED MULTINOMIAL REGRESSION: https://cran.r-project.org/web/packages/nnet/
  # RFlda: FACTOR-BASED LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/HiDimDA/
  # fda: FLEXIBLE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/earth/ : https://cran.r-project.org/web/packages/mda/
  # protoclass: GREEDY PROPOTYPE SELECTION: https://cran.r-project.org/web/packages/protoclass/ : https://cran.r-project.org/web/packages/proxy/
  # hda: HETEROSCEDASTIC DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/hda/
  # hdda: HIGH DIMENSIONAL DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/HDclassif/
  # lvq: LEARNING VECTOR QUANTIZATION: https://cran.r-project.org/web/packages/class/
  # lssvmRadial: LEAST SQUARES SUPPORT VECTOR MACHINE WITH RADIAL BASIS FUNCTION KERNEL: https://cran.r-project.org/web/packages/kernlab/
  # lda2: LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/MASS/
  # stepLDA: LINEAR DISCRIMINANT ANALYSIS WITH STEPWISE FEATURE SELECTION: https://cran.r-project.org/web/packages/MASS/ : https://cran.r-project.org/web/packages/klaR/
  # dwdLinear: LINEAR DISTANCE WEIGHTED DISCRIMINATION: https://cran.r-project.org/web/packages/kerndwd/ 
  # LMT: LOGISTIC MODEL TREES: https://cran.r-project.org/web/packages/RWeka/
  # mda: MIXTURE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/mda/
  # naive_bayes: NAIVE BAYES: https://cran.r-project.org/web/packages/naivebayes/
  # nb: NAIVE BAYES: https://cran.r-project.org/web/packages/klaR/
  # pam: NEAREST SHRUNKEN CENTROIDS: https://cran.r-project.org/web/packages/pamr/
  # ownn: OPTIMAL WEIGHTED NEAREST NEIGHBOR CLASSIFIER: https://cran.r-project.org/web/packages/snn/
  # pda: PENALIZED DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/mda/
  # pda: PENALIZED DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/mda/
  # stepQDA: QUADRATIC DISCRIMINANT ANALYSIS WITH STEPWISE FEATURE SELECTION: https://cran.r-project.org/web/packages/MASS/ : https://cran.r-project.org/web/packages/klaR/
  # rFerns: RANDOM FERNS: https://cran.r-project.org/web/packages/rFerns/: 
  # rda: REGURALIZED DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/klaR/
  # regLogistic: REGULARIZED LOGISTIC REGRESSION: https://cran.r-project.org/web/packages/LiblineaR/
  # rocc: ROC-BASED CLASSIFIER: https://cran.r-project.org/web/packages/rocc/
  # JRip: RULE-BASED CLASSIFIER: https://cran.r-project.org/web/packages/RWeka/
  # PART: RUE-BASED CLASSIFIER: https://cran.r-project.org/web/packages/RWeka/
  # sda: SHRINKAGE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/sda/
  
  fast.models <- c("AdaBag", "AdaBoost.M1", "bagFDAGCV", "LogitBoost", "J48", "C5.0", "multinom",
                   "RFlda", "fda", "protoclass", "hda", "hdda", "lvq", "lssvmRadial", "lda2", 
                   "stepLDA", "dwdLinear", "LMT", "mda", "naive_bayes", "nb", "pam", "ownn",
                   "pda", "pda2", "stepQDA", "rFerns", "rda", "regLogistic", "rocc", "JRip",
                   "PART", "sda")
  AdaBag.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10))
  AdaBoost.M1.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10), coeflearn = "Zhu")
  bagFDAGCV.Grid <-  expand.grid(degree = c(1, 2))
  LogitBoost.Grid <-  expand.grid(nIter = seq(100,300,100))
  J48.Grid <-  expand.grid(C = seq(0.01,0.05,0.01), M = 5)
  C5.0.Grid <-  expand.grid(trials = seq(20,100,20), model = c('rules', 'tree'), winnow = c(TRUE,FALSE))
  multinom.Grid <-  expand.grid(decay = seq(0.1,0.9,0.3))
  RFlda.Grid <-  expand.grid(q = 2:4)
  fda.Grid <-  expand.grid(degree = 2:4, nprune = c(5,10))
  protoclass.Grid <-  expand.grid(eps = 50, Minkowski = 1)
  hda.Grid <-  expand.grid(gamma = seq(0.1,0.9,0.3), lambda = seq(0.2,0.9,0.3), newdim = c(2, 5, 10))
  hdda.Grid <-  expand.grid(threshold = seq(0.1,0.9,0.3), model = c("AkjBkQkDk", "AkBkQkDk", "ABkQkDk"))
  lvq.Grid <-  expand.grid(size = c(1,2), k = c(5,15))
  lssvmRadial.Grid <-  expand.grid(sigma = seq(1,10,6), tau = seq(0.1,0.9,0.6))
  lda2.Grid <-  expand.grid(dimen = seq(1,30,3))
  stepLDA.Grid <-  expand.grid(maxvar = ncol(dataTrain), direction = c("both", "forward", "backward"))
  dwdLinear.Grid <-  expand.grid(lambda = seq(0.2,0.9,0.3), qval = c(5, 10))
  LMT.Grid <-  expand.grid(iter = c(3, 10))
  mda.Grid <-  expand.grid(subclasses = c(3, 10))
  naive_bayes.Grid <-  expand.grid(usekernel = c(TRUE, FALSE), laplace = seq(0.1,0.9,0.2), adjust = seq(0.1,1,0.2))
  nb.Grid <-  expand.grid(usekernel = c(TRUE, FALSE), fL = seq(0.1,0.9,0.2), adjust = seq(0.1,1,0.2))
  pam.Grid <-  expand.grid(threshold = seq(0.1,1,0.22))
  ownn.Grid <-  expand.grid(K = c(2,5))
  pda.Grid <-  expand.grid(lambda = seq(0.01,0.9,0.1))
  pda2.Grid <-  expand.grid(df = seq(1,20,10))
  stepQDA.Grid <-  expand.grid(maxvar = ncol(dataTrain), direction = 'both')
  rFerns.Grid <-  expand.grid(depth = c(1,8))
  rda.Grid <-  expand.grid(gamma = seq(0.1, 1, 0.3), lambda =  seq(0.1, 1, 0.3))
  regLogistic.Grid <-  expand.grid(cost = seq(0.1, 0.9, 0.3), loss = c("L1", "L2_dual", "L2_primal"), epsilon = 0.01)
  rocc.Grid <-  expand.grid(xgenes = 1:ncol(dataTrain))
  JRip.Grid <-  expand.grid(NumOpt  = seq(1,15,5), NumFolds  = seq(1,15,5), MinWeights  = 1)
  PART.Grid <-  expand.grid(threshold = seq(0.1,0.5,0.1), pruned = c("yes", "no"))
  sda.Grid <-  expand.grid(diagonal = c(TRUE,FALSE), lambda = seq(0, 1, length = 0.3))
  
  # bagFDA: BAGGED FLEXIBLE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/earth/ : https://cran.r-project.org/web/packages/mda/
  # dwdPoly: DISTANCE WEIGHTED DISCRIMINATION WITH POLYNOMIAL KERNEL: https://cran.r-project.org/web/packages/kerndwd/
  # dwdRadial: DISTANCE WEIGHTED DISCRIMINATION WITH RADIAL BASIS FUNCTION KERNEL: https://cran.r-project.org/web/packages/kerndwd/ : https://cran.r-project.org/web/packages/kernlab/ 
  # rmda: ROBUST MIXTURE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/robustDA/index.html
  
  slow.models <- c("bagFDA", "dwdPoly", "dwdRadial", "rmda")
  bagFDA.Grid <-  expand.grid(degree = c(1, 2), nprune = c (1,2))
  dwdPoly.Grid <-  expand.grid(lambda = 0.1, qval = 1, degree = 1, scale = 1)
  dwdRadial.Grid <-  expand.grid(lambda = 0.1, qval = 1, sigma = 4)
  rmda.Grid <-  expand.grid(K = seq(2,10,1), model = c("EII", "VII", "EEI", "EVI", "VEI", "VVI"))
  
  if (parallel) {
    cl <- makePSOCKcluster(3)
    registerDoParallel(cl)
  }
  
  models.Results <- list()
  for (model in fast.models) {
    model.Grid <- eval(parse(text = paste(model, '.Grid', sep='')))
    fitting(model.Grid, model, fmla, dataTrain, fitControl)
    #models.Results <- rbind(models.Results, model.Fit$results)
    #plot(model.Fit)
    #print(model.Fit$results)
  }
  
  if (slow) {
    for (model in slow.models) {
      model.Grid <- eval(parse(text = paste(model, '.Grid', sep='')))
      fitting(model.Grid, model, fmla, dataTrain, fitControl)
      
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


model.Grid <-  expand.grid(tau = 0.5)
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "lssvmLinear", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('lssvmLinear')
plot(model.Fit)
model.Fit


model.Grid <-  expand.grid(degree = 1, scale = 1, tau = 0.5)
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "lssvmPoly", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('lssvmPoly')
plot(model.Fit)
model.Fit


model.Grid <-  expand.grid(k = 3)
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "loclda", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('loclda')
plot(model.Fit)
model.Fit


# Error in .(lambda) : no se pudo encontrar la función "."
model.Grid <-  expand.grid(lambda = seq(0.1,0.9,0.3), K = 2)
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "PenalizedLDA", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('PenalizedLDA')
plot(model.Fit)
model.Fit


model.Grid <-  expand.grid(lambda = 0.1, hp = 0.5, penalty = c("L1", "L2"))
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "rrlda", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('rrlda')
plot(model.Fit)
model.Fit


# Data must be factor
model.Grid <-  expand.grid(k = 3, epsilon = 0.01, smooth = 1, final_smooth = 3, direction = c("forward", "backwards"))
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "nbSearch", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('nbSearch')
plot(model.Fit)
model.Fit

##########################################################################
###################### SPECIAL MODELS ####################################
##########################################################################

# Train data must be 0 or 1
model.Grid <-  expand.grid(lambda.freqs = c(0.1, 0.2))
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "binda", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)

plot(model.Fit)
model.Fit


# Data must be factor
model.Grid <-  expand.grid(smooth = c(3, 10), prior = 0.1)
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "manb", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('manb')
plot(model.Fit)
model.Fit


# Data must be factor
model.Grid <-  expand.grid(smooth = seq(1,10,2))
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "nbDiscrete", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('nbDiscrete')
plot(model.Fit)
model.Fit


# Data must be factor
model.Grid <-  expand.grid(smooth = seq(1,10,2))
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "awnb", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('awnb')
plot(model.Fit)
model.Fit


# Auxiliary function for fitting
fitting <- function(model.Grid, model, fmla, dataTrain, fitControl) {
tryCatch({message(paste(rep('-', 20), collapse = ''))
  st <- Sys.time()
  message(paste('TRYING:', model, sep = ' '))
  garbage <- capture.output(model.Fit <- train(fmla, data = dataTrain, 
                                               method = model, 
                                               trControl = fitControl,
                                               tuneGrid = model.Grid))
  # print(model.Fit$results)
  et <- Sys.time()
  print(et-st)
  return(model.Fit)},
  error = function(cond) {
    message('ERROR:')
    message(cond)
    return(NA)},
  finally = {
    message(paste(model, 'FINISHED',sep = ' '))})
}


