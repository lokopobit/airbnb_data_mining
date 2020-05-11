
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)
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

model.Grid <-  expand.grid(decay = seq(0.1,0.9,0.3))
model.Fit <- train(fmla, data = dataTrain, #dataTrain
                   method = "multinom", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)
getModelInfo('multinom')
plot(model.Fit)
model.Fit


model.Grid <-  expand.grid(maxvar = c(10,30), direction = c("both", "forward", "backward"))
garbage <- capture.output(model.Fit <- train(fmla, data = a, #dataTrain
                                             method = "stepLDA", 
                                             trControl = fitControl,
                                             tuneGrid = model.Grid))

getModelInfo('stepLDA')
plot(model.Fit)
model.Fit



###################### BINARY CLASSIFICATION ONLY ##########################
target.catBin.NoHyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # lda: LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/MASS/
  # Mlda: MAXIMUN UNCERTAINTY LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/HiDimDA/
  
  fast.models <- c("lda", "Mlda")
  
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


############################################################################
####################### MODELS WITH HYPERPARAMETERS ########################
############################################################################

###################### BINARY CLASSIFICATION ONLY ##########################
target.catBin.Hyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # ada: BOOSTED CLASSIFICATION TREES: https://cran.r-project.org/web/packages/ada/ : https://cran.r-project.org/web/packages/plyr/
  # C5.0Cost: COST-SENSITIVE C5.0: https://cran.r-project.org/web/packages/C50/ : https://cran.r-project.org/web/packages/plyr/
  # rpartCost: COST-SENSITIVE CART: https://cran.r-project.org/web/packages/rpart/ : https://cran.r-project.org/web/packages/plyr/
  # deepboost: DEEPBOOST: https://cran.r-project.org/web/packages/deepboost/
  # RFlda: FACTOR-BASED LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/HiDimDA/
  # fda: FLEXIBLE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/earth/ : https://cran.r-project.org/web/packages/mda/
  # protoclass: GREEDY PROPOTYPE SELECTION: https://cran.r-project.org/web/packages/protoclass/ : https://cran.r-project.org/web/packages/proxy/
  # hda: HETEROSCEDASTIC DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/hda/
  # hdda: HIGH DIMENSIONAL DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/HDclassif/
  # svmLinearWeights2: L2 REGULARIZED LINEAR SUPPORT VECTOR MACHINES WITH CLASS WEIGHTS: https://cran.r-project.org/web/packages/LiblineaR/
  # lvq: LEARNING VECTOR QUANTIZATION: https://cran.r-project.org/web/packages/class/
  # lssvmRadial: LEAST SQUARES SUPPORT VECTOR MACHINE WITH RADIAL BASIS FUNCTION KERNEL: https://cran.r-project.org/web/packages/kernlab/
  # lda2: LINEAR DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/MASS/
  # stepLDA: LINEAR DISCRIMINANT ANALYSIS WITH STEPWISE FEATURE SELECTION: https://cran.r-project.org/web/packages/MASS/ : https://cran.r-project.org/web/packages/klaR/
  # dwdLinear: LINEAR DISTANCE WEIGHTED DISCRIMINATION: https://cran.r-project.org/web/packages/kerndwd/
  # svmLinearWeights: LINEAR SUPPORT VECTOR MACHINES WITH CLASS WEIGHTS: https://cran.r-project.org/web/packages/e1071/
  # LMT: LOGISTIC MODEL TREES: https://cran.r-project.org/web/packages/RWeka/
  # mda: MIXTURE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/mda/
  # naive_bayes: NAIVE BAYES: https://cran.r-project.org/web/packages/naivebayes/
  # nb: NAIVE BAYES: https://cran.r-project.org/web/packages/klaR/
  # pam: NEAREST SHRUNKEN CENTROIDS: https://cran.r-project.org/web/packages/pamr/
  # ownn: OPTIMAL WEIGHTED NEAREST NEIGHBOR CLASSIFIER: https://cran.r-project.org/web/packages/snn/
  # PRIM: PATIENT RULE INDUCTION METHOD: https://cran.r-project.org/web/packages/supervisedPRIM/
  # pda: PENALIZED DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/mda/
  # pda: PENALIZED DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/mda/
  
  fast.models <- c("ada", "C5.0Cost", "rpartCost", "deepboost", "RFlda", "fda", "protoclass", "hda",
                   "hdda", "svmLinearWeights2", "lvq", "lssvmRadial", "lda2", "stepLDA", "dwdLinear",
                   "svmLinearWeights", "LMT", "mda", "naive_bayes", "nb", "pam", "ownn", "PRIM",
                   "pda", "pda2")
  ada.Grid <-  expand.grid(iter = 100, maxdepth = c(4, 6), nu = 0.5)
  C5.0Cost.Grid <-  expand.grid(trials = seq(10,30,10), model = c("tree", "rules"), winnow = c(TRUE, FALSE), cost = 1:3)
  rpartCost.Grid <-  expand.grid(cp = 1:3, Cost = 1:3)
  deepboost.Grid <-  expand.grid(num_iter = seq(10,30,20), tree_depth = 5:6, beta = seq(0.2,0.3,0.1), lambda = 0.3, loss_type = "l")
  RFlda.Grid <-  expand.grid(q = 2:4)
  fda.Grid <-  expand.grid(degree = 2:4, nprune = c(5,10))
  protoclass.Grid <-  expand.grid(eps = 50, Minkowski = 1)
  hda.Grid <-  expand.grid(gamma = seq(0.1,0.9,0.3), lambda = seq(0.2,0.9,0.3), newdim = c(2, 5, 10))
  hdda.Grid <-  expand.grid(threshold = seq(0.1,0.9,0.3), model = c("AkjBkQkDk", "AkBkQkDk", "ABkQkDk"))
  svmLinearWeights2.Grid <-  expand.grid(cost = c(0.1,0.9), Loss = c("L1","L2"), weight = c(5,15))
  lvq.Grid <-  expand.grid(size = c(1,2), k = c(5,15))
  lssvmRadial.Grid <-  expand.grid(sigma = seq(1,10,6), tau = seq(0.1,0.9,0.6))
  lda2.Grid <-  expand.grid(dimen = seq(1,30,3))
  stepLDA.Grid <-  expand.grid(maxvar = c(10,30), direction = c("both", "forward", "backward"))
  dwdLinear.Grid <-  expand.grid(lambda = seq(0.2,0.9,0.3), qval = c(5, 10))
  svmLinearWeights.Grid <-  expand.grid(cost = seq(2,9,3), weight = c(2, 4))
  LMT.Grid <-  expand.grid(iter = c(3, 10))
  mda.Grid <-  expand.grid(subclasses = c(3, 10))
  naive_bayes.Grid <-  expand.grid(usekernel = c(TRUE, FALSE), laplace = seq(0.1,0.9,0.2), adjust = seq(0.1,1,0.2))
  nb.Grid <-  expand.grid(usekernel = c(TRUE, FALSE), fL = seq(0.1,0.9,0.2), adjust = seq(0.1,1,0.2))
  pam.Grid <-  expand.grid(threshold = seq(0.1,1,0.22))
  ownn.Grid <-  expand.grid(K = c(2,5))
  PRIM.Grid <-  expand.grid(peel.alpha = seq(0.01,0.25,0.09), paste.alpha = seq(0.01,0.25,0.2), mass.min = seq(0.01,0.25,0.2))
  pda.Grid <-  expand.grid(lambda = seq(0.01,0.9,0.1))
  pda2.Grid <-  expand.grid(df = seq(1,20,10))
  
  # ADABOOST: ADABOOST CLASSIFICATION TREES: https://cran.r-project.org/web/packages/fastAdaboost/
  # ORFlog: OBLIQUE RANDOM FOREST: https://cran.r-project.org/web/packages/obliqueRF/
  # ORFpls: OBLIQUE RANDOM FOREST: https://cran.r-project.org/web/packages/obliqueRF/
  # ORFridge: OBLIQUE RANDOM FOREST: https://cran.r-project.org/web/packages/obliqueRF/
  # ORFsvm: OBLIQUE RANDOM FOREST: https://cran.r-project.org/web/packages/obliqueRF/
  # plr: PENALIZED LOGISTIC REGRESSION: https://cran.r-project.org/web/packages/stepPlr/
  
  slow.models <- c("adaboost", "ORFlog", "ORFpls", "ORFridge", "ORFsvm", "plr")
  adaboost.Grid <-  expand.grid(nIter = seq(100, 150, 25), method = "Adaboost.M1")
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
    tryCatch({message(paste(rep('-', 20), collapse = ''))
              message(paste('TRYING:', model, sep = ' '))
              garbage <- capture.output(model.Fit <- train(fmla, data = dataTrain, 
                       method = model, 
                       trControl = fitControl,
                       tuneGrid = model.Grid))
              print(model.Grid)
              print(model.Fit$results)},
             warning = function(cond) {
               message('WARNING:')
               message(cond)
               return(NA)},
              error = function(cond) {
                message('ERROR:')
                message(cond)
                message(paste(rep('-', 20), collapse = ''))
                return(NA)},
             finally = {
               message(paste(model, 'FINISHED',sep = ' '))})
    #models.Results <- rbind(models.Results, model.Fit$results)
    #plot(model.Fit)
    #print(model.Fit$results)
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

###################### MULTICLASS CLASSIFICATION #############################
target.catMult.Hyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # ADABAG: BAGGED ADABOOST: https://cran.r-project.org/web/packages/adabag/
  # ADABOOST.M1: ADABOOST.M1: https://cran.r-project.org/web/packages/adabag/
  # bagFDAGCV: BAGGED FDA USING gCV PRUNING: https://cran.r-project.org/web/packages/earth/
  # LogitBoost: BOOSTED LOGISTIC REGRESSION: https://cran.r-project.org/web/packages/caTools/
  # J48: C4.5-LIKE TREES: https://cran.r-project.org/web/packages/RWeka/
  # C5.0: C5.0: https://cran.r-project.org/web/packages/C50/
  # multinom: PENALIZED MULTINOMIAL REGRESSION: https://cran.r-project.org/web/packages/nnet/
  
  
  fast.models <- c("AdaBag", "AdaBoost.M1", "bagFDAGCV", "LogitBoost", "J48", "C5.0", "multinom")
  AdaBag.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10))
  AdaBoost.M1.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10), coeflearn = "Zhu")
  bagFDAGCV.Grid <-  expand.grid(degree = c(1, 2))
  LogitBoost.Grid <-  expand.grid(nIter = seq(100,300,100))
  J48.Grid <-  expand.grid(C = seq(0.01,0.05,0.01), M = 5)
  C5.0.Grid <-  expand.grid(trials = seq(20,100,20), model = c('rules', 'tree'), winnow = c(TRUE,FALSE))
  multinom.Grid <-  expand.grid(decay = seq(0.1,0.9,0.3))
  
  # bagFDA: BAGGED FLEXIBLE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/earth/ : https://cran.r-project.org/web/packages/mda/
  # dwdPoly: DISTANCE WEIGHTED DISCRIMINATION WITH POLYNOMIAL KERNEL: https://cran.r-project.org/web/packages/kerndwd/
  # dwdRadial: DISTANCE WEIGHTED DISCRIMINATION WITH RADIAL BASIS FUNCTION KERNEL: https://cran.r-project.org/web/packages/kerndwd/ : https://cran.r-project.org/web/packages/kernlab/ 
  
  slow.models <- c("bagFDA", "dwdPoly", "dwdRadial")
  bagFDA.Grid <-  expand.grid(degree = c(1, 2), nprune = c (1,2))
  dwdPoly.Grid <-  expand.grid(lambda = 0.1, qval = 1, degree = 1, scale = 1)
  dwdRadial.Grid <-  expand.grid(lambda = 0.1, qval = 1, sigma = 4)
  
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






# to multiclass: fda, protoclass, hda, hdda, lssvmRadial, lda2, stepLDA, dwdLinear, LMT, mda, naive_bayes, pam, ownn, pda, pda2.







