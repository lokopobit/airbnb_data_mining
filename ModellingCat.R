
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)
library(doParallel)

# Graphics sep up
trellis.par.set(caretTheme())

# Create binary variable
a <- clean_data
a$binPrice <- ifelse(clean_data$price <= mean(clean_data$price), 0, 1)
a$binPrice[a$beds > 1] <- 2
a$binPrice <- factor(a$binPrice)
a$price <- NULL

clean_data$binPrice <- factor(ifelse(clean_data$price <= mean(clean_data$price), 0, 1))

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


model.Grid <-  expand.grid(cp = seq(100,500,100), split = c('abs', 'quad'), prune = c('mr', 'mc'))
model.Fit <- train(fmla, data = a, #dataTrain
                   method = "chaid", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)

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

  fast.models <- c("ada")
  ada.Grid <-  expand.grid(iter = 100, maxdepth = c(4, 6), nu = 0.5)

  
  # ADABOOST: ADABOOST CLASSIFICATION TREES: https://cran.r-project.org/web/packages/fastAdaboost/
  
  slow.models <- c("adaboost")
  adaboost.Grid <-  expand.grid(nIter = seq(100, 150, 25), method = "Adaboost.M1")
  
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

###################### MULTICLASS CLASSIFICATION #############################
target.catMult.Hyper <- function(fmla, dataTrain, fitcontrol, parallel = TRUE, slow = FALSE) {
  # ADABAG: BAGGED ADABOOST: https://cran.r-project.org/web/packages/adabag/
  # ADABOOST.M1: ADABOOST.M1: https://cran.r-project.org/web/packages/adabag/
  # bagFDAGCV: BAGGED FDA USING gCV PRUNING: https://cran.r-project.org/web/packages/earth/
  # LogitBoost: BOOSTED LOGISTIC REGRESSION: https://cran.r-project.org/web/packages/caTools/
  # J48: C4.5-LIKE TREES: https://cran.r-project.org/web/packages/RWeka/
  # C5.0: C5.0: https://cran.r-project.org/web/packages/C50/
  
  fast.models <- c("AdaBag", "AdaBoost.M1", "bagFDAGCV", "LogitBoost", "J48", "C5.0")
  AdaBag.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10))
  AdaBoost.M1.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10), coeflearn = "Zhu")
  bagFDAGCV.Grid <-  expand.grid(degree = c(1, 2))
  LogitBoost.Grid <-  expand.grid(nIter = seq(100,300,100))
  J48.Grid <-  expand.grid(C = seq(0.01,0.05,0.01), M = 5)
  C5.0.Grid <-  expand.grid(trials = seq(20,100,20), model = c('rules', 'tree'), winnow = c(TRUE,FALSE))
  
  
  # bagFDA: BAGGED FLEXIBLE DISCRIMINANT ANALYSIS: https://cran.r-project.org/web/packages/earth/ : https://cran.r-project.org/web/packages/mda/
  
  slow.models <- c("bagFDA")
  bagFDA.Grid <-  expand.grid(degree = c(1, 2), nprune = c (1,2))
  
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
