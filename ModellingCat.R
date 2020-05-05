
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)
library(doParallel)

# Graphics sep up
trellis.par.set(caretTheme())

# Create binary variable
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

model.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10))
model.Fit <- train(fmla, data = a, 
                   method = "AdaBag", 
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
  # 

  fast.models <- c("")

  
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
  
  fast.models <- c("AdaBag", "AdaBoost.M1")
  AdaBag.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10))
  AdaBoost.M1.Grid <-  expand.grid(mfinal = c(3, 6), maxdepth = c(5, 10), coeflearn = "Zhu")
  
  # 
  
  slow.models <- c("")

  
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

model.Grid <-  expand.grid(nIter = seq(100, 150, 25), method = "Adaboost.M1")
model.Fit <- train(fmla, data = a, 
                   method = "adaboost", 
                   trControl = fitControl,
                   tuneGrid = model.Grid)

plot(model.Fit)
print(model.Fit$results)
