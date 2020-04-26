# Load external libreries
library(lmvar)
library(glmnet)
library(pls)

# Definition of parameters
portrain <- 75
maxlambda = 100
grid <- 10^seq(10, -2, length=maxlambda)
depen <- 'price'
fmla <- as.formula(paste(depen, '~.'))

x <- model.matrix(fmla, clean_data)[,-1]
y <- eval(parse(text = paste('clean_data', '$', depen, sep='')))

# Split data intro training and testing subsets
smp_size <- floor(portrain / 100 * nrow(clean_data))
train_ind <- sample(seq_len(nrow(clean_data)), size = smp_size)
train <- clean_data[train_ind,]
test <- clean_data[-train_ind,]

# MULTIPLE LINEAR REGRESSION
lm.multiple.fit <- lm(fmla, data = clean_data)
summary(lm.multiple.fit)
confint(lm.multiple.fit) # Confidence interval for the coefficient estimates
#contrasts(data$neighbourhood_cleansed) # Coding that R uses for the dummy variables
cor(clean_data) # Correlations among the predictors in a data set
coefs <- coef(lm.multiple.fit)
summary(lm.multiple.fit)$coef
#a <- data.frame(beds=(seq(1,3273)))
#predict(lm.multiple.fit, a, interval ="prediction")
#cv.lm(clean_data, lm.multiple.fit, K = 10) # ERROR



# RIDGE REGRESSION
ridge.mod <- glmnet(x[train_ind,], y[train_ind], alpha=0, lambda=grid, thresh=1e-12)
cv.out <- cv.glmnet(x[train_ind,], y[train_ind], alpha=0)
bestlam <- cv.out$lambda.min
ridge.pred <- predict(ridge.mod, s=bestlam, newx=x[-train_ind,])
MSE.ridge <- max((ridge.pred-y[-train_ind])^2)
out <- glmnet(x, y, alpha=0)
ridge.coef <- predict(out, type="coefficients", s=bestlam)[1:19,]


# LASSO REGRESSION
lasso.mod <- glmnet(x[train_ind,], y[train_ind], alpha=1, lambda=grid)
cv.out <- cv.glmnet(x[train_ind,], y[train_ind], alpha=1)
bestlam <- cv.out$lambda.min
lasso.pred <- predict(lasso.mod, s=bestlam, newx=x[-train_ind,])
MSE.lasso <- max((lasso.pred-y[-train_ind])^2)
out <- glmnet(x, y, alpha=1, lambda=grid)
lasso.coef <- predict(out, type="coefficients", s=bestlam)[1:19,]


# PRINCIPAL COMPONENTS REGRESSION
pcr.fit <- pcr(fmla, data=clean_data, subset=train_ind, scale=TRUE, validation="CV")
validationplot(pcr.fit, val.type="MSEP")
pcr.pred <- predict(pcr.fit, x[-train_ind,], ncomp=16)
MSE.pcr <- max((pcr.pred-y[-train_ind])^2)

# PARTIAL LEAST SQUARES
pls.fit <- plsr(fmla, data=clean_data, subset=train_ind, scale=TRUE, validation="CV")
validationplot(pls.fit, val.type="MSEP")
pls.pred <- predict(pls.fit, x[-train_ind,], ncomp=16)
MSE.partial <- max((pls.pred-y[-train_ind])^2)






















pred.logi <- function(fit, ctest) {
  #fit : ajuste
  #ctest : conjunto test
  
  # Rates and ROC curve
  glm.probs = predict(fit, ctest, type='response')
  thresh <- seq(0, 1, by = 0.01)
  Indic.list <- lapply(thresh, function(thrsh){
    
    glm.preds <- rep('menos', length(glm.probs))
    glm.preds[glm.probs >= thrsh] <- 'mas'
    glm.preds <- factor(glm.preds, levels = c('menos', 'mas'))
    testTable <- table(ctest$aux, glm.preds)
    testSensitivity <- testTable[2, 2] / rowSums(testTable)[2] * 100
    testSpecificity <- (1 - testTable[1, 2] / rowSums(testTable)[1]) * 100
    testErrorRate <- (testTable[1, 2] + testTable[2, 1]) / sum(testTable) * 100
    output <- c(testErrorRate, testSensitivity, testSpecificity)
    names(output) <- c('ErrorRate', 'Sensitivity', 'Specificity')
    return(output)
  })
  Indic <- Reduce(rbind, Indic.list)
  rownames(Indic) <- NULL
  Indic <- as.data.table(Indic)
  Indic[, thresh := thresh]
  aux11 <- Indic$Sensitivity - Indic$Specificity
  logic1 <- abs(aux11) == min(abs(aux11))
  cota2 <- Indic$thresh[logic1]
  
  glm.pred = ifelse(glm.probs > cota2, "mas", "menos")
  valor1 <- table(glm.pred, ctest$aux)
  valor2 <- mean(glm.pred == ctest$aux) # Test Accuracy
  Sensitivity <- valor1[2, 2] / rowSums(valor1)[2] * 100
  Specificity <- (1 - valor1[1, 2] / rowSums(valor1)[1]) * 100
  ErrorRate <- (valor1[1, 2] + valor1[2, 1]) / sum(valor1) * 100
  return(list(valor1, valor2, Sensitivity, Specificity, ErrorRate))
}