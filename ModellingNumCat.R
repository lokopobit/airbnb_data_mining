# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)


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
