
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)




# BRNN: BAYESIAN REGULARIZED NEURAL NETWORKS
# https://cran.r-project.org/web/packages/brnn/

# NEURALNET: NEURAL NETWORK
# https://cran.r-project.org/web/packages/neuralnet/

frbsGrid <-  expand.grid(layer1 = 10, layer2=10, layer3=10)

frbsFit1 <- train(fmla, data = dataTrain, 
                  method = "neuralnet", 
                  trControl = fitControl,
                  tuneGrid = frbsGrid)
frbsFit1
plot(frbsFit1) 


# QRNN: QUANTILE REGRESSION NEURAL NETWORK
# https://cran.r-project.org/web/packages/qrnn/