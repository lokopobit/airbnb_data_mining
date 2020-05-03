
# http://topepo.github.io/caret/
# https://cran.r-project.org/web/packages/caret/caret.pdf 

# Load external libreries 
library(caret)


# FRBS: ADAPTIVE-NETWORK-BASED FUZZY INFERENCE SYSTEM
# https://cran.r-project.org/web/packages/frbs/

frbsGrid <-  expand.grid(num.labels = c(1, 2), max.iter = (1:2)*2)

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
# Include FSHGD: SIMPLIFIED TSK FUZZY RULES
# Include SUBSTRACTIVE CLUSTERING AND FUZZY C-MEANS RULES
# Include WANG AND MENDEL FUZZY RULES


frbsGrid <-  expand.grid(num.labels = c(2,3), type.mf = 'GAUSSIAN')

frbsFit1 <- train(fmla, data = dataTrain, 
                  method = "WM", 
                  trControl = fitControl,
                  tuneGrid = frbsGrid)
frbsFit1
plot(frbsFit1) 