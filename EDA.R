
library('SmartEDA')

data <- read.csv('madrid-total-listings.csv')

ExpCatStat(data)

ExpCatViz(data, fname = 'amostu') # Create bar plots for each categorical variable

ExpCTable(data)
