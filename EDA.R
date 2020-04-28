
# LOAD EXTERNAL LIBRERIES
library('SmartEDA')

# LOAD DATA. Set working directory to current source file location.
data <- read.csv('madrid-total-listings.csv')

# Remove dollar sign of price features and convert them to numeric data type
data$price = as.numeric(gsub("\\$", "", data$price))
data$extra_people = as.numeric(gsub("\\$", "", data$extra_people))

# Choose target variable
num_target = 'price'
cat_target = 'host_response_time'


#######################################################
################## BASIC EDA ##########################
#######################################################

# OVERVIEW OF THE DATA
# Understanding the dimensions of the dataset, variable names, 
# overall missing summary and data types of each variables
ExpData(data = data, type=1) # Overview of the data
ExpData(data = data, type=2) # Structure of the data


# SUMMARY OF NUMERICAL VARIABLES
ExpNumStat(data,by="A",gp=NULL,Qnt=seq(0,1,0.1),MesofShape=2,Outlier=TRUE,round=2)

# GRAPHICAL REPRESENTATION OF ALL NUMERIC FEATURES
ExpNumViz(data,target = num_target,type=1,nlim=25,fname = 'num_target_scatter_plot',Page = c(2,2))
ExpNumViz(data,target = car_target,type=1,nlim=25,fname = 'cat_target_box_plot',Page = c(2,2)) #try changing type
ExpNumViz(data,target = NULL,type=1,nlim=25,fname = 'density_plot',Page = c(2,2))
ExpNumViz(data,target = NULL,type=1,nlim=25,fname = 'all_scatter_plot',Page = c(2,2), scatter = TRUE)

# SUMMARY OF CATEGORICAL VARIABLES
ExpCTable(data,Target=NULL,margin=1,clim=10,nlim=5,round=2,bin=NULL,per=T) # Frecuency tables for categorical variables
ExpCatStat(data,Target=cat_target,result = "Stat",clim=10,nlim=5,Pclass="Yes") # Summary statistics of categorical variables
ExpCatStat(data,Target=cat_target,result = "IV",clim=10,nlim=5,Pclass="Yes") # Inforamtion value and Odds value

# GRAPHICAL REPRESENTATION OF ALL CATEGORICAL VARIABLES
ExpCatViz(data,target=cat_target,fname='column_charts',clim=10,col=NULL,margin=2,Page = c(2,1),sample=2) # Column chart
ExpCatViz(data,target=cat_target,fname='stacked_bar_graph',clim=10,col=NULL,margin=2,Page = c(2,1),sample=2) # Stacked bar graph
ExpCatStat(data,Target=cat_target,result="Stat",Pclass="Yes",plot=TRUE,top=20,Round=2) # Variable importance graph using information values


#######################################################
################## ADVANCED EDA #######################
#######################################################


# Still don't now how they work
ExpCustomStat(data=data)


# Still don't know what they do
ExpInfoValue(data, Target=target) # Information value


# DESCRIPTIVE STATISTICS

ExpKurtosis(data, type='moment') # Measures of Shape - Kustosis
ExpKurtosis(data, type='excess') # Measures of Shape - Kustosis



# DATA VISUALIZATION

ExpOutQQ(data, fname='quantile_plots') # Quantile plots
# ExpParcoord(data) DON' USE UNDER ANY CIRCUNSTANCE --> REPORT

# OUTLIERS DETECTION
ExpOutliers(data, varlist = c('price'), method='boxplot')


# CUSTOM TABLE

# CREATE HTML EDA REPORT
ExpReport(data,Target=NULL,label=NULL,op_file="report.html",op_dir=getwd(),sc=2,sn=2,Rc="Yes")
