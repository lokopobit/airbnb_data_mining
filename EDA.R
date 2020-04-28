
# https://github.com/daya6489/SmartEDA#summary-of-categorical-variables
# https://cran.r-project.org/web/packages/SmartEDA/SmartEDA.pdf
# https://arxiv.org/pdf/1903.04754.pdf

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


# QUANTILE-QUANTILE PLOT FOR NUMERIC VARIABLES
ExpOutQQ(CData,nlim=10,fname=NULL,Page=c(2,2),sample=4)

# PARALLEL CO-ORDINATE PLOTS
## Defualt ExpParcoord funciton
ExpParcoord(CData,Group=NULL,Stsize=NULL,Nvar=c("Price","Income","Advertising","Population","Age","Education"))
## With Stratified rows and selected columns only
ExpParcoord(CData,Group="ShelveLoc",Stsize=c(10,15,20),Nvar=c("Price","Income"),Cvar=c("Urban","US"))
## Without stratification
ExpParcoord(CData,Group="ShelveLoc",Nvar=c("Price","Income"),Cvar=c("Urban","US"),scale=NULL)
## Scale change  
ExpParcoord(CData,Group="US",Nvar=c("Price","Income"),Cvar=c("ShelveLoc"),scale="std")
## Selected numeric variables
ExpParcoord(CData,Group="ShelveLoc",Stsize=c(10,15,20),Nvar=c("Price","Income","Advertising","Population","Age","Education"))
## Selected categorical variables
ExpParcoord(CData,Group="US",Stsize=c(15,50),Cvar=c("ShelveLoc","Urban"))

# UNIVARIATE OUTLIER ANALYSIS
##Identifying outliers mehtod - Boxplot
ExpOutliers(Carseats, varlist = c("Sales","CompPrice","Income"), method = "boxplot",  capping = c(0.1, 0.9))
##Identifying outliers mehtod - 3 Standard Deviation
ExpOutliers(Carseats, varlist = c("Sales","CompPrice","Income"), method = "3xStDev",  capping = c(0.1, 0.9))
##Identifying outliers mehtod - 2 Standard Deviation
ExpOutliers(Carseats, varlist = c("Sales","CompPrice","Income"), method = "2xStDev",  capping = c(0.1, 0.9))
##Create outlier flag (1,0) if there are any outliers 
ExpOutliers(Carseats, varlist = c("Sales","CompPrice","Income"), method = "3xStDev",  capping = c(0.1, 0.9), outflag = TRUE)
##Impute outlier value by mean or median valie
ExpOutliers(Carseats, varlist = c("Sales","CompPrice","Income"), method = "3xStDev", treatment = "mean", capping = c(0.1, 0.9), outflag = TRUE)


# CUSTOM TABLES, SUMMARY STATISTICS
ExpCustomStat(Carseats,Cvar=c("US","Urban","ShelveLoc"),gpby=FALSE)
ExpCustomStat(Carseats,Cvar=c("US","Urban"),gpby=TRUE,filt=NULL)
ExpCustomStat(Carseats,Cvar=c("US","Urban","ShelveLoc"),gpby=TRUE,filt=NULL)
ExpCustomStat(Carseats,Cvar=c("US","Urban"),gpby=TRUE,filt="Population>150")
ExpCustomStat(Carseats,Cvar=c("US","ShelveLoc"),gpby=TRUE,filt="Urban=='Yes' & Population>150")
ExpCustomStat(Carseats,Nvar=c("Population","Sales","CompPrice","Income"),stat = c('Count','mean','sum','var','min','max'))
ExpCustomStat(Carseats,Nvar=c("Population","Sales","CompPrice","Income"),stat = c('min','p0.25','median','p0.75','max'))
ExpCustomStat(Carseats,Nvar=c("Population","Sales","CompPrice","Income"),stat = c('Count','mean','sum','var'),filt="Urban=='Yes'")
ExpCustomStat(Carseats,Nvar=c("Population","Sales","CompPrice","Income"),stat = c('Count','mean','sum'),filt="Urban=='Yes' & Population>150")
ExpCustomStat(data_sam,Nvar=c("Population","Sales","CompPrice","Income"),stat = c('Count','mean','sum','min'),filt="All %ni% c(999,-9)")
ExpCustomStat(Carseats,Nvar=c("Population","Sales","CompPrice","Education","Income"),stat = c('Count','mean','sum','var','sd','IQR','median'),filt=c("ShelveLoc=='Good'^Urban=='Yes'^Price>=150^ ^US=='Yes'"))
ExpCustomStat(Carseats,Cvar = c("Urban","ShelveLoc"), Nvar=c("Population","Sales"), stat = c('Count','Prop','mean','min','P0.25','median','p0.75','max'),gpby=FALSE)
ExpCustomStat(Carseats,Cvar = c("Urban","US","ShelveLoc"), Nvar=c("CompPrice","Income"), stat = c('Count','Prop','mean','sum','PS','min','max','IQR','sd'), gpby = TRUE)
ExpCustomStat(Carseats,Cvar = c("Urban","US","ShelveLoc"), Nvar=c("CompPrice","Income"), stat = c('Count','Prop','mean','sum','PS','P0.25','median','p0.75'), gpby = TRUE,filt="Urban=='Yes'")
ExpCustomStat(data_sam,Cvar = c("Urban","US","ShelveLoc"), Nvar=c("Sales","CompPrice","Income"), stat = c('Count','Prop','mean','sum','PS'), gpby = TRUE,filt="All %ni% c(888,999)")
ExpCustomStat(Carseats,Cvar = c("Urban","US"), Nvar=c("Population","Sales","CompPrice"), stat = c('Count','Prop','mean','sum','var','min','max'), filt=c("ShelveLoc=='Good'^Urban=='Yes'^Price>=150"))

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
