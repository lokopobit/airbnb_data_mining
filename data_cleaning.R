#=========================================================================#
###########################################################################
#----------------------------DATA CLEANING--------------------------------#
###########################################################################
#=========================================================================#

# Load external libreries
library(editrules)

cleaning <- function(){
# Read csv data
# But first set de wording directory to source file location
print('Loadind data')
data <- read.csv('madrid-total-listings.csv')

# Ad hoc Feature selection. Other methods should be considered
# 75 features are removed
newdata <- subset(data, select = -c(name,id, listing_url, scrape_id, last_scraped,
                                    experiences_offered,space,description,summary,
                                    neighborhood_overview,notes,transit,access,interaction,
                                    house_rules, thumbnail_url,medium_url,picture_url,xl_picture_url,
                                    host_url,host_name,host_about,host_acceptance_rate,host_thumbnail_url,
                                    host_picture_url,host_verifications,state,country_code,smart_location,
                                    country,latitude,longitude,bed_type,square_feet,weekly_price,
                                    monthly_price,security_deposit,cleaning_fee,calendar_updated,
                                    has_availability,availability_30,availability_60,availability_90,
                                    calendar_last_scraped,requires_license,license,jurisdiction_names,
                                    calculated_host_listings_count,host_listings_count,
                                    host_total_listings_count,street,market,is_location_exact,
                                    neighbourhood,host_location,host_response_rate,zipcode,
                                    amenities,host_neighbourhood,first_review,last_review,
                                    host_id,host_since,neighbourhood_cleansed,neighbourhood_group_cleansed,
                                    city,host_response_time,property_type, cancellation_policy,
                                    host_is_superhost, host_has_profile_pic, instant_bookable,
                                    require_guest_profile_picture, require_guest_phone_verification,
                                    host_identity_verified))

# We filter by Private  and remove that column
# 8128 records are deleted
newdata <- newdata[as.character(newdata$room_type) == 'Private room',]
newdata$room_type = NULL

# Remove records containing not available (NA) cells 
if (sum(is.na(newdata[,])) == 0) {
  print('There are NOT NA')
} else {
  print(paste('Removed:',sum(is.na(newdata[,])),'NA', sep=' '))
  aux1 <- nrow(newdata)
  newdata <- na.omit(newdata)
  aux2 <- nrow(newdata)
  print(paste('Removed:',aux1-aux2,'records', sep=' '))
}

# Check if there are INF or NAN values
is.special <- function(x){
  if (is.numeric(x)) !is.finite(x) else is.na(x)
}
if(sum(sapply(newdata, is.special)) == 0) print('There are NO INF or NAN values')

# Remove dollar sign of price features and convert them to numeric data type
newdata$price = as.numeric(gsub("\\$", "", newdata$price))
newdata$extra_people = as.numeric(gsub("\\$", "", newdata$extra_people))

# Outliers treatment. The criterion considered are those records beyond the extremes
# of the whiskers of the box plot (for unimodal and symmetrical data)
boxplot.stats(newdata$accommodates,coef = 4)$out
aux <- newdata$accommodates<=8
newdata <- newdata[aux,]

boxplot.stats(newdata$bedrooms,coef = 4)$out
aux <- newdata$bedrooms<=5
newdata <- newdata[aux,]

boxplot.stats(newdata$beds,coef = 4)$out
aux <- newdata$beds<=8
newdata <- newdata[aux,]

boxplot.stats(newdata$guests_included,coef = 4)$out
aux <- newdata$guests_included<=6
newdata <- newdata[aux,]

boxplot.stats(newdata$extra_people,coef = 4)$out
aux <- newdata$extra_people<=99
newdata <- newdata[aux,]

boxplot.stats(newdata$minimum_nights,coef = 4)$out
aux <- newdata$minimum_nights<=30
newdata <- newdata[aux,]

boxplot.stats(newdata$maximum_nights,coef = 0.08)$out
aux <- newdata$maximum_nights<=1500 #si ponemos 1000 nos quitamos demasiadas obs.
newdata <- newdata[aux,]

boxplot.stats(newdata$availability_365,coef = 0.3)$out
aux <- newdata$availability_365>=2
newdata <- newdata[aux,]

boxplot.stats(newdata$number_of_reviews,coef = 4)$out
aux <- newdata$number_of_reviews<=250
newdata <- newdata[aux,]

# Check that no rule is broken
rules <- editfile('rules.txt')
ve <- violatedEdits(rules, newdata)
summary(ve)

# Save newdata
print('Saving clean data')
write.csv(newdata, file = "clean_data.csv", row.names = FALSE)

return(newdata)
}
