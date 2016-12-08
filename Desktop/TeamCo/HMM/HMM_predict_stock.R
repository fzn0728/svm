install.packages("RHmm")
library(xts)
library(zoo)
library(TTR)
library(quantmod)
getSymbols("^TWII",src="yahoo",from="1900-01-01",to="2015-12-31")
chartSeries(TWII,theme="black")
TWII_Subset <- window(TWII, start=as.Date("2013-01-01"),end=as.Date("2015-12-31"))
TWII_Train <- cbind(TWII_Subset$TWII.Close-TWII_Subset$TWII.Open)

########################
# Baum-Welch Algorithm
########################
library(RHmm)
