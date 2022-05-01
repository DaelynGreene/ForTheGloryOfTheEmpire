#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(fpp3)
library(tsibble)
library(seasonal)
library(forecast)
library(caret)
library(lubridate)
library(fpp3)
library(parallel)
library(doParallel)
library(dbplyr)


CREDITDATA <- read.csv("credit.csv")

CREDITDATA$Month <- 492:1
CREDITDATA$Month <- yearmonth(CREDITDATA$Month)
rename(CREDITDATA, credit_in_millions = ï..credit_in_millions)

CREDIT_TS <- tsibble(CREDITDATA, index = Month)
CREDIT_TS %>% 
  stretch_tsibble()

autoplot(CREDIT_TS)
gg_season(CREDIT_TS)

#lambda <- CREDIT_TS %>%
  #features(Month, features = guerrero) %>%
  #pull(lambda_guerrero)

#CREDIT_TS %>%
  #autoplot(box_cox(Month, lambda))

CREDIT_TS %>% 
  filter(year(Month) >= "1970 Feb") -> CREDIT_TS

train <- CREDIT_TS %>% 
  filter(Month <= yearmonth("2004 Jan"))

holdout <- CREDIT_TS %>% 
  filter(Month > yearmonth("2004 Jan"))

Model <- train %>% 
  stretch_tsibble(.init = 48, .step = 24) %>% 
  model(
    Naive = NAIVE(credit_in_millions),
    Drift = RW(credit_in_millions ~ drift()),
    arima012011 = ARIMA(credit_in_millions ~ pdq(0,1,2) + PDQ(0,1,1)),
    arima210011 = ARIMA(credit_in_millions ~ pdq(2,1,0) + PDQ(0,1,1)),
    stepwise = ARIMA(credit_in_millions),
    ETS = ETS(credit_in_millions),
    # search = ARIMA(credit_in_millions, stepwise=FALSE)
    # Neural = NNETAR(credit_in_millions)
  )

Model %>% 
  forecast(h = 6) %>% 
  accuracy(train) %>% 
  arrange(RMSE)

