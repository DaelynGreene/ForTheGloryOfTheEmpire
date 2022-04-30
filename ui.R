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


CREDITDATA <- read.csv('credit.csv')

CREDITDATA$Month <- 492:1
CREDITDATA$Month <- yearmonth(CREDITDATA$Month)

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
  filter(year(Month) >= '1970 Feb') -> CREDIT_TS

CREDIT_TS %>% 
  filter(year(Month) <= '2004 Jan') -> TRAINCREDIT
