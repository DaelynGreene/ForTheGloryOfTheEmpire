
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
names(CREDITDATA)[1] <- "credit_in_millions"

CREDIT_TS <- tsibble(CREDITDATA, index = Month)

autoplot(CREDIT_TS)
gg_season(CREDIT_TS)

# CREDIT_TS %>%
#   features(credit_in_millions, features = guerrero) %>%
#   pull(lambda_guerrero) -> lambda
# 
# TransCredit <- CREDIT_TS
# 
# box_cox(CREDIT_TS$credit_in_millions,lambda) -> TransCredit$credit_in_millions
# 
# autoplot(TransCredit)


train <- CREDIT_TS %>% 
  filter(Month <= yearmonth("2004 Jan"))

holdout <- CREDIT_TS %>% 
  filter(Month > yearmonth("2004 Jan"))

cluster <- makeCluster(detectCores() - 1) #Line 1 for parallelization
registerDoParallel(cluster)

Model <- train %>% 
  stretch_tsibble(.init = 48, .step = 24) %>% 
  model(
    Naive = NAIVE(credit_in_millions),
    Drift = RW(credit_in_millions ~ drift()),
    arima012011 = ARIMA(credit_in_millions ~ pdq(0,1,2) + PDQ(0,1,1)),
    arima210011 = ARIMA(credit_in_millions ~ pdq(2,1,0) + PDQ(0,1,1)),
    stepwise = ARIMA(credit_in_millions),
    ETS = ETS(credit_in_millions),
    search = ARIMA(credit_in_millions, stepwise=FALSE),
    Linear = TSLM(credit_in_millions ~ trend())
    #Neural = NNETAR(credit_in_millions)
  )

stopCluster(cluster) #Line 3 for parallelization
registerDoSEQ()

Model %>% 
  forecast(h = 6) %>% 
  accuracy(train) %>% 
  arrange(RMSE)

credit_best <- train %>%
  model(
    #stepwise = ARIMA(credit_in_millions)
    #ETS = ETS(credit_in_millions)
    Linear = TSLM(credit_in_millions ~ trend())
  )

credit_best %>% forecast(holdout) %>%
  autoplot(train) +
  labs(y = "% of GDP", title = "Credit in Millions Prediction")

report(credit_best)

credit_best %>% gg_tsresiduals()


credit_best %>%
  forecast(h = 84) %>%
  autoplot(holdout) +
  labs(y = "Credits", title = "Imperial Revenue")

pred <- credit_best %>%
  forecast(h = 84)

rmse <- function(y_actual, y_pred) {
  sqrt(mean((y_actual - y_pred)^2))
}

mape <- function(y_actual, y_pred) {
  mean(abs(y_actual - y_pred) / y_actual)
}

rmse(holdout$credit_in_millions, pred$.mean)
mape(holdout$credit_in_millions, pred$.mean)

credit_best2 <- CREDIT_TS %>%
  model(
    #stepwise = ARIMA(credit_in_millions)
    #ETS = ETS(credit_in_millions)
    Linear = TSLM(credit_in_millions ~ trend())
  )

predictions <- credit_best2 %>%
  forecast(h = 96)

predictions <- subset(predictions, Month > yearmonth("2011 Jan"))
predictions <- predictions[,c(-1,-3)]
names(predictions)[2] <- "Credit in Millions"

predictions

write.csv(predictions,file="predictions.csv",row.names = FALSE)
