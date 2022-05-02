
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
  filter(Month <= yearmonth("2005 Jan"))

holdout <- CREDIT_TS %>% 
  filter(Month > yearmonth("2005 Jan"))

cluster <- makeCluster(detectCores() - 1) #Line 1 for parallelization
registerDoParallel(cluster)

Model <- train %>% 
  stretch_tsibble(.init = 48, .step = 24) %>% 
  model(
    Naive = NAIVE(credit_in_millions),
    S_NAIVE = SNAIVE(credit_in_millions),
    Drift = RW(credit_in_millions ~ drift()),
    Linear = TSLM(credit_in_millions ~ trend()),
    arima000011 = ARIMA(credit_in_millions ~ pdq(0,0,0) + PDQ(0,1,1)),
    #arima210011 = ARIMA(credit_in_millions ~ pdq(0,0,0) + PDQ(0,1,3)),
    #arima210011 = ARIMA(credit_in_millions ~ pdq(0,0,0) + PDQ(0,1,0)),
    stepwise = ARIMA(credit_in_millions),
    #ETS = ETS(credit_in_millions),
    search = ARIMA(credit_in_millions, stepwise=FALSE),
    #arima210 = ARIMA(credit_in_millions ~ pdq(2,1,1)),
    arima013 = ARIMA(credit_in_millions ~ pdq(0,1,3)),
    #arima301012 = ARIMA(credit_in_millions ~ pdq(3,0,1) + PDQ(0,1,2)),
    #arima301111 = ARIMA(credit_in_millions ~ pdq(3,0,1) + PDQ(1,1,1)),
    #arima301110 = ARIMA(credit_in_millions ~ pdq(3,0,1) + PDQ(1,1,0)),
    #auto = ARIMA(credit_in_millions, stepwise = FALSE, approx = FALSE)
    #Neural = NNETAR(credit_in_millions)
    #`K = 1` = ARIMA((credit_in_millions) ~ fourier(K=1) + PDQ(0,0,0)),
    #`K = 2` = ARIMA((credit_in_millions) ~ fourier(K=2) + PDQ(0,0,0)),
    #`K = 3` = ARIMA((credit_in_millions) ~ fourier(K=3) + PDQ(0,0,0)),
    #`K = 4` = ARIMA((credit_in_millions) ~ fourier(K=4) + PDQ(0,0,0)),
    #`K = 5` = ARIMA((credit_in_millions) ~ fourier(K=5) + PDQ(0,0,0)),
    #`K = 6` = ARIMA((credit_in_millions) ~ fourier(K=6) + PDQ(0,0,0)),
    #additive = ETS(credit_in_millions ~ error("A") + trend("A") + season("A")),
    #multiplicative = ETS(credit_in_millions ~ error("M") + trend("A") + season("M"))
  )

stopCluster(cluster) #Line 3 for parallelization
registerDoSEQ()

Model %>% #accuracy against the training data
  forecast(h = 72) %>% 
  accuracy(train) %>% 
  arrange(RMSE)

Model %>% #accuracy against the holdout data
  forecast(h = 72) %>% 
  accuracy(holdout) %>% 
  arrange(RMSE)

credit_best <- train %>%
  model(
    #stepwise = ARIMA(credit_in_millions)
    #ETS = ETS(credit_in_millions)
    search = ARIMA(credit_in_millions, stepwise=FALSE),
    #neural = NNETAR(credit_in_millions)
    #Linear = TSLM(credit_in_millions ~ trend())
    #auto = ARIMA(credit_in_millions, stepwise = FALSE, approx = FALSE)
    #`K = 4` = ARIMA((credit_in_millions) ~ fourier(K=4) + PDQ(0,0,0))
    #`K = 5` = ARIMA((credit_in_millions) ~ fourier(K=5) + PDQ(0,0,0))
    arima000011 = ARIMA(credit_in_millions ~ pdq(0,0,0) + PDQ(0,1,1))
  )

credit_best %>% forecast(holdout) %>%
  autoplot(train) +
  labs(y = "Credits", title = "Credit in Millions Prediction")

credit_best %>%
  select("search") %>%
  report()

credit_best %>%
  select("arima000011") %>%
  report()

credit_best %>% select("search") %>% gg_tsresiduals()
credit_best %>% select("arima000011") %>% gg_tsresiduals()

credit_best %>%
  forecast(h = 72) %>%
  autoplot(holdout) +
  labs(y = "Credits", title = "Imperial Revenue")

pred1 <- credit_best %>% select("search") %>%
  forecast(h = 12)

pred2 <- credit_best %>% select("arima000011") %>%
  forecast(h = 12)


rmse <- function(y_actual, y_pred) {
  sqrt(mean((y_actual - y_pred)^2))
}

mape <- function(y_actual, y_pred) {
  mean(abs(y_actual - y_pred) / y_actual)
}

rmse(holdout$credit_in_millions, pred1$.mean)
mape(holdout$credit_in_millions, pred1$.mean)

rmse(holdout$credit_in_millions, pred2$.mean)
mape(holdout$credit_in_millions, pred2$.mean)

credit_best2 <- CREDIT_TS %>%
  model(
    #stepwise = ARIMA(credit_in_millions)
    #ETS = ETS(credit_in_millions)
    #neural = NNETAR(credit_in_millions)
    #Linear = TSLM(credit_in_millions ~ trend()),
    #search = ARIMA(credit_in_millions, stepwise=FALSE),
    arima000011 = ARIMA(credit_in_millions ~ pdq(0,0,0) + PDQ(0,1,1))
  )

credit_best2 %>%
  forecast(h = 12) %>%
  autoplot(CREDIT_TS) +
  labs(y = "Credits", title = "Imperial Revenue")


predictions <- credit_best2 %>%
  forecast(h = 12)

predictions <- subset(predictions, Month > yearmonth("2011 Jan"))
predictions <- predictions[,c(-1,-3)]
names(predictions)[2] <- "Credit in Millions"

predictions

write.csv(predictions,file="predictions.csv",row.names = FALSE)
