
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
library(ggeasy)
library(ggthemes)


CREDITDATA <- read.csv("credit.csv")

CREDITDATA$Month <- 492:1
CREDITDATA$Month <- yearmonth(CREDITDATA$Month)
names(CREDITDATA)[1] <- "credit_in_millions"

CREDIT_TS <- tsibble(CREDITDATA, index = Month)

autoplot(CREDIT_TS)+theme_wsj()+ggtitle("Monthly Imperial Credits of the Empire (in Millions)")+easy_center_title()+ylab("Imperial Credit (millions)") 
gg_season(CREDIT_TS)

train <- CREDIT_TS %>% 
  filter(Month <= yearmonth("2005 Jan"))

holdout <- CREDIT_TS %>% 
  filter(Month > yearmonth("2005 Jan"))

cluster <- makeCluster(detectCores() - 1) #Line 1 for parallelization
registerDoParallel(cluster)

Model <- CREDIT_TS %>% 
  stretch_tsibble(.init = 48, .step = 24) %>% 
  model(
    arima100021 = ARIMA(credit_in_millions ~ pdq(1,0,0) + PDQ(0,2,1))
  )

stopCluster(cluster) #Line 3 for parallelization
registerDoSEQ()

Model %>% #accuracy against the holdout data
  forecast(h = 72) %>% 
  accuracy(holdout) %>% 
  arrange(RMSE)

credit_best <- train %>%
  model(
    arima100021 = ARIMA(credit_in_millions ~ pdq(1,0,0) + PDQ(0,2,1))
  )

credit_best %>% forecast(holdout) %>%
  autoplot(train) +
  labs(y = "Credits", title = "Credit in Millions Prediction")

credit_best %>%
  select("arima100021") %>%
  report()

credit_best %>% select("arima100021") %>% gg_tsresiduals()

credit_best %>%
  forecast(h = 72) %>%
  autoplot(holdout) +
  theme_wsj()+
  ggtitle("Imperial Revenue") +
  easy_center_title()

pred2 <- credit_best %>% select("arima100021") %>%
  forecast(h = 12)

rmse <- function(y_actual, y_pred) {
  sqrt(mean((y_actual - y_pred)^2))
}

mape <- function(y_actual, y_pred) {
  mean(abs(y_actual - y_pred) / y_actual)
}

rmse(holdout$credit_in_millions, pred2$.mean)
mape(holdout$credit_in_millions, pred2$.mean)

credit_best2 <- CREDIT_TS %>%
  model(
    arima100021 = ARIMA(credit_in_millions ~ pdq(1,0,0) + PDQ(0,2,1))
  )

credit_best2 %>%
  forecast(h = 12) %>%
  autoplot(CREDIT_TS) +theme_wsj()+ggtitle("Monthly Imperial Credits of the Empire (in Millions)")+easy_center_title()+ylab("Imperial Credit (millions)") 


predictions <- credit_best2 %>%
  forecast(h = 12)

predictions <- subset(predictions, Month > yearmonth("2011 Jan"))
predictions <- predictions[,c(-1,-3)]
names(predictions)[2] <- "Credit in Millions"

predictions

write.csv(predictions,file="predictions.csv",row.names = FALSE)
