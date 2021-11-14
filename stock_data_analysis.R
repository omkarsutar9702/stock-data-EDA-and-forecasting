#import library
library(tsfknn)
library(tidyquant)
library(tidyverse)
library(ggplot2)
library(tidymodels)
library(plotly)
library(tseries)
library(timeSeries)
library(prophet)
library(dplyr)
library(forecast)
library(caret)
library(rugarch)
#import data
stock<-function(x){
  data<- tq_get(x, get = "stock.prices", from = "2017-01-01")
}

df<-data.frame(stock("AAPL"))
#####data EDA######

#line plot
fig<-plot_ly(df , type="scatter" , mode= "lines") %>% 
  add_trace(x = ~date , y=~close) %>% layout(
    showlegend = F , title = "time series of apple data",
    xaxis = list(rangeslider=list(visible = T)),
    plot_bgcolor='#e5ecf6'
  )
fig

#candlestick
candle<-tail(df , 30)
fig_1<-plot_ly(candle , x = ~date ,type="candlestick",
               open =~open , close = ~close,
               high = ~high , low = ~low) %>% 
  layout(title = "candlestick chart for apple" ,plot_bgcolor='#e5ecf6' )
fig_1

#bb chart
df %>%
  ggplot(aes(x = date, y = close, open = open,
             high = high, low = low, close = close)) +
  geom_candlestick() +
  geom_bbands(ma_fun = SMA, sd = 2, n = 20 ,linetype = 5) +
  labs(title = "AAPL Candlestick Chart", 
       subtitle = "BBands with SMA Applied", 
       y = "Closing Price", x = "") + 
  theme_tq()


#regression line
df %>% 
  ggplot(aes(x=date , y=close))+
  geom_line(color = palette_dark()[[1]])+
  scale_y_log10()+
  geom_smooth(method = "lm")+
  labs(title = "apple regression line" , 
       y = "price"  , x = "")+
  theme_grey()

#bar chart for volume
df %>% 
  ggplot(aes(x = date , y = volume))+
  geom_segment(aes(xend = date , yend = 0 , color = volume))+
  geom_smooth(method = "loess" ,se = F)+
  labs(title = "apple volume chart with loss line",
       y="volume" , x="")+
  scale_color_gradient(low = "red" , high="blue")+
  theme_grey()+
  theme(legend.position = "none")

#WEEKLY returns
apple_returns<-df %>% 
  tq_transmute(
    select = close,
    mutate_fun = periodReturn,
    period = "weekly",
    col_rename ="yearly_returns" 
  )
apple_returns %>% 
  ggplot(aes(x = year(date) , y = yearly_returns))+
  geom_bar(position = "dodge" , stat = "identity")+
  labs(title = "annual returns of apple data" , 
       y = "retruns" , x="")+
  scale_y_continuous(labels = scales::percent)+
  coord_flip()+
  theme_grey()+
  scale_fill_tq()

#returns in line 
apple_returns_1<-df %>% 
  tq_transmute(
    select = close,
    mutate_fun = periodReturn,
    period = "yearly",
    type = "log",
    col_rename ="yearly_return" 
  )
apple_returns_1 %>% 
  ggplot(aes(x = year(date) , y=yearly_return))+
  geom_hline(yintercept = 0  , color = palette_light()[[1]])+
  geom_point(size = 2 , color = palette_light()[[3]])+
  geom_line(size = 1 , color = palette_light()[[2]])+
  geom_smooth(method = "lm" , se = F)+
  labs(title = "apple trends in annual returns" , 
       x = "" , y = "annual returns")+
  theme_dark()

#daily log returns
apple_returns_2<-df %>% 
  tq_transmute(
    select = close,
    mutate_fun = periodReturn,
    period = "daily",
    type = "log",
    col_rename ="monthly_return" 
  )

apple_returns_2 %>%
  ggplot(aes(x = monthly_return , fill = "apple"))+
  geom_density(alpha=0.7, color = "blue")+
  labs(title = "daily log returns of apple" , 
       y = "density" , x = "monthly returns")+
  theme_minimal()+
  scale_fill_tq()

#moving average
appl_macd<-df %>% 
  tq_mutate(
    select = close,
    mutate_fun = MACD , 
    nFast = 15,
    nSlow = 25,
    nSig = 9,
    maType = SMA
  ) %>% 
  mutate(diff = macd - signal) %>% 
  select(-(open:volume))

appl_macd %>% 
  ggplot(aes(x = date))+
  geom_hline(yintercept = 0 , color = palette_light()[[1]])+
  geom_line(aes(y=macd ))+
  geom_line(aes(y = signal) , color  ="red" , linetype = 2)+
  geom_bar(aes(y = diff) ,stat = "identity" , color = palette_dark()[[1]])+
  labs(title = "MACD of appple" , y = "MACD" , x = "")+
  theme_tq()+
  scale_color_tq()

#####build the time series####
####arima
#ADF test
print(adf.test(df$close))
#series is not stationary

#plot acf and pacf
acf(df$close , main = "acf")
pacf(df$close , main = "pacf")

#auto arima
auto_arima<-auto.arima(df$close, lambda = "auto")
plot(auto_arima$residuals)

#ljung box test
test<-Box.test(auto_arima$residuals , type = "Ljung-Box")
test

#forecast next 30 dayss
forecast_price<-forecast(auto_arima , h = 30)
#plot the forecast
plot(forecast_price)
head(forecast_price$mean)
#lower bound
head(forecast_price$lower)
#upper bound
head(forecast_price$upper)

#create train and test data to train the model
close.price <- ts(df$close)
N = length(close.price)
n = 0.80*N
train<-close.price[1:n]
test<-close.price[(n+1):N]
train_model <- auto.arima(train, lambda = "auto")
predict_len <- length(test)
train_model_pred<-forecast(train_model , h = predict_len)

#plotting mean predicted values over real data set
meanvalues<-as.vector(train_model_pred$mean)
precios<-as.vector(test)

plot(meanvalues , type="l" , col = "blue")
lines(precios , type = "l")

######prophet model application 
dataset<-data.frame(ds = df$date , y  = as.numeric(df$close))

prophet_1<-prophet(dataset)
future<- make_future_dataframe(prophet_1 , periods = 30)
forecastprophet <-predict(prophet_1 , future)
plot(prophet_1 ,forecastprophet )

#create dataset for compairsion 
datapartition <- data.frame(forecastprophet$yhat , forecastprophet$ds)
trainlen<- length(close.price)
datapartition_new<-datapartition[c(1:trainlen) , ]

#cross validation 
accuracy(datapartition_new$forecastprophet.yhat , dataset$y)
prophet_plot_components(prophet_1 , forecastprophet)

###knn model
dataset_knn<-data.frame(ds = df$date , y  = as.numeric(df$close))
predknn <- knn_forecasting(dataset_knn$y , h = 30 , lags = 1:30 , k = 50 , msas = "MIMO")

#train set model accuray
ro<-rolling_origin(predknn)
print(ro$global_accu)
plot(predknn)
autoplot(predknn, highlight = "neighbors",faceting = TRUE)
