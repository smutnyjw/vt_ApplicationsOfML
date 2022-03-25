Project 1:  NOAA Next Day Weather Prediction Models
Completed:  03/22/2022
Authors:    John W. Smutny, Ben C. Johnson, Anagha Mudki, James A. Ensminger

Goal:
Predict 'if' there will be any precipitation event tomorrow and how much.

Models Used:
Classification Decision Tree: Predict if there will be precipitation tomorrow
Linear Regression: Predict how much precipitation there will be tomorrow.

Abstract:
Abstract — Predicting the weather is an act that influences
society everyday of every year. A prediction influences how an
individual may travel to work, whether a business can operate, or
any other countless effects. This report aims to use linear
regression and classification machine learning models to predict
how much precipitation will occur around the Charleston
International Airport in Charleston, South Caroline of the
United States of America. These models will be trained using
over 70 years’ worth of atmospheric data provided by the
National Oceanic and Atmospheric Administration (NOAA) of
the United States of America’s government.
Both models are trained by analyzing a full day’s worth of
measurements. After each model is trained, they attempt to
predict two items; 1) whether the next day will have a
precipitation event (IE: there is non-zero rain, snow, hail, etc)
and 2) how much the next day’s precipitation will be. These two
predictions will be made twice: first using only the current day’s
weather, then second using the current and previous day’s
weather.
After both sets of models were created; the influence of more
data affected the classification model and linear regression
models differently. As the amount of data used in the training set
of the classification models increased, so did the model’s testing
accuracy and model’s prediction confidence scores. Meanwhile,
as more data was considered for the linear regression model, the
model’s ability to predict how much precipitation would occur
did not noticeably improve.


References:
Menne, M.J., I.Durre, B. Korzeniewski, S. McNeal, K. Thomas, X. Yin, S. Anthony,
  R. Ray, R.S. Vose, B.E.Gleason, and T.G. Houston,  2012: Global Historical
  Climatology Network - Daily (GHCN-Daily), Version 3.26 ,  USW00013880 e.g.
  Version 3.12].  NOAA National Climatic Data Center. http://doi.org/10
  .7289/V5D21VHZ March 22nd 2022.
National Weather Service, Charleston, SC. Weather Forecast Office.  (2022, 3 20)
 Local Climate Data and Plots [Online].  Available: https://www.weather
 .gov/chs/climate
