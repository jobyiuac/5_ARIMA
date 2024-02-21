import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

# Read data from csv
dataset = pd.read_csv('TRVDATAFILTERED-300POINTS.csv')#('sinewave.csv')#
print (dataset)


# ARIMA model
AR_order = 3
integral = 2
MA_order = 3

for i in range(1,5):
    integral=i
    order = (AR_order, integral, MA_order)  # ARIMA order: (p, d, q) 


    ARIMAmodel = sm.tsa.arima.ARIMA(dataset,order = (AR_order, integral, MA_order))
    ARIMAmodel_fit = ARIMAmodel.fit()
    ypredicted = ARIMAmodel_fit.predict(300,400)


    plt.plot(dataset, color='b', label='original data') # Plot the csv file
    plt.title('Dataset', color = 'r', fontsize = '25')
    #plt.show()
    
    plt.plot(ypredicted, color='r', label='forecasted data')
    plt.title(f'AR_order = 3,MA_order = 3, integral =' +str(i),fontsize = 20, color = 'green')

    plt.legend()
    plt.show()


















