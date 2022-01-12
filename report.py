# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 02:02:34 2021

@author: jaisa
"""

# /// Importing modules ///
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from colour import Color
import glob
from statsmodels.stats.stattools import durbin_watson
import statistics
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import itertools
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn import preprocessing
import math

# /// Loding files from Directory ///
b1=pd.read_pickle(r"C:\Users\jaisa\OneDrive - Uppsala universitet\uni drive\thesis uni\Data B08 test 1sec 30days.pkl")
ctcurve=pd.read_csv(r"C:\Users\jaisa\OneDrive - Uppsala universitet\THESIS LIL GRUND\ct digitized.csv")    
powercurve1=pd.read_excel(r"D:\thesis\power curve digitized.xlsx")


# /// Creating various Error metric functions ///


def MAPE(Y_actual,Y_Predicted): #\\\ Mean average percentage error \\\
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape
def nrmse(y_actual,y_predicted): # \\\ Nomalized root mean square function \\\ 
    Nrmse = math.sqrt(mean_squared_error(y_actual, y_predicted))
    nrmse=Nrmse/(y_actual.max()-y_predicted.min())
    return nrmse
def rmse (y_actual, y_predicted): # \\\ Root mean squre Error \\\
    rmse= math.sqrt(mean_squared_error(y_actual, y_predicted))
    return rmse

# /// PRE-PROCESSING ///

# /// Filtering & Cleaning Raw Data ///
b1=b1[(b1[" AcWindSp"]>3.5)&(b1[" AcWindSp"]<15) ]
b1=b1[(b1[" Actpower"]>3.5)  & (b1[" ActPower"]>20)]

# Creating short cut for accessing the necesssary parameters from DataFrame 
wsp=b1[" AcWindSp"]
actpow=b1[" ActPower"]


# /// Convoluting the required parameter ///
N=300 # 30 seconds aggrigation 
wspave=np.convolve(b1[" AcWindSp"],  np.ones(N)/N, mode='same')
PCov=np.convolve(b1[" ActPower"],  np.ones(N)/N, mode='same')


# /// Creating Various functions and Data processsing ///


#/// Determining the optimum convolution frequency using MOB ///

# /// MOB model for wind speed and active power ///

def PfromPowercurve(wspave, N): # \\\ function for Calculating Power from wind speed \\\
    _powercurve=powercurve1.to_numpy()
    u = wsp.to_numpy()
    u = np.convolve( u, np.ones(N)/N, mode='same')
    Pinterp = np.interp(u, _powercurve[:,0], _powercurve[:,1])
    return Pinterp
def wspcalculated(powa,N): # \\\ function for Calculating  wind speed from Power \\\
    p=powa.to_numpy()
    _powercurvenew1=powercurve1.to_numpy()
    #_powercurve=powercurve.to_numpy()
    p = np.convolve( p, np.ones(N)/N, mode='same')
    b1["p"]=p
    rpinterp = np.interp(p,_powercurvenew1[:,0], _powercurvenew1[:,1] )
    #b1["rp"]=rpinterp
    return rpinterp

Ntest=[10,30,60]
n_rows = b1.shape[0]
step = 600
MAPEs = []
wspConv = []

PConv = []
for n in Ntest: # \\\ Creating a loop for new calulated power and wind speed \\\
    pwrCal=PfromPowercurve(wsp, n)
    PConv.append(pwrCal)
    mapeList = []
    wspcal1=wspcalculated(actpow, n)
    wspConv.append(wspcal1)
    mapewsp=[]
    for start in range(0, n_rows, step):
        end = start + step
        x_np = pwrCal[start : end]
        y_np = actpow[start : end]
        mapeList.append( MAPE(x_np, y_np) )
    MAPEs.append(mapeList)

    for start in range(0, n_rows, step):
        end = start + step
        x_np = wspcal1[start : end]
        y_np = wsp[start : end]
        mapewsp.append( MAPE(x_np, y_np) )
    mapewsp.append(mapeList)
        
ACTwspList1m = []
for start in range(0, n_rows, step):
    end = start + step
    df_slicew1m = wsp[start : end]

    ACTwspList1m.append( df_slicew1m.mean() )
ACTpowList1m = []
for start in range(0, n_rows, step):
    end = start + step
    df_slicep1m = actpow[start : end]

    ACTpowList1m.append( df_slicep1m.mean() )


TurbList = [] # \\\ Creating function for turbulence \\\
for start in range(0, n_rows, step):
    end = start + step
    df_slicetx = wsp[start : end]
    #df_slicety= [start : end]
    x_np = df_slicetx.to_numpy()
    #y_np = df_slicey.to_numpy()
    TurbList.append(  x_np.std()/x_np.mean() )


#/// Creating the plots for visual representation of the different convolution rates ///
for i, n in enumerate(Ntest):
    plt.plot(ACTpowList1m, MAPEs[i], marker='o', linestyle='none', label=str(n), alpha=0.09) 
    plt.xlabel('Actual Power (kW)')
    plt.ylabel('Error %')
    
    plt.plot(actpow, PConv[i], marker='o', linestyle='none', label=str(n), alpha=0.05)
    plt.plot(ACTwspList1m, MAPEs[i], marker='o', linestyle='none', label=str(n), alpha=0.9)
    plt.plot(wspave, PConv[i], marker='o', linestyle='none', label=str(n), alpha=0.05)
    plt.acorr(MAPEs[i], maxlags=100,label=str(n), alpha=0.5)
    print(np.mean(MAPEs[i]))
    plt.legend()


# /// MOB model for Tower Momentum ///



def CtFromthrustCruve(wspave, N): # \\\ function for Calculating Power from wind speed \\\
    _ctcurve=ctcurve.to_numpy()
    u = wsp.to_numpy()
    u = np.convolve( u, np.ones(N)/N, mode='same')
    Pinterp = np.interp(u, _powercurve[:,0], _powercurve[:,1])
    return Pinterp

ct=CtFromthrustCruve(wsp,N=1) 



Tmc=(ct*6362*1.204*wsp**2*0.068)/2 #\\\ Tower momentum calculated from MOB model \\\


#\\\ With this the  Methodology 1 flow chart is completed \\\

b1.to_csv(r"C:\Users\jaisa\OneDrive - Uppsala universitet\THESIS LIL GRUND\Process_data.csv")# \\\ Saving all the pre-processed data into one file \\\

    
# =============================================================================


# /// Creating Regression Models as in Methodology 2


# /// Calibration coefficient for converting raw strain gauge data into Tower base bending moment data ///
df["CalibratedG4"]=df[' StrainG4']*3.67-36430.8 
df["CalibratedG5"]=df[' StraingG5']*3.58-36808.9
df["Tower momentum from gauges"]=(df["CalibratedG4"]**2+df["CalibratedG5"]**2)**0.5

TM=df["Tower momentum from gauges"]  # \\\ RAW thrust force data \\\

df = pd.read_csv(r"C:\Users\jaisa\OneDrive - Uppsala universitet\THESIS LIL GRUND\Process_data.csv") # \\\ Loading all data \\\
df.head()

feature= df[[" AcWindSp"," Actpower"," NacelPos"," GenRpm"]] #\\\ Creating a seperate DataFrame for Traing data \\\

Target=df[" Tower momentum from gauges"] #\\\ DataFrame for Test data \\\
    

# /// Linear Regression ///

X= feature #\\\ for convinence \\\
y= Target #\\\ for convinence \\\

import numpy as np
from sklearn.linear_model import LinearRegression
lineareg = LinearRegression().fit(X, y)
lin_score=LinearRegression.reg.score(X, y)



# /// kNN Regression ///

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler().fit(X_train)   #\\\ Standardizing the data \\\
X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index.values, columns=X_train.columns.values)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index.values, columns=X_test.columns.values)


#Train-Test Split
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(feature, Target, test_size=0.25, random_state=42)
#Training

rmse_val = [] # \\\ to store rmse values for different k\\\
for K in range(20):
    K = K+1
    model = neighbors.KNeighborsRegressor(n_neighbors = K)

    model.fit(X_train, y_train)  # \\\ fit the model \\\
    pred=model.predict(X_test) # \\\ make prediction on test set \\\
    error = sqrt(mean_squared_error(y_test,pred)) # \\\calculate rmse \\\
    rmse_val.append(error) # \\\store rmse values \\\
    print('RMSE value for k= ' , K , 'is:', error)


predict = model.predict(X_test)



# /// Random forest classifier ///

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr


rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0) 
rf.fit(X_train, y_train)



predicted_train = rf.predict(X_train) # \\\ fit the model \\\
predicted_test = rf.predict(X_test)   # \\\ Testing 
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)




# procedure B is complete 


# =============================================================================


# /// Functions and loops for different Visualizations ///

for i, n in enumerate(Ntest):
    plt.plot(TurbList, MAPEs[i], marker='o', linestyle='none', label=str(n), alpha=0.09) 
    #plt.plot(ACTwspList1m, mapewsp[i], marker='o', linestyle='none', label=str(n), alpha=0.09) 
    plt.xlabel('Turbulence Intensity')
    plt.ylabel('Error %')
    plt.plot()
    #plt.plot(actpow, PConv[i], marker='o', linestyle='none', label=str(n), alpha=0.05)
    #plt.plot(ACTwspList1m, MAPEs[i], marker='o', linestyle='none', label=str(n), alpha=0.9)
    #plt.plot(wspave, PConv[i], marker='o', linestyle='none', label=str(n), alpha=0.05)
    #plt.acorr(MAPEs[i], maxlags=100,label=str(n), alpha=0.5)
    print(np.mean(MAPEs[i]))
plt.legend()
        
# /// Main graph for representation of Regresssion model and convolution  


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 5),dpi=100) 
fig.suptitle('Linear Regression',fontsize=15)
#plt.plot(df['Linear Regression'])
#plt.plot(df['TM'],df['Linear Regression'],marker='o', linestyle='none',label='30-Sec Avg',alpha=0.5)
axs[0,0].plot(df['TM'],df['Linear Regression 30'],marker='o', linestyle='none',label='30-Sec Avg',alpha=0.5) 
axs[0,0].plot(df.TM,df['Linear Regression 60'],marker='o', linestyle='none',label='10-min Avg',alpha=0.5)
axs[0,0].set_xlabel('Actual Tower Moment (kNm)')
axs[0,0].set_ylabel('Modelled Tower moment (kNm)')
axs[0,0].legend()
axs[0,0].set_title('Linear Regression')



axs[0,1].plot(df['TM'],df['Random Forest 30'],marker='o', linestyle='none',label='30-Sec Avg',alpha=0.5) 
axs[0,1].plot(df.TM,df['Random Forest 60'],marker='o', linestyle='none',label='10-min Avg',alpha=0.5)
axs[0,1].set_xlabel('Actual Tower Moment (kNm)')
axs[0,1].set_ylabel('Modelled Tower moment (kNm)')
axs[0,1].legend()
axs[0,1].set_title('Random Forest ')


axs[1,0].plot(df['TM 30'],df.kNN,marker='o', linestyle='none',label='30-Sec Avg',alpha=0.5) 
axs[1,0].plot(df.TM60,df.kNN,marker='o', linestyle='none',label='10-min Avg',alpha=0.5)
axs[1,0].set_xlabel('Actual Tower Moment (kNm)')
axs[1,0].set_ylabel('Modelled Tower moment (kNm)')
axs[1,0].legend()
axs[1,0].set_title('kNN')



axs[1,1].plot(df.AcWindSp,df.TM,marker='o', linestyle='none',label='30-Sec Avg',alpha=0.5) 
axs[1,1].plot(df.AcWindSp,df.TM,marker='o', linestyle='none',label='10-min Avg',alpha=0.5)
axs[1,1].set_xlabel('Actual Wind Speed (m/s)')
axs[1,1].set_ylabel('Actual Tower moment (kNm)')
axs[1,1].legend()


# /// Represntation of Error distiribution of various regression modelss ///


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7, 5),dpi=100) 
fig.suptitle('Linear Regression',fontsize=15)
#plt.plot(df['Linear Regression'])
#plt.plot(df['TM'],df['Linear Regression'],marker='o', linestyle='none',label='30-Sec Avg',alpha=0.5)
axs[0,0].plot(df['Linear Regression 30'],scipy.stats.norm.pdf(df['Linear Regression 30'], np.mean(df['Linear Regression 30']), np.std(df['Linear Regression 30'])), alpha=0.5,linewidth=3)
axs[0,0].set_xlabel('Standardized Residual Errors',fontsize=12)
axs[0,0].set_ylabel('probability of total error',fontsize=12)
axs[0,0].sns.displot(df.TM30,  bins=25, stat="probability")


axs[0,0].legend()
axs[0,0].set_title('Linear Regression')


axs[0,1].plot(df['Random Forest 30'],scipy.stats.norm.pdf(df['Random Forest 30'], np.mean(df['Random Forest 30']), np.std(df['Random Forest 30'])), alpha=0.5,linewidth=3)
axs[0,1].hist(df['Random Forest 30'],density=True,bins=25)
axs[0,1].set_xlabel('Standardized Residual Errors',fontsize=12)
axs[0,1].set_ylabel('probability of total error',fontsize=12)
axs[0,1].legend()
axs[0,1].set_title('Random Forest ')


axs[1,0].plot(df.kNN,scipy.stats.norm.pdf(newk3, np.mean(df.kNN), np.std((df.kNN)), alpha=0.5,linewidth=3)
axs[1,0].hist(df['kNN'],density=True,bins=25)
axs[1,0].set_xlabel('Standardized Residual Errors',fontsize=12)
axs[1,0].set_ylabel('probability of total error',fontsize=12)
axs[1,0].legend()
axs[1,0].set_title('kNN')



axs[1,1].plot(df.AcWindSp,df.TM,marker='o', linestyle='none',label='30-Sec Avg',alpha=0.5) 
axs[1,1].plot(df.AcWindSp,df.TM,marker='o', linestyle='none',label='10-min Avg',alpha=0.5)
axs[1,1].set_xlabel('Actual Wind Speed (m/s)',fontsize=12)
axs[1,1].set_ylabel('Actual Tower moment (kNm)',fontsize=12)
axs[1,1].legend()
axs[1,1].set_title('Actual data')