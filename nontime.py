import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, Normalizer
import datetime

# returns the train-test split
# output: xTr,xTe,yTrs,yTes:
#         yTrs and yTes are columns for each training set.
def load_data(fpath, out_vars):
    # try:
    #     df = pd.read_csv(fpath[:-4]+'_cleaned_'+'.csv')
    # except expression as identifier:
    #     pass
    # finally:
    #     pass
    df = pd.read_csv(fpath)
    df = df.dropna() # questionable: this halves our data
    print("--table read--")
    for item in out_vars: assert item in df.columns[-12:]
    # input variables: without traffic counts
    X = df.iloc[:,:-12]
    # feat_lst = ['latitude', 'longitude', 'year', 'count_date', 'hour',
    #            'road_type','direction_of_travel','link_length_km']
    # X = df.loc[:,feat_lst]
    ys = df[out_vars]
    del df
    print("--x and y separated--")
    # replace datestring with day of year
    X.loc[:,"count_date"] = X["count_date"].map(
        lambda x: datetime.datetime.strptime(x[5:],'%m-%d').timetuple().tm_yday
        )
    xTrain,xTest,yTrain,yTest = train_test_split(X, ys, test_size=0.1)
    print("--train_test_split--")
    cats = [col for col in X.columns if X[col].dtype=='O']
    # dogs = [col for col in X.columns if X[col].dtype!='O']
    # convert categorical data into numerical
    enc = OrdinalEncoder()
    enc.fit(xTrain[cats])
    print("--huh?--")
    xTrain.loc[:,cats] = enc.transform(xTrain[cats])
    print("--huh2?--")
    xTest.loc[:,cats] = enc.transform(xTest[cats])
    print("--cat -> num--")
    normer = Normalizer().fit(xTrain)
    xTrain.loc[:,:] = normer.transform(xTrain)
    xTest.loc[:,:] = normer.transform(xTest)
    # outputs that we wish to predict
    return xTrain,xTest,yTrain,yTest,enc

if __name__ == "__main__":
    lst = ["all_hgvs", "all_motor_vehicles"]
    xTr,xTe,yTr,yTe,cat_enc = load_data('dft_traffic_counts_raw_counts.csv', lst)
    models = [LinearRegression(), Lasso()]
    for outvar in lst:
        print(outvar)
        yTr_i = yTr[outvar]
        yTe_i = yTe[outvar]
        for model in models:
            print('  '+model.__repr__())
            model.fit(xTr,yTr_i)
            print('  R_sq: training : %f' % model.score(xTr,yTr_i))
            print('  R_sq: testing  : %f' % model.score(xTe,yTe_i))
            print()
        print()

# naive bayes