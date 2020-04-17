import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
import datetime,time
from sklearn.utils.extmath import density
import sklearn.gaussian_process as gp

# returns the train-test split
# output: xTr,xTe,yTrs,yTes:
#         yTrs and yTes are columns for each training set.
def load_traffic(fpath, out_vars):
    df = pd.read_csv(fpath)
    df = df.dropna() # questionable: this halves our data
    print("--table read--")
    for item in out_vars: assert item in df.columns[-13:]
    # input variables: without traffic counts
    X = df.iloc[:,:-13]
    # feat_lst = ['latitude', 'longitude', 'year', 'count_date', 'hour',
    #            'road_type','direction_of_travel','link_length_km']
    # X = df.loc[:,feat_lst]
    del X['local_authoirty_ons_code']
    del X['local_authority_id']
    del X['region_ons_code']
    del X['region_id']
    del X['count_point_id']
    del X['start_junction_road_name']
    del X['end_junction_road_name']
    ys = df[out_vars]
    print(X.columns)
    del df
    print("--x and y separated--")
    # replace datestring with day of year
    X.loc[:,"count_date"] = X["count_date"].map(
        lambda x: datetime.datetime.strptime(x[5:],'%m-%d').timetuple().tm_yday
        )
    cats = [col for col in X.columns if X[col].dtype=='O']
    # dogs = [col for col in X.columns if X[col].dtype!='O']
    # X = X.fillna(X[dogs].mean())
    xTrain,xTest,yTrain,yTest = train_test_split(X, ys, test_size=0.1)
    print("--train_test_split--")
    # convert categorical data into numerical
    # enc = OrdinalEncoder()
    enc = OneHotEncoder()
    enc.fit(X[cats])
    
    xenc = enc.transform(xTrain[cats])
    for cat in cats:
        del xTrain[cat]
    xTrainSp = scipy.sparse.csr_matrix(xTrain)
    xTrain = scipy.sparse.hstack((xTrainSp,xenc))

    xenc = enc.transform(xTest[cats])
    for cat in cats:
        del xTest[cat]
    xTestSp = scipy.sparse.csr_matrix(xTest)
    xTest = scipy.sparse.hstack((xTestSp,xenc))

    print("--cat -> num--")
    scaler = StandardScaler(with_mean=False)
    scaler.fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)
    # outputs that we wish to predict
    return xTrain,xTest,yTrain,yTest,enc

def load_air_sparse(fpath):
    df = pd.read_csv(fpath)
    print("--table read--")
    # input variables: without traffic counts
    y = df['no2_ugm3']
    X = df.iloc[:,3:] # rm unnamed: 0 and date_UTC
    del X['ratification_status']
    del X['date']
    del X['pod_id_location']
    del df
    print(X.columns)
    print(X.head())
    print("--x and y separated--")
    cats = [col for col in X.columns if X[col].dtype=='O']
    # catvals = X[cats].fillna('0')
    # del X[cats]
    values = {col:'0' if col in cats else 0 for col in X.columns}
    X = X.fillna(values)
    X['sin_hour'] = np.sin((X['hour']-2.0)/6.0*np.pi)
    xTrain,xTest,yTrain,yTest = train_test_split(X, y, test_size=0.1)
    print("--train_test_split--")
    # convert categorical data into numerical
    # enc = OrdinalEncoder()
    onehotenc = OneHotEncoder()
    #enc.fit(catvals)
    cats.extend(['pod_id','hour'])
    onehotenc.fit(X[cats])
    del X
    del y

    xenc = onehotenc.transform(xTrain[cats])
    for cat in cats:
        del xTrain[cat]
    xTrainSp = scipy.sparse.csr_matrix(xTrain)
    xTrain = scipy.sparse.hstack((xTrainSp,xenc))

    xenc = onehotenc.transform(xTest[cats])
    for cat in cats:
        del xTest[cat]
    xTestSp = scipy.sparse.csr_matrix(xTest)
    xTest = scipy.sparse.hstack((xTestSp,xenc))

    print("--cat -> num--")
    scaler = StandardScaler(with_mean=False)
    scaler.fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)
    # outputs that we wish to predict
    return xTrain,xTest,yTrain,yTest,onehotenc

def load_air_from_split(fpaths=None):
    assert fpaths==None or len(fpaths)==4
    direct = 'data/'
    fileext = '.csv'
    tr_fe = pd.read_csv(direct+'train_features'+fileext)
    te_fe = pd.read_csv(direct+'test_features'+fileext)
    tr_la = pd.read_csv(direct+'train_labels'+fileext)
    te_la = pd.read_csv(direct+'test_labels'+fileext)
    return tr_fe,te_fe,tr_la,te_la

def process_air(xTrain,xTest):
    rem_fields = ['date_UTC','ratification_status', 'date']
    for rf in rem_fields:
        del xTrain[rf]
        del xTest[rf]
    xTrain['sin_hour'] = np.sin((xTrain['hour']-2.0)/6.0*np.pi)
    xTest['sin_hour'] = np.sin((xTest['hour']-2.0)/6.0*np.pi)
    scaler = StandardScaler(with_mean=False)
    scaler.fit(xTrain)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)
    return xTrain,xTest

if __name__ == "__main__":
    #lst = ["all_hgvs", "all_motor_vehicles"]
    #xTr,xTe,yTr,yTe,cat_enc = load_traffic('data/dft_traffic_counts_raw_counts.csv', lst)
    lst = ['no2_ugm3']
    # xTr,xTe,yTr,yTe,cat_enc = load_air_sparse('data/air_met_data.csv')
    xTr,xTe,yTr,yTe = load_air_from_split()
    xTr,xTe = process_air(xTr,xTe)
    kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    gp_model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
    models = [LinearRegression(),
              Ridge(),
              #Lasso(),
              #ElasticNet(l1_ratio = 0.75),
              gp_model
              ]
    for outvar in lst:
        print(outvar)
        print(yTr)
        if len(lst)==1:
            yTr_i = yTr
            yTe_i = yTe
        else:
            yTr_i = yTr[outvar]
            yTe_i = yTe[outvar]
        for i,model in enumerate(models):
            print('  '+model.__repr__())
            time_st = time.process_time()
            model.fit(xTr,yTr_i)
            print('  training time  : %f' % (time.process_time()-time_st))
            if hasattr(model, 'coef_'):
                print("  density: %f" % density(model.coef_))
            print('  R_sq: training : %f' % model.score(xTr,yTr_i))
            print('  R_sq: testing  : %f' % model.score(xTe,yTe_i))
            print()
        print()

# naive bayes