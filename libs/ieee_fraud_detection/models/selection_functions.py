import numpy as np

def col_TransactionAmt(X):
    return np.log10(X[['TransactionAmt']])


def col_V1(X):
    return X[['V1']].fillna('NaN')


def col_V1NaN(X):
    return X[['V1']] == 'NaN'


def col_V2(X):
    return X[['V2']].fillna('NaN')


def col_V2NaN(X):
    return X[['V2']] == 'NaN'


def col_V3(X):
    return X[['V3']].fillna('NaN')


def col_V3NaN(X):
    return X[['V3']] == 'NaN'


def col_V4(X):
    return X[['V4']].fillna('NaN')


def col_V4NaN(X):
    return X[['V4']] == 'NaN'


def col_V5(X):
    return X[['V5']].fillna('NaN')


def col_V5NaN(X):
    return X[['V5']] == 'NaN'


def col_V6(X):
    return X[['V6']].fillna('NaN')


def col_V6NaN(X):
    return X[['V6']] == 'NaN'


def col_V7(X):
    return X[['V7']].fillna('NaN')


def col_V7NaN(X):
    return X[['V7']] == 'NaN'


def col_V8(X):
    return X[['V8']].fillna('NaN')


def col_V8NaN(X):
    return X[['V8']] == 'NaN'


def col_V9(X):
    return X[['V9']].fillna('NaN')


def col_V9NaN(X):
    return X[['V9']] == 'NaN'


def col_V10(X):
    return X[['V10']].fillna('NaN')


def col_V10NaN(X):
    return X[['V10']] == 'NaN'


def col_V11(X):
    return X[['V11']].fillna('NaN')


def col_V11NaN(X):
    return X[['V11']] == 'NaN'


def col_V12(X):
    return X[['V12']].fillna('NaN')


def col_V12NaN(X):
    return X[['V12']] == 'NaN'


def col_V13(X):
    return X[['V13']].fillna('NaN')


def col_V13NaN(X):
    return X[['V13']] == 'NaN'


def col_V14(X):
    return X[['V14']].fillna('NaN')


def col_V14NaN(X):
    return X[['V14']] == 'NaN'


def col_V15(X):
    return X[['V15']].fillna('NaN')


def col_V15NaN(X):
    return X[['V15']] == 'NaN'


def col_V16(X):
    return X[['V16']].fillna('NaN')


def col_V16NaN(X):
    return X[['V16']] == 'NaN'


def col_V17(X):
    return X[['V17']].fillna('NaN')


def col_V17NaN(X):
    return X[['V17']] == 'NaN'


def col_V18(X):
    return X[['V18']].fillna('NaN')


def col_V18NaN(X):
    return X[['V18']] == 'NaN'


def col_V19(X):
    return X[['V19']].fillna('NaN')


def col_V19NaN(X):
    return X[['V19']] == 'NaN'






def col_dist1(X):
    return np.log10(X[['dist1']].astype(float) + 1).fillna('NaN')


def col_dist1NaN(X):
    return X[['dist1']] == 'NaN'


def col_dist2(X):
    return X[['dist2']].fillna('NaN')


def col_dist2NaN(X):
    return X[['dist2']] == 'NaN'



def col_ProductCD(X):
    return X[['ProductCD']].fillna('NaN')


def col_card4(X):
    return X[['card4']].fillna('NaN')


def col_card5(X):
    return X[['card5']].fillna('NaN')


def col_card6(X):
    return X[['card6']].fillna('NaN')


def col_P_emaildomain(X):
    return X[['P_emaildomain']].fillna('NaN')


def col_R_emaildomain(X):
    return X[['R_emaildomain']].fillna('NaN')


def col_M4(X):
    return X[['M4']].fillna('NaN')




def col_DeviceType(X):
    return X[['DeviceType']].fillna('NaN')

def col_DevicePlatform(X):
    return X[['DevicePlatform']]

def col_id_12(X):
    return X[['id_12']].fillna('NaN')

def col_id_15(X):
    return X[['id_15']].fillna('NaN')

def col_id_16(X):
    return X[['id_16']].fillna('NaN')

def col_id_23(X):
    return X[['id_23']].fillna('NaN')

def col_id_27(X):
    return X[['id_27']].fillna('NaN')

def col_id_28(X):
    return X[['id_28']].fillna('NaN')

def col_id_29(X):
    return X[['id_29']].fillna('NaN')

def col_id_34(X):
    return X[['id_34']].fillna('NaN')


def col_hourofday(X):
    return X[['hourofday']]

def col_dayofweek(X):
    return X[['dayofweek']]

def col_dayofmonth(X):
    return X[['dayofmonth']]