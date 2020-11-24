import pandas as pd

def downsample():


    #read in data
    X = pd.read_csv('X.csv')
    y = pd.read_csv('y.csv')
    print('original X and y length is:', len(X), flush = True)
    X = X.sort_values(by = 'time')
    y = y.sort_values(by = 'time')
    #flag duplicates 
    X['duplicate'] = X[X.columns.difference(['time'])].duplicated()
    y['duplicate'] = y[y.columns.difference(['time'])].duplicated()
    print('number of X and y duplicates are:', X['duplicate'].sum(), y['duplicate'].sum(), flush = True)
    X['duplicate'] = X['duplicate'].astype(int)
    y['duplicate'] = y['duplicate'].astype(int)


    X['duplicateplus1'] = X['duplicate'].shift(1)
    X['duplicatemin1'] = X['duplicate'].shift(-1)
    y['duplicateplus1'] = y['duplicate'].shift(1)
    y['duplicatemin1'] = y['duplicate'].shift(-1)


    if X['duplicate'].iloc[0] == 0 or X['duplicate'].iloc[1] == 0:
        X['duplicateplus1'] = X['duplicateplus1'].fillna(0)
    else:
        X['duplicateplus1'] =  X['duplicateplus1'].fillna(1)


    if y['duplicate'].iloc[0] == 0 or y['duplicate'].iloc[1] == 0:
        y['duplicateplus1'] = y['duplicateplus1'].fillna(0)
    else:
        y['duplicateplus1'] =  y['duplicateplus1'].fillna(1)

    if X['duplicate'].iloc[len(X['duplicate'])-1] == 0 or X['duplicate'].iloc[len(X['duplicate'])-1] == 0:
        X['duplicatemin1'] = X['duplicatemin1'].fillna(0)
    else:
        X['duplicatemin1'] =  X['duplicatemin1'].fillna(1)

    if y['duplicate'].iloc[len(y['duplicate'])-1] == 0 or y['duplicate'].iloc[len(y['duplicate'])-1] == 0:
        y['duplicatemin1'] = y['duplicatemin1'].fillna(0)
    else:
        y['duplicatemin1'] =  y['duplicatemin1'].fillna(1)


    X['sum'] = X['duplicateplus1']+X['duplicatemin1']+X['duplicate']
    y['sum'] = y['duplicateplus1']+y['duplicatemin1']+y['duplicate']

    X_final = X[X['sum'] != 3]
    y_final = y[y['sum'] != 3]
    X_final.drop(['duplicate', 'duplicateplus1', 'duplicatemin1', 'sum'], axis=1, inplace = True)
    y_final.drop(['duplicate', 'duplicateplus1', 'duplicatemin1', 'sum'], axis=1, inplace = True)
    
    X_exclude= [i for i in X_final['time'] if i not in y_final['time'].values]
    X_final = X_final[~X_final['time'].isin(X_exclude)]
    y_exclude = [i for i in y_final['time'] if i not in X_final['time'].values]
    y_final = y_final[~y_final['time'].isin(y_exclude)]

    assert len([i for i in X_final['time'] if i not in y_final['time'].values]) == 0
    assert len([i for i in y_final['time'] if i not in X_final['time'].values]) == 0

    print('final X and y shapes are:', X_final.shape, y_final.shape, flush = True)

   #save data
    X_final.to_csv("X_downsampled.csv", index=False)
    y_final.to_csv("y_downsampled.csv", index=False)    

if __name__ == "__main__":
    downsample()
