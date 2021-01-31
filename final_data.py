def append_6507():
    sim6507 = pd.read_csv('sim6507_clean_10000.csv')
    sim6507_pd = sim6507.append(pd.read_csv('sim6507_clean_20000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_30000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_50000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_60000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_70000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_80000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_90000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_100000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_120000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_130000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_140000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_160000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_180000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_200000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_220000.csv'))
    sim6507_pd = sim6507_pd.append(pd.read_csv('sim6507_clean_236088.csv'))
    for i in range(1, 236088):
        print(i)
        assert i in sim6507_pd['time'].values
    sim6507_pd.to_csv('sim6507_clean_final.csv', index =False)

def append_tia():

    simTIA = pd.read_csv('simTIA_clean_20000.csv')
    simTIA_pd = simTIA.append(pd.read_csv('simTIA_clean_40000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_60000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_80000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_100000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_120000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_130000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_140000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_150000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_160000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_170000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_180000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_190000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_200000.csv'))
    simTIA_pd = simTIA_pd.append(pd.read_csv('simTIA_clean_236088.csv'))
    for i in range(1, 236088):
        print(i)
        assert i in simTIA_pd['time'].values
    simTIA_pd.to_csv('simTIA_clean_final.csv', index = False)

def append_emu():
    emu = pd.read_csv('emu_clean_20000.csv')
    emu_pd = emu.append(pd.read_csv('emu_clean_40000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_60000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_80000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_100000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_120000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_140000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_160000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_180000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_200000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_220000.csv'))
    emu_pd = emu_pd.append(pd.read_csv('emu_clean_236088.csv'))
    for i in range(1, 236088):
        print(i)
        assert i in emu_pd['time'].values
    emu_pd.to_csv('emu_clean_final.csv', index = False)

def model_predictions():
    preds = pd.read_csv('model_predictions_100000.csv')
    #preds = preds.append(pd.read_csv('model_predictions_100000.csv'))
    #preds = preds.append(pd.read_csv('model_predictions_150000.csv'))
    preds = preds.append(pd.read_csv('model_predictions_236088.csv'))
    preds.to_csv('model_predictions.csv', index = False)
   
def create_data():
    #read in raw data
    emu = pd.read_csv('emu_clean_final.csv')
    sim6507 = pd.read_csv('sim6507_clean_final.csv')
    simTIA = pd.read_csv('simTIA_clean_final.csv')

    #create X dataframe
    X = pd.merge(emu, sim6507, on=['time'], how= 'inner')
    X = X.sort_values(by  = 'time')

    #create y dataframe
    y = X[X.columns[152:]]
    mask = ~(y.columns.isin(['time']))
    cols_to_shift = y.columns[mask]
    y[cols_to_shift] = y.loc[:,mask].shift(-1)
    y.fillna(method='ffill', inplace = True)
    y = pd.merge(y, simTIA, on=['time'], how= 'inner')


    assert len([i for i in y['time'] if i not in X['time'].values]) == 0
    assert len([i for i in X['time'] if i not in y['time'].values]) == 0
    assert X.shape[0] == y.shape[0]
    #save data
    X.to_csv("X.csv", index=False)
    y.to_csv("y.csv", index=False)


if __name__ == '__main__':
    import pickle as pkl
    import numpy as np
    import pandas as pd
    #append_6507()
    #append_emu()
    #append_tia()
    #create_data()
    model_predictions()
