def append_data():
    column_names = pkl.load(open('outFrames/simTIA.pkl', 'rb'))
    names = column_names['WIRE_NAMES']
    #print(len(names), flush = True)
    names.append('time')
    simTIA_pd = pd.DataFrame(columns = names)
    count = 0
    for i in range(160001, 170001):
        count +=1 
        if count % 1000 ==0:
            print('have processes this many files:', count, flush = True)
        obj = pkl.load(open('outFrames_TIA/simTIA_{}.pkl'.format(i), 'rb'))
        #print(np.array(obj).reshape((len(obj), 1)).T.shape, flush = True)
        #print(len(names), flush = True)
        column_names = pkl.load(open('outFrames/simTIA.pkl', 'rb'))
        current_row = pd.DataFrame(np.array(obj).reshape((len(obj), 1)).T, columns = column_names['WIRE_NAMES'])
        current_row['time'] = i
        simTIA_pd = simTIA_pd.append(current_row)
    simTIA_pd.to_csv("simTIA_clean_{}.csv".format('170000'), index=False)



if __name__ == '__main__':
    import pickle as pkl
    import numpy as np
    import pandas as pd
    append_data()
