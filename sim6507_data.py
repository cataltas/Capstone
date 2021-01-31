def append_data():
    column_names = pkl.load(open('outFrames/sim6507.pkl', 'rb'))
    names = column_names['WIRE_NAMES']
    #print(len(names), flush = True)
    names.append('time')
    #print(names_new, flush = True)
    sim6507_pd = pd.DataFrame(columns = names)
    count = 0
    for i in range(130001, 140001):
        count +=1 
        if count % 5000 ==0:
            print('have processes this many files:', count, flush = True)
        obj = pkl.load(open('outFrames_6507/sim6507_{}.pkl'.format(i), 'rb'))
        #print(np.array(obj).reshape((len(obj), 1)).T.shape, flush = True)
        #print(len(names), flush = True)
        column_names = pkl.load(open('outFrames/sim6507.pkl', 'rb'))
        current_row = pd.DataFrame(np.array(obj).reshape((len(obj), 1)).T, columns = column_names['WIRE_NAMES'])
        current_row['time'] = i
        sim6507_pd = sim6507_pd.append(current_row)
    sim6507_pd.to_csv("sim6507_clean_{}.csv".format('140000'), index=False)



if __name__ == '__main__':
    import pickle as pkl
    import numpy as np
    import pandas as pd
    append_data()
