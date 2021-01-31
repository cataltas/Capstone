def append_data():
    #column_names = pkl.load(open('outFrames/sim6507.pkl', 'rb'))
    #names = column_names['WIRE_NAMES']
    #print(len(names), flush = True)
    names = list(np.arange(1, 153).astype(str))
    names.append('time')
    emu_pd = pd.DataFrame(columns = names)
    count = 0
    for i in range(220001, 236088):
        count +=1 
        if count % 5000 ==0:
            print('have processes this many files:', count, flush = True)
        obj = pkl.load(open('outFrames_emu/emuPIA_{}.pkl'.format(i), 'rb'))
        #print(np.array(obj).reshape((len(obj), 1)).T.shape, flush = True)
        #print(len(names), flush = True)
        #column_names = pkl.load(open('outFrames/sim6507.pkl', 'rb'))
        row_emu = np.append(obj['RAM'], obj['IOT'], axis = 0)
        names = list(np.arange(1, 153).astype(str))
        current_row = pd.DataFrame(np.array(row_emu).reshape((len(row_emu), 1)).T, columns = names)
        current_row['time'] = i
        emu_pd = emu_pd.append(current_row)
    emu_pd.to_csv("emu_clean_{}.csv".format('236088'), index=False)



if __name__ == '__main__':
    import pickle as pkl
    import numpy as np
    import pandas as pd
    append_data()
