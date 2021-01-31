
def isHigh(current_row):
    return bool((current_row & (ANY_HIGH)))
def isLow(current_row):
    return bool( current_row & ANY_LOW)
def get4BitColor(current_row):
    col = 0
    if isHigh(current_row['COLCNT_T0'][0]):
        col += 1
    if isHigh(current_row['COLCNT_T1'][0]):
        col += 2
    if isHigh(current_row['COLCNT_T2'][0]):
        col += 4
    if isHigh(current_row['COLCNT_T3'][0]):
        col += 8
    return col

def get3BitLuminance(current_row):
    lum = 7

    # If L0_lowCtrl is high, then the pad for the least significant bit of
    # luminance is pulled low, so subtract 1 from the luminance
    if isHigh(current_row['L0_lowCtrl'][0]):
        lum -= 1

    # If L1_lowCtrl is high, then the pad for the twos bit of luminance
    # is pulled low, so subtract 2 from the luminance
    if isHigh(current_row['L1_lowCtrl'][0]):
        lum -= 2

    # If the most significant bit is pulled low, subtract 4
    if isHigh(current_row['L2_lowCtrl'][0]):
        lum -= 4
        
    return lum

def initColLumLUT():
    colLumToRGB8LUT = []
    col = [[]] * 16
    col[0]  = [(0,0,0),        (236, 236, 236)]
    col[1]  = [(68, 68, 0),    (252, 252, 104)]
    col[2]  = [(112, 40, 0),   (236, 200, 120)]
    col[3]  = [(132, 24, 0),   (252, 188, 148)]
    col[4]  = [(136, 0, 0),    (252, 180, 180)]
    col[5]  = [(120, 0, 92),   (236, 176, 224)]
    col[6]  = [(72, 0, 120),   (212, 176, 252)]
    col[7]  = [(20, 0, 132),   (188, 180, 252)]
    col[8]  = [(0, 0, 136),    (164, 164, 252)]
    col[9]  = [(0, 24, 124),   (164, 200, 252)]
    col[10] = [(0, 44, 92),    (164, 224, 252)]
    col[11] = [(0, 60, 44),    (164, 252, 212)]
    col[12] = [(0, 60, 0),     (184, 252, 184)]
    col[13] = [(20, 56, 0),    (200, 252, 164)]
    col[14] = [(44, 48, 0),    (224, 236, 156)]
    col[15] = [(68, 40, 0),    (252, 224, 140)]
    
    colLumToRGB8LUT = [0]*128
    
    for intKey in range(len(col)):
        colPair = col[intKey]
        start = colPair[0]
        end   = colPair[1]
        dif = ()
        for i, startv in enumerate(start):
            # result is tuple of same dim as 'start' and 'end'      
            dif += (end[i] - startv,)
        # lumInt from 0 to 7
        for lumInt in range(8):
            lumFrac = lumInt / 7.0
            ctup = ()
            for i, startv in enumerate(start):
                ctup += (int(startv + dif[i]*lumFrac),)
            colLumInd = (intKey << 3) + lumInt
            colLumToRGB8LUT[colLumInd] = ctup
    return colLumToRGB8LUT
def getColorRGBA8(current_row):
    lum = get3BitLuminance(current_row)
    col = get4BitColor(current_row)

        # Lowest 4 bits of col, shift them 3 bits to the right,
        # and add the low 3 bits of luminance
    index = ((col & 0xF) << 3) + (lum & 0x7)
    #print('index', index, flush = True)
    #print(colLumToRGB8LUT)
    rgb8Tuple = colLumToRGB8LUT[index]
    return (rgb8Tuple[0] << 24) | (rgb8Tuple[1] << 16) | \
               (rgb8Tuple[2] << 8) | 0xFF

def generate_image():
    imagePIL_new = imagePIL.getInterface()
    count = 0
    for i in range(1, 143366):
        count += 1
        if count % 5000 == 0:
            print('proccesses this many files {}'. format(count), flush = True)
          
        
        obj = pkl.load(open('outFrames_TIA/simTIA_{}.pkl'.format(i), 'rb'))
        column_names = pkl.load(open('outFrames/simTIA.pkl', 'rb'))
        current_row = pd.DataFrame(np.array(obj).reshape((len(obj), 1)).T, columns = column_names['WIRE_NAMES'])
        #print(current_row['VSYNC'] , flush = True)
        if isLow(current_row['CLK0'][0]):
            print('CLK0 low at {}'.format(i), current_row['CLK0'][0], flush = True)
            restartImage = False
            if isHigh(current_row['VSYNC'][0]):
                
                print('VSYNC hight at {}'.format(i), current_row['VSYNC'][0], flush = True)
                restartImage = True
            rgba = getColorRGBA8(current_row)
            if imagePIL_new != None:
                if restartImage == True:
                    imagePIL_new.restartImage()
                imagePIL_new.setNextPixel(rgba)
    
if __name__ == '__main__':
    import pickle as pkl
    import numpy as np
    import pandas as pd
    import cython
    import imagePIL
    PULLED_HIGH  = 1 << 0 # 1 
    PULLED_LOW     = 1 << 1 # 2
    GROUNDED       = 1 << 2 # 4
    HIGH           = 1 << 3 # 8
    FLOATING_HIGH  = 1 << 4 # 16
    FLOATING_LOW   = 1 << 5 # 32
    FLOATING       = 1 << 6 # 64cpdef enum WireState:
    PULLED_HIGH  = 1 << 0 # 1 
    PULLED_LOW     = 1 << 1 # 2
    GROUNDED       = 1 << 2 # 4
    HIGH           = 1 << 3 # 8
    FLOATING_HIGH  = 1 << 4 # 16
    FLOATING_LOW   = 1 << 5 # 32
    FLOATING       = 1 << 6 # 64
    ANY_HIGH = (FLOATING_HIGH | HIGH | PULLED_HIGH)
    ANY_LOW  = (FLOATING_LOW | GROUNDED | PULLED_LOW)
    colLumToRGB8LUT = initColLumLUT()
    #print(colLumToRGB8LUT, flush = True)
    generate_image()
    
    

    
    
    

