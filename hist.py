import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

y = pd.read_csv("y.csv")
y= y.drop("time",axis=1)

y_next = y.iloc[1:y.shape[0]]


y_sim6507=y.iloc[:,0:1725]
y_next_sim6507 = y_next.iloc[:,0:1725]
y_simTIA=y.iloc[:,1725:y.shape[1]]
y_next_simTIA = y_next.iloc[:,1725:y.shape[1]]



# label_6507 = []
# for i in range(y_next_sim6507.shape[0]):
#   label_6507.append(sum(y_next_sim6507.iloc[i]!=y_sim6507.iloc[i]))

# fig, axs = plt.subplots(2,1,figsize=(15,15))
# fig.suptitle('Frequency of the Total Number of Wires which Changed States after One Step')
# axs[0].hist(label_6507,bins=20)
# axs[0].set_ylabel("Number of Occurences")
# axs[0].set_xlabel("Number of Wires which Changed States")
# axs[0].set_title("Sim6507")

# label_tia = []
# for i in range(y_next_simTIA.shape[0]):
#   label_tia.append(sum(y_next_simTIA.iloc[i]!=y_simTIA.iloc[i]))
  
# axs[1].hist(label_tia,bins=20)
# axs[1].set_ylabel("Number of Occurences")
# axs[1].set_xlabel("Number of Wires which Changed States")
# axs[1].set_title("SimTIA")
# fig.savefig("hist1.png")

# label_2_6507= []
# for i in range(y_next_sim6507.shape[0]):
#     label_2_6507.append(np.where(y_next_sim6507.iloc[i]!=y_sim6507.iloc[i])[0])
# label_2_6507= np.concatenate(label_2_6507,axis=0)

# fig2, axs2 = plt.subplots(2,1,figsize=(15,15))
# fig2.suptitle('Number of Times Each Wire Changed State')
# axs2[0].hist(label_2_6507,bins=100)
# axs2[0].set_ylabel("Number of Times Changed")
# axs2[0].set_xlabel("Wires")
# axs2[0].set_title("Sim6507")

# label_2_tia= []
# for i in range(y_next_simTIA.shape[0]):
#     label_2_tia.append(np.where(y_next_simTIA.iloc[i]!=y_simTIA.iloc[i])[0])
# label_2_tia= np.concatenate(label_2_tia,axis=0)

# axs2[1].hist(label_2_tia,bins =100)
# axs2[1].set_ylabel("Number of Times Changed")
# axs2[1].set_xlabel("Wires")
# axs2[1].set_title("SimTIA")
# fig2.savefig("hist2.png")

columns = ['COLCNT_T0', 'COLCNT_T1', 'COLCNT_T2', 'COLCNT_T3','L0_lowCtrl', 'L1_lowCtrl', 'L2_lowCtrl']
y_next_simTIA_render = y_next_simTIA[columns]
label_3_TIA= y_next_simTIA_render.values.tolist()
(values,counts)=np.unique(label_3_TIA,return_counts=True)
fig3 = plt.figure(figsize=(15,15))
plt.bar(x=values,height=counts)
plt.xlim(0,32)
plt.title('Frequency of the Values of the Wires Used for Image Rendering')
plt.ylabel("Frequency")
plt.xlabel("Wire Value")
fig3.savefig("hist3.png")

