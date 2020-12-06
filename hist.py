import pandas as pd 
import matplotlib.pyplot as plt 

y = pd.read_csv("y.csv")
y_next = y.iloc[1:y.shape[0]]

print(y.shape)
y_sim6507=y.iloc[:,0:1725]
y_next_sim6507 = y_next.iloc[:,0:1725]
y_simTIA=y.iloc[:,1725:y.shape[1]]
y_next_simTIA = y_next.iloc[:,1725:y.shape[1]]

print(y_sim6507.shape,y_simTIA.shape)

label_6507 = []
for i in range(y_next_sim6507.shape[0]):
  label_6507.append(sum(y_next_sim6507.iloc[i]!=y_sim6507.iloc[i]))

fig, axs = plt.subplots(2)
fig.suptitle('Histogram of the Number of Wires which Changed States after one Step')
axs[0].hist(label_6507)
axs[0].ylabel("Number of Occurences")
axs[0].xlabel("Number of Wires which Changed States")
axs[0].title("Sim6507")
axs[0].savefig("sim6507_hist1.png")
axs[0].show()

label_tia = []
for i in range(y_next_simTIA.shape[0]):
  label_tia.append(sum(y_next_simTIA.iloc[i]!=y_simTIA.iloc[i]))
  
axs[1].hist(label_tia)
axs[1].ylabel("Number of Occurences")
axs[1].xlabel("Number of Wires which Changed States")
axs[1].title("SimTIA")
axs[1].savefig("simtia_hist1.png")
axs[1].show()




