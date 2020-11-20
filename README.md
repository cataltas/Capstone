# Capstone
The python file PCA.py has two functions which each perform PCA or tSNE.
The PNG file PCA_labels.png is the 3d scatterplot of the first three eigenvectors when performing PCA for the whole dataset.
The PNG file PCA_labels_2d.png is the 2d scatterplot of the first two eigenvectors when performing PCA for the whole dataset.
The PNG file TNSE.png is the 2d scatterplot of the two dimensions when performing tSNE on a randomized subset of 10,000 rows.

The labels for all three plots have been encoded such that each unique array has been attributed an integer value. For the whole dataset we have 8 such unique labels. 

The dataset used for this data visualisation is that of Eric Jonas' github repository. We will re-run these experiments once we have our actual dataset. 

Take aways: These plots are hard to interpret due to the fact that we have a disproportional amount of steps where all the transistors are off. 

Modeling script usage:
  
  ```python -u modeling_v4.py hidden_dim bsize ep LR```
