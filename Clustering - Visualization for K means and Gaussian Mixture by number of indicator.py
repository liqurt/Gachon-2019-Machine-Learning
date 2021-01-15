import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Visualize(df):
    X = []
    Y = []
    for i in range(df.shape[0]):
        xSample = df.loc[i, :]
        X.append(xSample.count())
        Y.append(df.iloc[i,-1])
        print("i : {},X : {},Y : {}".format(i, X[i], Y[i]))
# title is various by algorithm
    plt.title("Gaussian Mixture score by indicator")
    plt.xlabel("The number of indicator")
    plt.ylabel("Score")
    plt.plot(X, Y)
    plt.show()

KORdataLocation = "C:/Users/Y190313/Desktop/Apocrypha/4-2(19-2)/Machine Learning/TP/world-development-indicators/Korea data/"
OriginalFile = np.array([KORdataLocation+"indicators1344-3-0-end.csv",
                         KORdataLocation+"indicators1344-3-1-end.csv",
                         KORdataLocation+"indicators1344-3-2-end-g.csv"])
df_file0 = pd.read_csv(OriginalFile[0])
df_file1 = pd.read_csv(OriginalFile[1])
df_file2 = pd.read_csv(OriginalFile[2])

Visualize(df_file0)
Visualize(df_file1)
Visualize(df_file2)

