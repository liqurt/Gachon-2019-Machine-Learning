import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from sklearn.metrics import silhouette_score

Indicator_birthRate = "SP.DYN.CBRT.IN"
OriginalData0 = pd.read_csv("Indicators1344-3-0-p(updated2).csv")

df0_avg = OriginalData0.groupby(['CountryCode'], as_index=False).mean().drop('Year', axis=1)
df0_avg_noCC = df0_avg.drop('CountryCode',axis=1)

graphX = []
graphY = []

EPS = 0
MINSAMPLE = 0

###

EPS_ = 0.15
MINSAMPLE_ = 6
#DBSCAN
colListLocation = pd.read_csv("Korea data/indicators1344-3-0-end.csv")
for row in range(colListLocation.shape[0]):
    # nowRow from K means's result
    nowRow = np.array(colListLocation.iloc[row, :-1].to_frame().dropna())
    nowRow = np.squeeze(nowRow)
    df0_avg_noCC = df0_avg.drop('CountryCode', axis=1)
    df0_avg_noCC = df0_avg_noCC[nowRow]
    # scaling for feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df0_avg_noCC)
    X_normalized = normalize(X_scaled)
    # make lower dimension
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(X_normalized)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    # make DBSCAN clustering model
    db_default = DBSCAN(eps=EPS_, min_samples=MINSAMPLE_).fit(X_principal)
    labels = db_default.labels_
    colours = {}
    colours[0] = 'r'
    colours[1] = 'g'
    colours[2] = 'b'
    colours[3] = 'm'
    colours[4] = 'y'
    colours[-1] = 'k'

    # erase ''' for visualization DBSCAN graph(WARNING : you can see 219 times of graph)
    '''
    if row == lowPoint or row == highPoint:
        print("Combination : ",nowRow)
        cvec = [colours[label] for label in labels]
        r = plt.scatter(X_principal['P1'], X_principal['P2'], color='r')
        g = plt.scatter(X_principal['P1'], X_principal['P2'], color='g')
        b = plt.scatter(X_principal['P1'], X_principal['P2'], color='b')
        m = plt.scatter(X_principal['P1'], X_principal['P2'], color='m')
        y = plt.scatter(X_principal['P1'], X_principal['P2'], color='y')
        k = plt.scatter(X_principal['P1'], X_principal['P2'], color='k')
        plt.figure(figsize=(5, 5))
        plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
        plt.show()
    '''
    graphX.append(row)
    graphY.append(silhouette_score(X_principal, labels, metric='euclidean'))
    print("Row{} / Silhouette score".format(row),silhouette_score(X_principal, labels, metric='euclidean'))
    EPS = db_default.eps
    MINSAMPLE = db_default.min_samples

plt.title("DBSCAN")
plt.suptitle("EPS : {}, MIN_SAMPLE : {}".format(EPS,MINSAMPLE))
plt.xlabel("attempt")
plt.ylabel("Silhouette score")
plt.plot(graphX,graphY)
plt.show()
