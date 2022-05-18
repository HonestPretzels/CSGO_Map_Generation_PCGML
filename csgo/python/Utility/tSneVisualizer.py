from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

def main():
    X = np.load(sys.argv[1])
    print(X.shape)
    y = np.load(sys.argv[2])
    X = np.reshape(X, (X.shape[0], 30*512))
    y = np.argmax(y, axis=1)
    
    print(X.shape, y.shape)
    # Randomly select 1000 samples for performance reasons
    # np.random.seed(100)
    # subsample_idc = np.random.choice(X.shape[0], 3000, replace=False)
    # X = X[subsample_idc,:]
    # y = y[subsample_idc]
    
    # We want to get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    tsne_result.shape
    # (1000, 2)
    # Two dimensions for each of our images
    
    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()

if __name__ == "__main__":
    main()