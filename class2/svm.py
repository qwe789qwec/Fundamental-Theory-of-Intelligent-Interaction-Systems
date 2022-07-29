import pandas as pd
import numpy as np
import plotly.express as px
import math
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import plotly.graph_objects as go

# refence
# https://wizardforcel.gitbooks.io/scikit-and-tensorflow-workbooks-bjpcjp/content/ch05-support-vector-machines.html
# https://jcabelloc.github.io/machine%20learning/2019/04/08/linear-svm.html

def print_svm_linear(data):
    # X = data.iloc[:,1:].values  # Get non-class columns
    # y = data.iloc[:, 0].values  # Get class columns
    X = data.iloc[:,0:-1].values  # Get non-class columns
    y = data.iloc[:, -1].values  # Get class columns
    # nor_data = normalize_data(y)
    # print(data)
    # print(X)
    # print(y)
    clf = svm.SVC(kernel='linear') # Linear Kernel
    #Train the model using the training sets

    clf.fit(X, y)
    y_pred = clf.predict(X)
    # print("Accuracy:",metrics.accuracy_score(y, y_pred))
    w = clf.coef_
    b = clf.intercept_
    x0 = np.linspace(-5, 5, 200)
    decision_boundary = [(-w0 / w1) * x0 - (b0 / w1) for w0,w1,b0 in zip(w[:,0], w[:,1], b)]
    # print(type(x0))
    class12_up = (decision_boundary[0] + 1/w[0,1]).tolist()
    class12_down = (decision_boundary[0] - 1/w[0,1]).tolist()
    class13_up = (decision_boundary[1] + 1/w[1,1]).tolist()
    class13_down = (decision_boundary[1] - 1/w[1,1]).tolist()
    class23_up = (decision_boundary[2] + 1/w[2,1]).tolist()
    class23_down = (decision_boundary[2] - 1/w[2,1]).tolist()
    # margin = 1/w[1]

    num_data = data.replace({"class": {'1': 1, '2': 2, '3': 3}})
    svs = clf.support_vectors_
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=svs[:, 0], y=svs[:, 1],
                    mode='markers',
                    name='support_vectors',
                    opacity=0.5,
                    marker=dict(
                    size=10,
                    color='LightSkyBlue',
                        line=dict(
                            color='MediumPurple',
                            width=20
                    ))))
    fig.add_trace(go.Scatter(x=num_data['pca1'], y=num_data['pca2'],
                    mode='markers',
                    name='wine datas',
                    marker=dict(
                    size=10,
                    color=num_data['class'],
                    colorscale=[[0, 'red'], [0.5, 'green'], [1, 'blue']],)))
    fig.add_trace(go.Scatter(x=x0, y=decision_boundary[0],
                    line_color='rgb(0,100,80)',
                    mode='lines',
                    name='class1/2'))
    fig.add_trace(go.Scatter(x=x0, y=decision_boundary[1],
                    line_color='rgb(0,176,246)',
                    mode='lines',
                    name='class1/3'))
    fig.add_trace(go.Scatter(x=x0, y=decision_boundary[2],
                    line_color='rgb(231,107,243)',
                    mode='lines',
                    name='class2/3'))

    fig.add_trace(go.Scatter(
        x=x0.tolist() + x0.tolist()[::-1],
        y=class12_up + class12_down[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='class1/2',
    ))
    fig.add_trace(go.Scatter(
        x=x0.tolist() + x0.tolist()[::-1],
        y=class13_up + class13_down[::-1],
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='class1/3',
    ))
    fig.add_trace(go.Scatter(
        x=x0.tolist() + x0.tolist()[::-1],
        y=class23_up + class23_down[::-1],
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='class2/3',
    ))

    fig.update_traces(marker_size=10)
    fig.update_layout(
        title="Support Vector Machine",
        xaxis_title="pca1",
        yaxis_title="pca2",
        font=dict(
        # family="Courier New, monospace",
        size=32,
        # color="RebeccaPurple"
        ),
        yaxis_range=[-5,5],
        xaxis_range=[-5,5],
    )
    fig.show()

def print_svm_rbf(data):
    # X = data.iloc[:,1:].values  # Get non-class columns
    # y = data.iloc[:, 0].values  # Get class columns
    X = data.iloc[:,0:-1].values  # Get non-class columns
    y = data.iloc[:, -1].values  # Get class columns
    # nor_data = normalize_data(y)
    # print(data)
    # print(X)
    # print(y)
    # clf = svm.SVC(kernel='linear', C=1,gamma='auto') # Linear Kernel
    # clf = svm.SVC(kernel='rbf', C=1000, gamma='scale') # Linear Kernel
    c_size = 1000
    clf = svm.SVC(kernel='rbf', C=c_size, gamma='auto') # Linear Kernel
    #Train the model using the training sets

    clf.fit(X, y)
    # y_pred = clf.predict(X)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 6),
                        np.linspace(y_min, y_max, 6))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    num_data = data.replace({"class": {'1': 1, '2': 2, '3': 3}})
    svs = clf.support_vectors_
    fig = go.Figure(data =
         go.Contour(
            z=Z,
            contours_coloring='lines',
            dx=2,
            x0=-5,
            dy=2,
            y0=-5,
            showscale=False,
            colorscale=[[0, 'red'], [0.5, 'green'], [1, 'blue']],
            ))

    fig.add_trace(go.Scatter(x=svs[:, 0], y=svs[:, 1],
                    mode='markers',
                    name='support_vectors',
                    opacity=0.5,
                    marker=dict(
                    size=10,
                    color='LightSkyBlue',
                        line=dict(
                            color='MediumPurple',
                            width=20
                    ))))
    fig.add_trace(go.Scatter(x=num_data['pca1'], y=num_data['pca2'],
                    mode='markers',
                    name='wine datas',
                    marker=dict(
                    size=10,
                    color=num_data['class'],
                    colorscale=[[0, 'red'], [0.5, 'green'], [1, 'blue']],)))

    fig.update_traces()
    fig.update_layout(
        title="Support Vector Machine C="+str(c_size),
        xaxis_title="pca1",
        yaxis_title="pca2",
        font=dict(
        # family="Courier New, monospace",
        size=32,
        # color="RebeccaPurple"
        ),
        yaxis_range=[-5,5],
        xaxis_range=[-5,5],
    )

    fig.show()

def print_pca(data):
    # https://laid-back-scientist.com/en/pca-imple
    X = data.iloc[:,1:].values  # Get non-class columns
    y = data.iloc[:, 0].values  # Get class columns
    # print(y)
    # print(X)
    # standardization
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    # Create variance-covariance matrix
    cov_mat = np.cov(X_std.T)
    # Create eigenvalues and eigenvectors of variance-covariance matrix
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    # Create pairs of eigenvalues and eigenvectors
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
    # Sort the above pairs in order of increasing eigenvalue
    eigen_pairs.sort(key=lambda k: k[0], reverse=True)

    w1 = eigen_pairs[0][1]  # Eigenvector corresponding to the first principal component
    w2 = eigen_pairs[1][1]  # Eigenvector corresponding to the second principal component
    # Projection Matrix Creation
    W = np.stack([w1, w2], axis=1)
    # Dimensional compression (13D -> 2D)
    X_pca = X_std.dot(W)

    pca_data = pd.DataFrame(X_pca, columns = ["pca1","pca2"])
    pca_data["class"] = y
    pca_data = pca_data.replace({"class": {1: '1', 2: '2', 3: '3'}})
    fig = px.scatter(pca_data, x="pca1", y="pca2", color="class")
    fig.update_traces(marker_size=10)
    fig.update_layout(font=dict(
        # family="Courier New, monospace",
        size=32,
        # color="RebeccaPurple"
    ))
    # fig.show()

    return pca_data

def normalize_data(data):
    normalized_data=(data-data.min())/(data.max()-data.min())
    return normalized_data

debug = True
# debug = 0

wine_data = pd.read_csv('./class2/wine.data')
attributes = [
    'class', 
    'Alcohol', 
    'Malic acid', 
    'Ash', 
    'Alcalinity of ash', 
    'Magnesium', 
    'Total phenols', 
    'Flavanoids', 
    'Nonflavanoid phenols', 
    'Proanthocyanins', 
    'Color intensity',
    'Hue', 
    'OD280/OD315 of diluted wines', 
    'Proline'
]

wine_data.columns = attributes
# print(len(wineData))
# print(len(wineData.loc[wineData['class'] == 1]))
# print(wineData.loc[0])

# print_parallel(wineData)

nor_data = normalize_data(wine_data)

wine_pca_data = print_pca(wine_data)

# wineData = normalized_data

if debug != True:
    print_svm_linear(wine_pca_data)

    print_svm_rbf(wine_pca_data)

print(wine_pca_data)
wine_pca_data.to_csv('wine_pca_data.data', sep='\t', encoding='ASCII')
