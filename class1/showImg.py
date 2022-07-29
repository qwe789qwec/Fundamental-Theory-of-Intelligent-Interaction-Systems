import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
import math
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def normalize_data(data):
    normalized_data=(data-data.min())/(data.max()-data.min())
    return normalized_data

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
    fig.show()

def print_tsne(data):
    X = data.iloc[:,1:].values  # Get non-class columns
    y = data.iloc[:, 0].values  # Get class columns
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results1 = tsne.fit_transform(X)
    tsne_data = pd.DataFrame(tsne_results1, columns = ["tsne1","tsne2"])
    tsne_data["class"] = y
    tsne_data = tsne_data.replace({"class": {0.0: '1', 0.5: '2', 1: '3'}})
    fig = px.scatter(tsne_data, x="tsne2", y="tsne1", color="class")
    fig.update_traces(marker_size=10)
    fig.update_layout(font=dict(
        # family="Courier New, monospace",
        size=32,
        # color="RebeccaPurple"
    ))
    fig.show()

def print_parallel(data):
    fig = px.parallel_coordinates(
        data, 
        color="class", 
        labels=data.columns,
        color_continuous_scale=px.colors.diverging.Tealrose,
        color_continuous_midpoint=2)

    fig.show()

def print_starpolar(data):
    data = data.drop(['class'], axis=1)
    theta_att = [
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
    fig = px.line_polar(
        data.loc[1], 
        r = pd.Series(data.loc[1].values), 
        theta = theta_att, 
        line_close = True,
        )

    fig.update_polars(
        # angularaxis_showgrid = False,   
        # angularaxis_showline = False, 
        # angularaxis_showticklabels = False,
        # radialaxis_showgrid = False,
        # radialaxis_showline = False,
        radialaxis_showticklabels = False
        )
    fig.show()

def print_grid(data):
    fig = sp.make_subplots(rows=3, cols=2, specs=[[{'type': 'polar'}]*2]*3)

    star = go.Scatterpolar(
        r = pd.Series(data.loc[1].values),
        theta = data.columns,
        line_color = 'deepskyblue',
        )

    fig.add_trace(star, 1, 1)
    fig.add_trace(star, 1, 2)
    fig.add_trace(star, 2, 1)
    fig.add_trace(star, 2, 2)
    
    polars_parameters = dict(
            angularaxis_showgrid = False,   
            angularaxis_showline = False, 
            angularaxis_showticklabels = False,
            radialaxis_showgrid = False,
            radialaxis_showline = False,
            radialaxis_showticklabels = False
        )
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar = polars_parameters, 
        polar2 = polars_parameters, 
        polar3 = polars_parameters, 
        polar4 = polars_parameters
    )

    fig.show()

def print_starpolar_grid(data):
    X = data.iloc[:,1:].values  # Get non-class columns
    y = data.iloc[:, 0].values  # Get class columns
    colatt = [
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
    # print(X[1,:])
    data = pd.DataFrame(X, columns = colatt)
    length = len(data)
    figCols = 20
    figRows = math.ceil(length/figCols)
    fig = sp.make_subplots(
        rows = figRows, 
        cols = figCols, 
        specs = [[{'type': 'polar'}]*figCols]*figRows, 
        subplot_titles = range(1, length+1, 1))

    for index, row in data.iterrows():
        star = go.Scatterpolar(
            r = pd.Series(row.values),
            theta = data.columns,
            mode = 'lines',
            line_color = 'deepskyblue',
            )
        fig.add_trace(star, (int(index)//figCols)+1, (int(index)%figCols)+1)
        fig.update_traces(fill='toself')
    
    polars_parameters = dict(
            angularaxis_showgrid = False,   
            angularaxis_showline = False, 
            angularaxis_showticklabels = False,
            radialaxis_showgrid = False,
            radialaxis_showline = False,
            radialaxis_showticklabels = False, 
        )

    fig.update_layout(
        showlegend = False,
        polar    = polars_parameters,
        polar1	 = polars_parameters,
        polar2	 = polars_parameters,
        polar3	 = polars_parameters,
        polar4	 = polars_parameters,
        polar5	 = polars_parameters,
        polar6	 = polars_parameters,
        polar7	 = polars_parameters,
        polar8	 = polars_parameters,
        polar9	 = polars_parameters,
        polar10	 = polars_parameters,
        polar11	 = polars_parameters,
        polar12	 = polars_parameters,
        polar13	 = polars_parameters,
        polar14	 = polars_parameters,
        polar15	 = polars_parameters,
        polar16	 = polars_parameters,
        polar17	 = polars_parameters,
        polar18	 = polars_parameters,
        polar19	 = polars_parameters,
        polar20	 = polars_parameters,
        polar21	 = polars_parameters,
        polar22	 = polars_parameters,
        polar23	 = polars_parameters,
        polar24	 = polars_parameters,
        polar25	 = polars_parameters,
        polar26	 = polars_parameters,
        polar27	 = polars_parameters,
        polar28	 = polars_parameters,
        polar29	 = polars_parameters,
        polar30	 = polars_parameters,
        polar31	 = polars_parameters,
        polar32	 = polars_parameters,
        polar33	 = polars_parameters,
        polar34	 = polars_parameters,
        polar35	 = polars_parameters,
        polar36	 = polars_parameters,
        polar37	 = polars_parameters,
        polar38	 = polars_parameters,
        polar39	 = polars_parameters,
        polar40	 = polars_parameters,
        polar41	 = polars_parameters,
        polar42	 = polars_parameters,
        polar43	 = polars_parameters,
        polar44	 = polars_parameters,
        polar45	 = polars_parameters,
        polar46	 = polars_parameters,
        polar47	 = polars_parameters,
        polar48	 = polars_parameters,
        polar49	 = polars_parameters,
        polar50	 = polars_parameters,
        polar51	 = polars_parameters,
        polar52	 = polars_parameters,
        polar53	 = polars_parameters,
        polar54	 = polars_parameters,
        polar55	 = polars_parameters,
        polar56	 = polars_parameters,
        polar57	 = polars_parameters,
        polar58	 = polars_parameters,
        polar59	 = polars_parameters,
        polar60	 = polars_parameters,
        polar61	 = polars_parameters,
        polar62	 = polars_parameters,
        polar63	 = polars_parameters,
        polar64	 = polars_parameters,
        polar65	 = polars_parameters,
        polar66	 = polars_parameters,
        polar67	 = polars_parameters,
        polar68	 = polars_parameters,
        polar69	 = polars_parameters,
        polar70	 = polars_parameters,
        polar71	 = polars_parameters,
        polar72	 = polars_parameters,
        polar73	 = polars_parameters,
        polar74	 = polars_parameters,
        polar75	 = polars_parameters,
        polar76	 = polars_parameters,
        polar77	 = polars_parameters,
        polar78	 = polars_parameters,
        polar79	 = polars_parameters,
        polar80	 = polars_parameters,
        polar81	 = polars_parameters,
        polar82	 = polars_parameters,
        polar83	 = polars_parameters,
        polar84	 = polars_parameters,
        polar85	 = polars_parameters,
        polar86	 = polars_parameters,
        polar87	 = polars_parameters,
        polar88	 = polars_parameters,
        polar89	 = polars_parameters,
        polar90	 = polars_parameters,
        polar91	 = polars_parameters,
        polar92	 = polars_parameters,
        polar93	 = polars_parameters,
        polar94	 = polars_parameters,
        polar95	 = polars_parameters,
        polar96	 = polars_parameters,
        polar97	 = polars_parameters,
        polar98	 = polars_parameters,
        polar99	 = polars_parameters,
        polar100 = polars_parameters,
        polar101 = polars_parameters,
        polar102 = polars_parameters,
        polar103 = polars_parameters,
        polar104 = polars_parameters,
        polar105 = polars_parameters,
        polar106 = polars_parameters,
        polar107 = polars_parameters,
        polar108 = polars_parameters,
        polar109 = polars_parameters,
        polar110 = polars_parameters,
        polar111 = polars_parameters,
        polar112 = polars_parameters,
        polar113 = polars_parameters,
        polar114 = polars_parameters,
        polar115 = polars_parameters,
        polar116 = polars_parameters,
        polar117 = polars_parameters,
        polar118 = polars_parameters,
        polar119 = polars_parameters,
        polar120 = polars_parameters,
        polar121 = polars_parameters,
        polar122 = polars_parameters,
        polar123 = polars_parameters,
        polar124 = polars_parameters,
        polar125 = polars_parameters,
        polar126 = polars_parameters,
        polar127 = polars_parameters,
        polar128 = polars_parameters,
        polar129 = polars_parameters,
        polar130 = polars_parameters,
        polar131 = polars_parameters,
        polar132 = polars_parameters,
        polar133 = polars_parameters,
        polar134 = polars_parameters,
        polar135 = polars_parameters,
        polar136 = polars_parameters,
        polar137 = polars_parameters,
        polar138 = polars_parameters,
        polar139 = polars_parameters,
        polar140 = polars_parameters,
        polar141 = polars_parameters,
        polar142 = polars_parameters,
        polar143 = polars_parameters,
        polar144 = polars_parameters,
        polar145 = polars_parameters,
        polar146 = polars_parameters,
        polar147 = polars_parameters,
        polar148 = polars_parameters,
        polar149 = polars_parameters,
        polar150 = polars_parameters,
        polar151 = polars_parameters,
        polar152 = polars_parameters,
        polar153 = polars_parameters,
        polar154 = polars_parameters,
        polar155 = polars_parameters,
        polar156 = polars_parameters,
        polar157 = polars_parameters,
        polar158 = polars_parameters,
        polar159 = polars_parameters,
        polar160 = polars_parameters,
        polar161 = polars_parameters,
        polar162 = polars_parameters,
        polar163 = polars_parameters,
        polar164 = polars_parameters,
        polar165 = polars_parameters,
        polar166 = polars_parameters,
        polar167 = polars_parameters,
        polar168 = polars_parameters,
        polar169 = polars_parameters,
        polar170 = polars_parameters,
        polar171 = polars_parameters,
        polar172 = polars_parameters,
        polar173 = polars_parameters,
        polar174 = polars_parameters,
        polar175 = polars_parameters,
        polar176 = polars_parameters,
        polar177 = polars_parameters,
        polar178 = polars_parameters,
        polar179 = polars_parameters,
    )

    fig.show()

debug = True
# debug = 0

wineData = pd.read_csv('./wine.data')
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

wineData.columns = attributes
# print(len(wineData))
# print(len(wineData.loc[wineData['class'] == 1]))
# print(wineData.loc[0])

# print_parallel(wineData)

nor_data=normalize_data(wineData)

# wineData = normalized_data

if debug != True:
    print_parallel(wineData)

    print_starpolar(nor_data)

    print_grid(nor_data)
    
    print_starpolar_grid(nor_data)

print_pca(wineData)

print_tsne(nor_data)