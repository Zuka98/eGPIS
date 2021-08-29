import numpy as np
import plotly.graph_objs as go
import plotly
import random
import math
import plotly.graph_objs as go
from skimage import measure

def scatter_points(points,color='blue', size=2, opacity=0.8):
    x,y,z = zip(*points)
    trace = go.Scatter3d(
        x= x,
        y= y,
        z= z,
        mode='markers',
        marker={
            'size': size,
            'opacity': opacity,
            'color': color
        }
    )

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(plot_figure)


def scatter_points_with_threshold(grid,outp,colors, threshold=0.01):
    x,y,z,c = [],[],[],[]

    for i in range(len(outp)):
        if abs(outp[i]) < threshold:
            x.append(grid[i][0])
            y.append(grid[i][1])
            z.append(grid[i][2])
            c.append(colors[i])
    
    scatter_points(np.dstack((x,y,z))[0], colors)


def downsample_uniform(X,Y,randvalue=100):      
    X_down, Y_down = [], []
    i = 0
    while i < len(X):
        coords = X[i]
        X_down.append([coords[0], coords[1], coords[2]])
        Y_down.append(Y[i])
        i += random.randrange(1,randvalue + 1)
    
    return np.asarray(X_down), np.asarray(Y_down)


def generate_grid(vertices, size=40):
    maxx = np.max(vertices[:, 0])
    maxy = np.max(vertices[:, 1])
    maxz = np.max(vertices[:, 2])

    minx = np.min(vertices[:, 0])
    miny = np.min(vertices[:, 1])
    minz = np.min(vertices[:, 2])

    x = np.linspace(minx, maxx, size)
    y = np.linspace(miny, maxy, size)
    z = np.linspace(minz, maxz, size)

    grid = []
    for i in range(len(x)):
        for j in range(len(y)):
            for w in range(len(z)):
                grid.append([x[i],y[j],z[w]])

    return np.asarray(grid)


def radial_augmentation(vertices, threshold=0.05):
    x,y,z = zip(*vertices)
    minx, miny, minz = min(x), min(y), min(z)
    maxx, maxy, maxz = max(x), max(y), max(z)
    center = [(minx + maxx)/2, (maxy + miny) / 2, (minz + maxz)/2]
    radius = math.sqrt((maxx - center[0]) ** 2 + (maxy - center[1] ** 2 + (maxz - center[2]) ** 2))
    
    sphere_points = []
    labels = []
    
    #Generate Points outside the Object
    for phi in range(0,360, 10):
        for theta in range(0,360, 20):
            p = center.copy()
            p[0] += radius * math.cos(theta) * math.sin(phi) + threshold
            p[1] += radius * math.sin(theta) * math.sin(phi) + threshold
            p[2] += radius * math.cos(phi) + threshold
            sphere_points.append(p)
            labels.append(1)

    #Generate Points inside the Object (Depending on the object's topology this is may not always be correct)
    radius = 0.05
    for phi in range(0,360, 60):
        for theta in range(0,360, 60):
            p = center.copy()
            p[0] += radius * math.cos(theta) * math.sin(phi)
            p[1] += radius * math.sin(theta) * math.sin(phi)
            p[2] += radius * math.cos(phi)
            sphere_points.append(p)
            labels.append(-1)
    
    return sphere_points, labels


def normal_augmentation(vertices, normals, threshold = 0.0025, layers=1):
    points = []
    labels = []
    for i in range(len(vertices)):
        #Find Unit Vector
        vertex = vertices[i]
        normal = normals[i]
        l2_norm = np.linalg.norm(normals[i])
        normal_unit = (normal / l2_norm) * threshold

        #Generate points inside and outside the object's surface
        for i in range(1,layers + 1):
            points.append(vertex + normal_unit * i)
            labels.append(1)
            points.append(vertex - normal_unit * i)
            labels.append(-1)
    
    return points, labels


def segment_critical_set(vertices,normals,interval,median_threshold = 0.6):
    critical_v, critical_n, arbitrary_v, arbitrary_n = [], [], [], []
    indices = np.argsort(vertices,axis=0)[:,0]
    vertices,normals = vertices[indices],normals[indices]

    #For each subsset of size 100
    for k in range(0,len(vertices),interval):
        # print(k)
        subset = normals[k:k+interval-1]
        ### Here we go and create dot products
        for i in range(0,len(subset)):
            dot_prods = []
            for j in range(0, len(subset)):
                dot_product = abs(np.dot(subset[i],subset[j]) / ((np.linalg.norm(subset[i]) * np.linalg.norm(subset[j]))))                              
                dot_prods.append(dot_product)
            
            if np.median(dot_prods) < median_threshold:
                critical_v.append(vertices[k+i])
                critical_n.append(normals[k+i])
            else:
                arbitrary_v.append(vertices[k+i])
                arbitrary_n.append(normals[k+i])
      
    return np.asarray(critical_v), np.asarray(critical_n), np.asarray(arbitrary_v),np.asarray(arbitrary_n)


def show_mesh(vertices, faces,colors=[]):
    x, y, z = zip(*vertices)
    xt, yt, zt = zip(*faces)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, 
                y=y, 
                z=z, 
                i = list(xt),
                j = list(yt),
                k = list(zt),
                colorscale='jet',
                intensity=colors,
            ),

    ])
    fig.show()


def get_scale(arr, grid_size):
    x,y,z = zip(*arr)
    
    return (
        (min(x) + max(x)) / grid_size,
        (min(y) + max(y)) / grid_size,
        (min(z) + max(z)) / grid_size
    )
