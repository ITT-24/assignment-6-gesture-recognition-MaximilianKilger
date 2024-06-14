# $1 gesture recognizer
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample

import sys

import matplotlib.pyplot as plt

# XML parser
import xml.etree.ElementTree as ET

NUM_POINTS = 64

# read Data
data = []
data_X = []
data_y = []

for root, subdirs, files in os.walk('xml_logs'):
    if 'ipynb_checkpoint' in root:
        continue
    
    if len(files) > 0:
        for f in tqdm(files):
            if '.xml' in f:
                fname = f.split('.')[0]
                label = fname[:-2]
                
                xml_root = ET.parse(f'{root}/{f}').getroot()
                
                points = []
                for element in xml_root.findall('Point'):
                    x = element.get('X')
                    y = element.get('Y')
                    points.append([x, y])
                    
                points = np.array(points, dtype=float)
                
                scaler = StandardScaler()
                points = scaler.fit_transform(points)
                
                resampled = resample(points, NUM_POINTS)
                
                data_X.append(resampled)
                data_y.append(label)
    
data_X = np.array(data_X)
data_y = np.array(data_y)

# train templates
templates = {}
for label in np.unique(data_y):
    examples = data_X[data_y == label]
    template = np.median(examples, axis=0)
    templates[label] = template
    print(label)
    print(template.shape)
    #plt.plot(template.T[0], template.T[1]*-1)
    #plt.show()

def shift_to_centroid (points):
    centroid = np.mean(points, axis=0)
    points -= centroid
    return points, centroid

# https://stackoverflow.com/a/26757297
def cart2pol(point):
    x = point[0]
    y = point[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(np.array((rho, phi)))

def pol2cart(point):
    rho = point[0]
    phi = point[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(np.array((x, y)))

def rotate_to_zero(points):
    points_polar = np.apply_along_axis(cart2pol, 1, points)
    phi_0 = points_polar[0][1]
    points_polar[:,1] = points_polar[:,1] - phi_0
    return np.apply_along_axis(pol2cart, 1, points_polar)

def scale_to(points, new_size:tuple[int,int]):
    max_x = np.max(points[:,0])
    min_x = np.min(points[:,0])

    max_y = np.max(points[:,1])
    min_y = np.min(points[:,1])

    points[:,0] = (points[:,0] - min_x) / (max_x - min_x) * new_size[0]
    points[:,1] = (points[:,1] - min_y) / (max_y - min_y) * new_size[1]

    return points

def classify (gesture, templates):
    best_score = 9999999999999999999999999
    for label in templates.keys():
        template = templates[label]

        template, temp_centroid = shift_to_centroid(template)
        gesture, gest_centroid = shift_to_centroid(gesture)
        
        template = rotate_to_zero(template)
        gesture = rotate_to_zero(gesture)

        print(f"SHAPE: {template.shape}")

        size = (100,100)
        template = scale_to(template, size)
        gesture = scale_to(gesture, size)

for gesture in data_X:
    classify(gesture, templates)