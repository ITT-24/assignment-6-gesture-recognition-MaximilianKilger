# $1 gesture recognizer
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import sys, math

import matplotlib.pyplot as plt

# XML parser
import xml.etree.ElementTree as ET

import pyglet

import json

import time

NUM_POINTS = 64




#returns the euclidian distance between points (x1, y1) and (x2, y2)
def euclidian(pos1:np.ndarray, pos2:np.ndarray):
    return math.sqrt( (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

# single steps of preprocessing pipeline

def shift_to_centroid (points:np.ndarray):
    centroid = np.mean(points, axis=0)
    points -= centroid
    return points, centroid

# https://stackoverflow.com/a/26757297
def cart2pol(point:np.ndarray):
    x = point[0]
    y = point[1]
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(np.array((rho, phi)))

def pol2cart(point:np.ndarray):
    rho = point[0]
    phi = point[1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(np.array((x, y)))

def rotate_to_zero(points:np.ndarray):
    points_polar = np.apply_along_axis(cart2pol, 1, points)
    phi_0 = points_polar[0][1]
    points_polar[:,1] = points_polar[:,1] - phi_0
    return np.apply_along_axis(pol2cart, 1, points_polar)

def scale_to(points:np.ndarray, new_size:tuple[int,int]):
    max_x = np.max(points[:,0])
    min_x = np.min(points[:,0])

    max_y = np.max(points[:,1])
    min_y = np.min(points[:,1])

    points[:,0] = (points[:,0] - min_x) / (max_x - min_x) * new_size[0]
    points[:,1] = (points[:,1] - min_y) / (max_y - min_y) * new_size[1]

    return points

def preprocess(points:np.ndarray, num_points:int=NUM_POINTS, size_x:int=100, size_y:int=100):
    size = (size_x,size_y)
    points = resample(points, num_points)
    points, centroid = shift_to_centroid(points)
    points = rotate_to_zero(points)
    points = scale_to(points, size)
    return points

def calculate_score(gesture:np.ndarray, template:np.ndarray):
    score = 0
    for i in range(len(gesture)):
        pt_gesture = gesture[i]
        pt_template = template[i]
        distance = euclidian(pt_gesture, pt_template)
        score += distance
    return score

def train_templates(daataa_X:np.ndarray, daataa_y:np.ndarray):
    # train templates
    templates = {}
    for label in np.unique(daataa_y):
        examples = daataa_X[daataa_y == label]
        template = np.mean(examples, axis=0)
        templates[label] = template
        plt.plot(template.T[0], template.T[1]*-1)
        plt.show()  
    return templates

def classify (gesture:np.ndarray, templates:np.ndarray):
    best_score = 9999999999999999999999999
    best_label = None
    for label in templates.keys():
        template = templates[label]

        gesture = preprocess(gesture,NUM_POINTS)
        template = preprocess(template, NUM_POINTS)
        if len(gesture) == len(template):
            score = min(calculate_score(gesture, template), calculate_score(np.flip(gesture, axis=0), template))
            if score < best_score:
                best_score = score
                best_label = label
        else:
            print(f"Cannot compare gesture to {label}: Gesture has {len(gesture)} points, template has {len(template)} points")
    return best_label, best_score


if __name__ == "__main__":
    TEMPLATE_FILE_PATH = "templates.json"
    GESTURE_DIR_PATH = "test"
    
    TRAIN = "-T" in sys.argv[1:]
    SAVE_GESTURE_TEMPLATES = "-S" in sys.argv
    OPEN_APPLICATION = "-A" in sys.argv
    RECORD_GESTURES = "-R" in sys.argv
    

    templates = {}
    if TRAIN:
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


        train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.05, random_state=445)
        templates = train_templates(train_X, train_y)

        if SAVE_GESTURE_TEMPLATES:
            with open(TEMPLATE_FILE_PATH, "w") as f:
                templates_serializable = {}
                for label in templates.keys():
                    templates_serializable[label] = templates[label].tolist()
                json.dump(templates_serializable, f)
    else:
        templates = {}
        
        with open(TEMPLATE_FILE_PATH) as f:
            templates = json.load(f)
        for label in templates.keys():
            points = templates[label]
            print(len(points))
            templates[label] = np.asarray(points)


    WIDTH = 900
    HEIGHT = 600

    window = pyglet.window.Window(width=WIDTH, height=HEIGHT)

    drawn_shape:list[np.ndarray] = []
    lines:list[pyglet.shapes.Line] = []
    linebatch = pyglet.shapes.Batch()
    linecolor = (255,0,0)
    linewidth = 3

    background = pyglet.shapes.Rectangle(0,0,WIDTH,HEIGHT,(255,255,255))

    textcolor = (128, 0, 128, 255)
    prediction_label = pyglet.text.Label("", "Arial", 30, x=0,y=HEIGHT*0.9,width=WIDTH, height=HEIGHT/10, color=textcolor)

    min_points_for_prediction = 10

    is_mouse_down = False
    gestures_saved = []

    def add_point(x, y):
        pt = np.array([x,y])
        if len(drawn_shape) > 0:
            last_pt = drawn_shape[-1]
            line = pyglet.shapes.Line(x, y, last_pt[0], HEIGHT - last_pt[1], linewidth, linecolor, batch=linebatch)
            lines.append(line)
        if RECORD_GESTURES:
            drawn_shape.append(np.array((pt[0], HEIGHT-pt[1], time.time())))
        else:
            drawn_shape.append(np.array((pt[0], HEIGHT-pt[1])))
    
    def save_gestures_xml (gestures, gesture_name, directory):
        for i, gesture in enumerate(gestures):
            zero_string = "0"
            name = f"{gesture_name}{str(i+1) if i+1 >= 10 else ''.join([zero_string,str(i+1)])}"
            root = ET.Element("Gesture")
            attrib = {
                "Name": name
            }
            root.attrib = attrib

            for point in gesture:
                pt = ET.SubElement(root,"Point")
                pt.attrib = {
                    "X": str(int(point[0])),
                    "Y": str(int(point[1])),
                    "T": str(int(point[2]))
                }
            tree = ET.ElementTree(root)
            fname = os.path.join(GESTURE_DIR_PATH,f"{name}.xml")
            tree.write(fname,encoding="utf-8", xml_declaration=True)



    @window.event
    def on_draw():
        global prediction_label
        window.clear()
        background.draw()
        linebatch.draw()
        if len(drawn_shape) >= min_points_for_prediction:
            label, score = classify(drawn_shape, templates)

            prediction_label.text = f"{label} : {round(score,2)}"
        prediction_label.draw()

    @window.event
    def on_key_press(key, modifiers):
        if key == pyglet.window.key.Q:
            os._exit()
        elif key == pyglet.window.key.S and RECORD_GESTURES:
            gesture_name = "question_mark"
            save_gestures_xml(gestures_saved, gesture_name, GESTURE_DIR_PATH)
            pass # TODO: SAVE ALL RECORDED GESTURES
        elif key == pyglet.window.key.ENTER:
            drawn_shape_list = []
            for point in drawn_shape:
                drawn_shape_list.append(point.tolist())
            gestures_saved.append(drawn_shape_list)



    @window.event
    def on_mouse_press(x,y, button, modifiers):
        global linebatch
        global lines
        global is_mouse_down
        global drawn_shape
        if button == pyglet.window.mouse.LEFT:
            lines = []
            drawn_shape = []
            linebatch.invalidate()
            linebatch = pyglet.shapes.Batch()
            is_mouse_down = True
            add_point(x,y)

    @window.event
    def on_mouse_drag(x,y, dx, dy, button, modifiers):
        global is_mouse_down
        if button == pyglet.window.mouse.LEFT and is_mouse_down:
            #print("DRAG")
            add_point(x,y)

    @window.event
    def on_mouse_release(x,y, button, modifiers):
        global is_mouse_down
        if button == pyglet.window.mouse.LEFT:
            is_mouse_down = False
            #print("UP")
    if OPEN_APPLICATION:
        pyglet.app.run()

