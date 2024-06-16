# application for task 3


# This is a media controller!
# Use Ctrl-Alt-W to bring the window to the foreground.

import numpy as np
import os
from scipy.signal import resample

import sys, math

import pyglet

from pynput.keyboard import Key, Listener, GlobalHotKeys, Controller
#from pynput import keyboard
import json

from recognizer import classify

TEMPLATES_FILEPATH = "templates_media_controller.json"

WINDOW_WIDTH:int = 200
WINDOW_HEIGHT:int = 200

window = pyglet.window.Window(200,200)
window.set_caption("GESTURE MEDIA CONTROLLER")


with open(TEMPLATES_FILEPATH) as f:
    templates = json.load(f)


drawn_shape:list[np.ndarray] = []
lines:list[pyglet.shapes.Line] = []
linebatch = pyglet.shapes.Batch()
linecolor = (168, 40, 60)
linewidth = 3

background = pyglet.shapes.Rectangle(0,0,WINDOW_WIDTH, WINDOW_HEIGHT,(255,255,255))

min_points_for_prediction = 10

is_mouse_down = False
is_playing = False

def on_window_toggle():
    global window
    window.activate()
    window.activate()


def add_point(x, y):
    pt = np.array([x,y])
    if len(drawn_shape) > 0:
        last_pt = drawn_shape[-1]
        line = pyglet.shapes.Line(x, y, last_pt[0], WINDOW_HEIGHT - last_pt[1], linewidth, linecolor, batch=linebatch)
        lines.append(line)
    drawn_shape.append(np.array((pt[0], WINDOW_HEIGHT-pt[1])))


listener =  GlobalHotKeys({
    '<ctrl>+<alt>+w': on_window_toggle})
listener.start()

kcontrol = Controller()


def parse_gesture(gesture):
    if gesture == "caret":
        kcontrol.press(Key.media_volume_up)
    if gesture == "v":
        kcontrol.press(Key.media_volume_down)
    if gesture == "triangle" or gesture == "rectangle":
        kcontrol.press(Key.media_play_pause)
    if gesture == "check":
        kcontrol.press(Key.media_next)
    if gesture == "circle":
        kcontrol.press(Key.media_previous)

@window.event
def on_draw():
    window.clear()
    background.draw()
    linebatch.draw()

@window.event
def on_key_press(key, modifiers):
    if key == pyglet.window.key.Q:
        if modifiers & pyglet.window.key.MOD_CTRL:
            os._exit(0)


@window.event
def on_mouse_press(x,y,button, modifiers):
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
        add_point(x,y)

@window.event
def on_mouse_release(x,y, button, modifiers):
    global is_mouse_down
    if button == pyglet.window.mouse.LEFT:
        is_mouse_down = False
        
        if len(drawn_shape) >= min_points_for_prediction:
            label, score = classify(drawn_shape, templates)
            print(label)
            parse_gesture(label)




pyglet.app.run()