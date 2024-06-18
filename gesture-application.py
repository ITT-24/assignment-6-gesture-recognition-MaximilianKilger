# application for task 3


# This is a media controller!
# Use Ctrl-Alt-W to bring the window to the foreground.

import numpy as np
import os
from scipy.signal import resample

import sys, math

import pyglet

from pynput.keyboard import Key, Listener, GlobalHotKeys, Controller, HotKey
#from pynput import keyboard
import json

from recognizer import classify

TEMPLATES_FILEPATH = "templates_media_controller.json"

ICON_FILEPATHS = {
    "next":"icons/fastforward.png",
    "previous":"icons/reverse2.png",
    "vol_up": "icons/volume_up.png",
    "vol_down": "icons/volume_down.png",
    "play": "icons/play-pause.png"
}

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

action_icons = {}
for action in ICON_FILEPATHS.keys():
    action_icons[action] = pyglet.image.load(ICON_FILEPATHS[action])

icon_sprite = pyglet.sprite.Sprite(action_icons["play"], 0,0)
icon_sprite.opacity = 0

icon_base_size = 50
icon_base_opacity = 128
icon_growth_rate = 3
icon_fade_rate = 2

# keeps track of the number of times volume was increased using this application to protect the correctors' eardrums
MAXIMUM_ALLOWED_VOLUME_INCREASE = 50
internal_volume_counter = 0


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


#listener =  GlobalHotKeys({
#    '<ctrl>+<alt>+w': on_window_toggle})
#listener.start()

kcontrol = Controller()



def on_activate():
    on_window_toggle()

def for_canonical(f):
    return lambda k: f(l.canonical(k))

hotkey = HotKey(
    HotKey.parse('<ctrl>+<alt>+w'),
    on_activate)
l = Listener(
        on_press=for_canonical(hotkey.press),
        on_release=for_canonical(hotkey.release))
l.start()

def parse_gesture(gesture):
    global internal_volume_counter
    if gesture == "caret":
        if internal_volume_counter < MAXIMUM_ALLOWED_VOLUME_INCREASE:
            kcontrol.press(Key.media_volume_up)
            icon_sprite.image = action_icons["vol_up"]
            internal_volume_counter += 1

    elif gesture == "v":
        kcontrol.press(Key.media_volume_down)
        icon_sprite.image = action_icons["vol_down"]
        internal_volume_counter -= 1

    elif gesture == "triangle" or gesture == "rectangle":
        kcontrol.press(Key.media_play_pause)
        icon_sprite.image = action_icons["play"]

    elif gesture == "check":
        kcontrol.press(Key.media_next)
        icon_sprite.image = action_icons["next"]

    elif gesture == "circle":
        kcontrol.press(Key.media_previous)
        icon_sprite.image = action_icons["previous"]

def animate_icon(dt=0):
    if icon_sprite.opacity > 0:
        icon_sprite.opacity = max(0, icon_sprite.opacity - icon_fade_rate)
        icon_sprite.width = icon_sprite.width + icon_growth_rate
        icon_sprite.height = icon_sprite.height + icon_growth_rate
        
        # recenter
        icon_sprite.x = (WINDOW_WIDTH - icon_sprite.width) / 2
        icon_sprite.y = (WINDOW_HEIGHT - icon_sprite.height) / 2
        
animation_interval = 0.2
pyglet.clock.schedule_interval(animate_icon,animation_interval)

@window.event
def on_draw():
    window.clear()
    background.draw()
    linebatch.draw()
    icon_sprite.draw()
    animate_icon()

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

            icon_sprite.opacity = icon_base_opacity
            icon_sprite.width = icon_base_size
            icon_sprite.height = icon_base_size
        
            # recenter
            icon_sprite.x = (WINDOW_WIDTH - icon_sprite.width) / 2
            icon_sprite.y = (WINDOW_HEIGHT - icon_sprite.height) / 2




pyglet.app.run()