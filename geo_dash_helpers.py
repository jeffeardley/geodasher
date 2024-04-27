import keyboard
import pygetwindow as gw
import numpy as np
import cv2
import mss
import time

# 0% position {"top": 40, "left": 3875, "width": 90, "height": 40}
# end screen position {"top": 840, "left": 3750, "width": 200, "height": 200}
WIDTH = 1024
HEIGHT = WIDTH
SCREEN_POS = 'left'
if SCREEN_POS == 'left':
    LEFT = -1250
else:
    LEFT = 3230
TOP = 40

def capture_game_screen(ai_view=False):
    with mss.mss() as sct:
        # Set the capture region (coordinates and dimensions)
        #play_area = {"top": 60, "left": 3230, "width": 1200, "height": 950}
        play_area = {"top": TOP, "left": LEFT, "width": WIDTH, "height": HEIGHT}

        # Capture the screen region
        frame = sct.grab(play_area)

        #Convert the screen capture to a format that OpenCV can work with
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(np.array(frame), 127, 255, cv2.THRESH_BINARY)
        print(frame[1023])

        if ai_view:
            cv2.imshow('ai view', frame)
            cv2.waitKey(0)

        # normalize frame to values between 0 - 1
        # frame = cv2.resize(frame, (512, 512))
        frame = frame.astype(np.float32)
        frame = frame / 255.0
        return frame
    
def create_blocks(frame, root_of_blocks): #Takes the number of desired blocks along the width and height and seperates the frame into blocks
    width_of_block = len(frame[0]) / root_of_blocks
    width_of_block = int(width_of_block)
    blocks = []
    
    x_index = 0
    y_index = 0
    while y_index < len(frame):
        row = []
        for i in range(root_of_blocks):
            block = []
            for n in range(width_of_block):
                block.extend(frame[y_index][x_index + (width_of_block*i): x_index + (width_of_block*(i+1))])
                y_index += 1
            row.append(block)
            y_index -= (width_of_block)
        blocks.append(row)
        y_index += width_of_block

    return blocks


def calc_percent_on(block): # takes an array of 1s and 0s and calculates the percentage of 1s
    on = 0
    for pixel in block:
        if pixel == 1:
            on += 1
    return on / len(block)


def create_nn_input(frame):
    blocks = create_blocks(frame, 32)
    percent_on_per_block = []
    for row in blocks:
        row_percents = []
        for block in row:
            row_percents.append(calc_percent_on(block))
        percent_on_per_block.append(row_percents)
    return percent_on_per_block


def game_over(ai_view=False):
    with mss.mss() as sct:
        zero_img = cv2.imread("zeropercent.png", cv2.IMREAD_GRAYSCALE)

        # Set the capture region (coordinates and dimensions)
        zero_percent = {"top": TOP, "left": LEFT + 645, "width": 90, "height": 40}
        frame = sct.grab(zero_percent)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(np.array(frame), 127, 255, cv2.THRESH_BINARY)

        # Displays if true, for debugging
        if ai_view:
            cv2.imshow('ai view', frame)
            cv2.waitKey(0)

        # Calculates the similarity between the reference image, and the given image
        mse = ((frame - zero_img) ** 2).mean()
        if mse == 0:
            return True
        else:
            return False

def game_won():
    with mss.mss() as sct:
        end_screen_img = cv2.imread("endscreen.png", cv2.IMREAD_GRAYSCALE)
        # Set the capture region (coordinates and dimensions)
        end_screen = {"top": TOP + 800, "left": LEFT + 520, "width": 200, "height": 200}
        frame = sct.grab(end_screen)
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        _, frame = cv2.threshold(np.array(frame), 250, 255, cv2.THRESH_BINARY)

        mse = ((frame - end_screen_img) ** 2).mean()
        if mse == 0:
            return True
        else:
            return False
        
def jump():
    keyboard.press("space")
    keyboard.release("space")

def hold_jump():
    keyboard.release("space")

def stop_jump():
    keyboard.release("space")

def pause():
    keyboard.press("esc")
    keyboard.release("esc")

def unpause():
    jump()

def calculate_reward(last_reward_time):
    current_reward = time.time() - last_reward_time
    return current_reward

# while True:
#     if game_over():
#         print('dead')
#     else:
#         print('alive')

# while True:
#     capture_game_screen(True)
def create_test_frame():
    test_frame = []
    for i in range(1024):
        row = []
        for n in range(1024):
            row.append(n + i)
        test_frame.append(row)
    return test_frame

# test_frame = create_test_frame()

# test_frame = [[0, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 1, 1],
#               [0, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 1, 1],
#               [0, 0, 0, 0, 0, 0],
#               [1, 1, 1, 1, 1, 1]]
# start = time.time()
# print(len(create_nn_input()))
# print(time.time() - start)