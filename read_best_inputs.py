import pygetwindow as gw
import geo_dash_helpers as gd
import time

actions = []

# Open the file in read mode
with open("best_inputs.txt", "r") as file:
    # Read each line from the file and add it to the list
    for line in file:
        actions.append(float(line.strip()))  # Use strip() to remove newline characters

window = gw.getWindowsWithTitle('Geometry Dash')[0]
window.activate()
gd.jump()

while not gd.game_over():
    pass

while True:
    while gd.game_over():
        pass

    episode_start_time = time.time()
    for t in actions:
        while True:
            if abs(t - (time.time() - episode_start_time)) < .001:
                window = gw.getWindowsWithTitle('Geometry Dash')[0]
                window.activate()
                gd.jump()
                break
        if gd.game_over:
            break
