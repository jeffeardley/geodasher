import pygetwindow as gw
import save_inputs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import geo_dash_helpers as gd
import time

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6400, 512)  # Adjust input dimensions based on output of last conv layer
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create a Q-network instance
model = QNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epsilon = 0.8  # Exploration rate
gamma = 0.5  # Discount factor

window = gw.getWindowsWithTitle('Geometry Dash')[0]
window.activate()
gd.jump()

highest_reward = 0
episode = 0

inputs = save_inputs.Save_Inputs()

# Training loop
while True:
    window = gw.getWindowsWithTitle('Geometry Dash')[0]
    window.activate()
    episode += 1
    epsilon *= 0.95
    if epsilon <= 0.1:
        epsilon = 0.1
    print('epsilon', epsilon)

    frame = gd.capture_game_screen()
    gd.pause()
    state = gd.create_nn_input(frame)  # Capture game screen as input image
    state = np.expand_dims(state, axis=-1)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    total_reward = 0
    round_pause_time = 0
    episode_pause_time = 0
    last_reward_time = time.time()

    model.eval()
    with torch.no_grad():
        q_values = model(state)
    q_values[0][0:1] = 0
    gd.unpause()
    time.sleep(.05)

    while gd.game_over():
        time.sleep(.1)

    inputs.new_episode()
    gd.pause()

    while not gd.game_over() and not gd.game_won():
        reward = 0

        if np.random.rand() < epsilon:
            action = np.random.choice([0, 1, 2])  # Exploration
            print("explore")
        else:
            model.eval()
            with torch.no_grad():
                q_values = model(state)
            print(q_values)
            action = torch.argmax(q_values[0]).item()  # Exploitation
            print("exploit")

        gd.unpause()
        time.sleep(.1)
        if action == 1:
            gd.jump()
            inputs.add_jump(episode_pause_time)
            print(q_values[0], "Jump")
        if action == 2:
            gd.hold_jump()
            print(q_values[0], "Hold jump")
        else:
            gd.stop_jump()
            print(q_values[0], "Don't jump")
            
        frame = gd.capture_game_screen()
        gd.pause()
        next_state = gd.create_nn_input(frame)
        next_state = np.expand_dims(next_state, axis=-1)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        reward += gd.calculate_reward(last_reward_time - round_pause_time)  # Use your reward function
        
        last_reward_time = time.time()

        before_pause = time.time()

        # Update Q-values
        model.eval()
        with torch.no_grad():
            q_values_next = model(next_state)
        target = reward + (gamma * torch.max(q_values_next).item())
        q_values[0][action] = target
        
        # Train the model
        model.train()
        optimizer.zero_grad()
        loss = criterion(q_values, model(state))
        loss.backward()
        optimizer.step()

        round_pause_time = time.time() - before_pause
        episode_pause_time += round_pause_time
        
        state = next_state
        total_reward += reward

    if total_reward > highest_reward:
        highest_reward = total_reward
        inputs.new_best()

    print("Episode:", episode, "Total Reward:", total_reward, "Highest Reward:", highest_reward)

