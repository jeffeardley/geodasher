import time

class Save_Inputs:

    def __init__(self):
        self.start_time = time.time()
        self.current_inputs = []
        self.best_inputs = []

    def add_jump(self, episode_pause_time):
        self.current_inputs.append(time.time() - self.start_time - episode_pause_time)
    
    def new_episode(self):
        self.current_inputs = []
        self.start_time = time.time()

    def new_best(self):
        self.best_inputs = self.current_inputs
        with open("best_inputs.txt", "w") as file:
            for item in self.best_inputs:
                file.write(str(item) + "\n")