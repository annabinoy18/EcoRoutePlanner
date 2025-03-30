import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# ✅ Custom EV Environment (Uses Latitude & Longitude)
class EVRouteEnv:
    def __init__(self):
        self.charge_stations = [(10.0, 76.0), (10.5, 76.5)]  # Example stations (lat, long)
        self.destination = (11.0, 77.0)  # Destination coordinates
        self.state = [10.0, 76.0, 100, 200]  # [latitude, longitude, battery %, range]
        self.done = False

    def reset(self):
        """Resets the environment for a new episode."""
        self.state = [10.0, 76.0, 100, 200]  # Reset position, battery, range
        self.done = False
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        """Defines EV movement based on action."""
        if self.done:
            return self.state, 0, self.done  

        if action == 0:  # Take Route A
            self.state[0] += 0.1  # Latitude change
            self.state[1] += 0.1  # Longitude change
            self.state[2] -= 20  # Battery drain
            self.state[3] -= 30  # Range decreases
        elif action == 1:  # Take Route B
            self.state[0] += 0.05
            self.state[1] += 0.2
            self.state[2] -= 15
            self.state[3] -= 25
        elif action == 2:  # Charge (if at a charging station)
            if (self.state[0], self.state[1]) in self.charge_stations:
                self.state[2] = 100  # Full battery
                self.state[3] += 50  # Extra range
            else:
                self.state[2] -= 5  # Small penalty for stopping at wrong place

        # Check if destination is reached
        if (self.state[0], self.state[1]) == self.destination:
            self.done = True
            reward = 100
        elif self.state[2] <= 0:  # Battery dead
            self.done = True
            reward = -50
        else:
            reward = -1  # Small penalty for each step

        return np.array(self.state, dtype=np.float32), reward, self.done

# ✅ Define DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ✅ Define DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = []
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)  

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()  

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state))).item()
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))

# ✅ Train the DQN Model
env = EVRouteEnv()
state_size = len(env.reset())
action_size = 3  # (0: Route A, 1: Route B, 2: Charge)
agent = DQNAgent(state_size, action_size)

EPISODES = 500
for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    agent.replay()

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# ✅ Save the trained model
agent.save("dqn_ev_route.pth")
print("Training completed! Model saved as dqn_ev_route.pth")
