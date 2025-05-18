import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import kart
import time
import copy
import math
import os

# Number of simultaneously training agents
count = 40

# DQN Neural Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done
    
    def __len__(self):
        return len(self.buffer)

# Environment Wrapper
class KartEnv:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.cars = []
        self.max_checkpoint_idx = self._get_max_checkpoint_index()
        self.screen = pygame.display.set_mode((kart.MAP_WIDTH, kart.MAP_HEIGHT))
        pygame.display.set_caption("Karting Training")
        self.font = pygame.font.SysFont('arial', 20)
        self.global_max_checkpoints = 0
        self.reset()

    def _get_surface_color(self, x, y):
        if 0 <= x < kart.MAP_WIDTH and 0 <= y < kart.MAP_HEIGHT:
            return kart.map_image.get_at((int(x), int(y)))[:3]
        return kart.COLOR_WALL

    def _is_checkpoint(self, x, y):
        color = self._get_surface_color(x, y)
        return color[0] == 150 and color[2] == 150

    def _get_checkpoint_index(self, x, y):
        if self._is_checkpoint(x, y):
            return kart.map_image.get_at((int(x), int(y)))[1]
        return -1

    def _get_max_checkpoint_index(self):
        max_idx = 0
        for y in range(kart.MAP_HEIGHT):
            for x in range(kart.MAP_WIDTH):
                if self._is_checkpoint(x, y):
                    idx = self._get_checkpoint_index(x, y)
                    max_idx = max(max_idx, idx + 1)
        return max_idx or 1

    def reset(self):
        self.cars = [kart.Car(*kart.find_start_position(), 0, render_enabled=True) for _ in range(self.num_agents)]
        for car in self.cars:
            car.next_checkpoint = 0
            car.total_reward = 0
            car.checkpoints_passed = 0
        return [self._get_state(car) for car in self.cars]

    def _get_state(self, car):
        return np.array([
            car.x / kart.MAP_WIDTH,
            car.y / kart.MAP_HEIGHT,
            car.speed / kart.MAX_SPEED,
            car.angle / (2 * math.pi),
            car.steering_angle / (math.pi / 6),
            car.velocity_x / kart.MAX_SPEED,
            car.velocity_y / kart.MAX_SPEED,
            car.next_checkpoint / self.max_checkpoint_idx
        ])

    def step(self, actions):
        rewards = []
        next_states = []
        done = False
        max_checkpoints = max(car.checkpoints_passed for car in self.cars)

        for car, action in zip(self.cars, actions):
            if car.checkpoints_passed < max_checkpoints - 2:
                car.reset()
                rewards.append(-1.0)
                next_states.append(self._get_state(car))
                continue

            keys = {k: False for k in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_LALT]}
            if action in [1, 5, 6]:
                keys[pygame.K_UP] = True
            if action in [2, 7, 8]:
                keys[pygame.K_DOWN] = True
            if action in [3, 5, 7]:
                keys[pygame.K_LEFT] = True
            if action in [4, 6, 8]:
                keys[pygame.K_RIGHT] = True

            prev_x, prev_y = car.x, car.y
            prev_checkpoint = car.next_checkpoint

            car.update(keys)
            reward = -0.01

            surface_color = self._get_surface_color(car.x, car.y)
            if surface_color == kart.COLOR_WALL:
                reward -= 0.5
            elif surface_color == kart.COLOR_SAND:
                reward -= 0.1
            elif self._is_checkpoint(car.x, car.y):
                checkpoint_idx = self._get_checkpoint_index(car.x, car.y)
                if checkpoint_idx == car.next_checkpoint:
                    reward += 10.0
                    car.next_checkpoint = (checkpoint_idx + 1) % self.max_checkpoint_idx
                    car.checkpoints_passed += 1
                    if checkpoint_idx == 0 and car.next_checkpoint == 1:
                        reward += 50.0

            car.total_reward += reward
            rewards.append(reward)
            next_states.append(self._get_state(car))

        self.global_max_checkpoints = max(self.global_max_checkpoints, max_checkpoints)
        return next_states, rewards, done

    def render(self):
        self.screen.fill((0, 0, 0))

        faded_surface = pygame.Surface((kart.MAP_WIDTH, kart.MAP_HEIGHT), pygame.SRCALPHA)
        faded_surface.blit(kart.trail_surface, (0, 0))
        faded_surface.set_alpha(int(255 * kart.TRAIL_FADE_RATE))
        kart.trail_surface.fill((0, 0, 0, 0))
        kart.trail_surface.blit(faded_surface, (0, 0))

        self.screen.blit(kart.map_image, (0, 0))
        self.screen.blit(kart.trail_surface, (0, 0))

        for car in self.cars:
            car.draw()

        # Render statistics in green
        avg_checkpoints = sum(car.checkpoints_passed for car in self.cars) / self.num_agents
        max_checkpoints = max(car.checkpoints_passed for car in self.cars)
        max_reward = max(car.total_reward for car in self.cars)
        stats_text = [
            f"Global Max Checkpoints: {self.global_max_checkpoints}",
            f"Episode Checkpoints: {avg_checkpoints:.2f} ({max_checkpoints})",
            f"Max Reward: {max_reward:.2f}",
            f"Episode: {self.current_episode}"
        ]
        for i, text in enumerate(stats_text):
            rendered_text = self.font.render(text, True, (0, 255, 0))
            self.screen.blit(rendered_text, (10, 10 + i * 20))

        pygame.display.flip()

# Training Parameters
STATE_SIZE = 8
ACTION_SIZE = 9
EPISODE_DURATION = 20
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000
MODEL_PATH = "best_dqn_model.pth"

def select_action(state, policy_net, epsilon, device):
    if random.random() > epsilon:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.argmax().item()
    else:
        return random.randrange(ACTION_SIZE)

def optimize_model(policy_net, target_net, memory, optimizer, device):
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
    
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)
    
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    
    loss = nn.MSELoss()(q_values, expected_q_values)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    pygame.init()
    
    env = KartEnv(count)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
    target_net = DQN(STATE_SIZE, ACTION_SIZE).to(device)
    
    if os.path.exists(MODEL_PATH):
        print(f"Model save file found: {MODEL_PATH}")
        try:
            policy_net.load_state_dict(torch.load(MODEL_PATH))
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model from {MODEL_PATH}: {e}")
    else:
        print("No model save file found")
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(MEMORY_CAPACITY)
    epsilon = EPSILON
    
    episode = 0
    frame_count = 0
    
    while True:
        episode += 1
        env.current_episode = episode
        states = env.reset()
        total_rewards = [0] * count
        start_time = time.time()
        steps = 0
        
        while time.time() - start_time < EPISODE_DURATION:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            actions = [select_action(state, policy_net, epsilon, device) for state in states]
            next_states, rewards, done = env.step(actions)
            
            for i in range(count):
                memory.push(states[i], actions[i], rewards[i], next_states[i], done)
                total_rewards[i] += rewards[i]
                states[i] = next_states[i]
            
            optimize_model(policy_net, target_net, memory, optimizer, device)
            
            frame_count += 1
            steps += 1
            if frame_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            env.render()
        
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        best_idx = np.argmax(total_rewards)
        print(f"Episode {episode}, Best Reward: {total_rewards[best_idx]:.2f}, Epsilon: {epsilon:.3f}, Steps: {steps}")
        
        torch.save(policy_net.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main()