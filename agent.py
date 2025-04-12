from datetime import datetime, timedelta
import itertools
import random
import matplotlib
from pettingzoo.atari import space_invaders_v2
import torch
import torch.nn as nn
import yaml
import os
import psutil
from cnn import CNN
from experience_replay import ReplayMemory
from ddqn import DQN
import torchvision.transforms as T

from matplotlib import pyplot as plt
import graph

transform = T.Compose([
    T.ToPILImage(),                      # np.array -> PIL
    T.Grayscale(),                       # RGB → 1 channel
    T.Resize((84, 84)),                  # resize
    T.ToTensor(),                        # shape → [C, H, W], normalize เป็น [0,1]
])

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')
g = graph.ShowGraph()
device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    def __init__(self, hyperparameters_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameters_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameters_set]

        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_params',{})
        self.enable_double_dqn = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        self.pretrained_model = hyperparameters.get('pretrained_model', None)

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        self.scaler = torch.amp.GradScaler(device=device)

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameters_set}.png')

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states,actions,new_states,rewards,terminations = zip(*mini_batch)
        # states = transform(states).to(device) 
        with torch.no_grad():
            states = torch.stack(states).to(device)
            
            actions = [torch.tensor([a], dtype=torch.long, device=device) for a in actions]
            actions = torch.stack(actions).to(device)   
            # print("actions shape:", actions.shape)
            # print("actions sample:", actions[:10])
            # new_states = transform(new_states).to(device) 
            new_states = torch.stack(new_states).to(device) 
            
            rewards = torch.stack(rewards).to(device)   
            terminations = torch.as_tensor(terminations).float().to(device)
        with torch.amp.autocast(device_type=device):
            # print("new_states shape:", new_states.shape)
            # print("target_dqn(new_states) shape:", target_dqn(new_states).shape)
            
            if self.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                # print("best_actions_from_policy shape:", best_actions_from_policy.shape)
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).gather(dim=1,index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                target_q = rewards + (1-terminations )* self.discount_factor_g  * target_dqn(new_states).max(dim=1)[0]


        current_q = policy_dqn(states).gather(dim=1,index=actions).squeeze()
        # print(f"current_q shape: {current_q.shape}")
        # print(f"target_q shape: {target_q.shape}")
        loss = self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    def log_message(self,str):
        print(str)
        with open(self.LOG_FILE, 'w') as log_file:
                log_file.write(str+ '\n')

    def run(self,is_training=True,render=False):
        start_time = datetime.now()
        last_graph_update_time = start_time
        self.log_message(f"DDQN Start time: {start_time}")

        env = space_invaders_v2.env(render_mode="human")

        num_states = 3
        num_actions = 6


        rewards_per_episode = []
        # List to keep track of epsilon decay
        epsilon_history = []
        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        # Track best reward
        best_reward = -9999999
        self.log_message(f"States: {num_states} | Actions: {num_actions}")

        policy_dqn = CNN(num_actions,enable_dueling_dqn=self.enable_dueling_dqn ).to(device)

        if is_training:
            if self.pretrained_model and os.path.exists(self.pretrained_model):
                print(f"Loading pretrained model from {self.pretrained_model}")
                policy_dqn.load_state_dict(torch.load(self.pretrained_model))
                print(f"Loaded pretrained model from {self.pretrained_model}")
            memory = ReplayMemory(self.replay_memory_size)
            epsilon = self.epsilon_init
            
            target_dqn = CNN(num_actions,enable_dueling_dqn=self.enable_dueling_dqn  ).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            epsilon_history = []
            best_reward = -9999999
        else:
            # Load the model
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            print(f"Loaded pretrained model from {self.MODEL_FILE}")
            policy_dqn.eval()





        for episode in itertools.count():
            env.reset(seed=42)
            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode
            terminated_agents = set()
            for agent in env.agent_iter():
                
                # print(f'Agent: {agent}')

                state, reward, terminated, truncated, info = env.last()
                state = transform(state).to(device)  # ได้ [1, 84, 84]
                # state = state.unsqueeze(0)           # เพิ่ม batch dim → [1, 1, 84, 84]
                state = torch.as_tensor(state, dtype=torch.float, device=device)
                if terminated or truncated:
                    # action = None
                    env.step(None)
                    continue
                else:
                    if is_training and random.random() < epsilon:
                        action = env.action_space(agent).sample()
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax().item()
                env.step(action)
                if agent not in env.agents:
                    break
                new_state,reward,terminated,truncated,info = env.last() 

                new_state = transform(new_state).to(device)  # ได้ [1, 84, 84]
                # new_state = new_state.unsqueeze(0)           # เพิ่ม batch dim → [1, 1, 84, 84]
                episode_reward += reward


                new_state = torch.as_tensor(new_state, dtype=torch.float, device=device)
                reward = torch.as_tensor(reward, dtype=torch.float, device=device)
                if is_training:
                    # Save experience into memory
                    memory.append((state.detach().cpu(), action, new_state.detach().cpu(), reward.detach().cpu(), terminated))

                    # Increment step counter
                    step_count+=1

                # Move to the next state
                state = new_state
                if len(terminated_agents) == len(env.possible_agents):
                    print("Both Death")
                    env.reset()
                    break
            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now()}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward


                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    g.save_graph(rewards_per_episode, epsilon_history,self.GRAPH_FILE)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory)>self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Decay epsilon
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0

            env.reset()


