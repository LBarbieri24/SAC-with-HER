import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import deque
import random
import torch.optim as optim
from VAE import VariationalAutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae_model = VariationalAutoEncoder() 
vae_model.load_state_dict(torch.load('vae_model_new.pth'))
vae_model.eval()  # Set the model to evaluation mode

def encode_state(state):
    with torch.no_grad():
        return vae_model.encoder(state).squeeze(0)
    
def reparameterize(mean, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mean + eps * std


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 64)
        # self.fc4 = nn.Linear(64, 32)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
        # Inizializzazione ortogonale + bias zero
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0)
        
        # Inizializzazione più stretta per gli output
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)

        self.dropout1 = nn.Dropout(p=0.1)  # Dopo fc1
        self.dropout2 = nn.Dropout(p=0.1)  # Dopo fc2
        

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)  
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        #log_std = torch.clamp(log_std, min=-20, max=2)  # Clamp to prevent instability
        return mean, log_std#.sum(dim=-1, keepdim=True)
    
    # def get_action(self, state_encoded):
    #     mean, log_std = self.forward(state_encoded)
    #     std = torch.exp(log_std)
    #     # Sample from a normal distribution and squash using tanh
    #     normal = torch.distributions.Normal(mean, std)
    #     z = normal.rsample()  # Reparameterization trick
    #     action = torch.tanh(z)*2 # Le azioni hanno bound -2 2
    #     log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
    #     return action.detach().cpu().numpy(), log_prob.sum(dim=-1, keepdim=False)
    def get_action(self, state_encoded):
        mean, log_std = self.forward(state_encoded)
        log_std = torch.clamp(log_std, min=-5, max=2) 
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        
        # Scaling corretto con compensazione del log_prob
        action = torch.tanh(z)*2 # Scaling tra [-1, 1]
        log_prob = normal.log_prob(z) - torch.log(2.0 * (1 - torch.tanh(z).pow(2)) + 1e-6)#torch.log(1 - (action/2).pow(2) + 1e-6)  # Termine di correzione standard
        
        return action, log_prob.sum(dim=-1, keepdim=True)
    
class Critic(nn.Module):
    def __init__(self, state_action_dim):
        super(Critic, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        self.fc1 = nn.Linear(state_action_dim,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,64)
        # self.fc4 = nn.Linear(64, 32) 
        self.fc5 = nn.Linear(64, 1)   

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.orthogonal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0.0) 
    
        self.dropout1 = nn.Dropout(p=0.1)  # Dopo fc1
        self.dropout2 = nn.Dropout(p=0.1)  # Dopo fc2

    def forward(self, state , action): #         
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  
        x = F.relu(self.fc2(x))
        x = self.dropout2(x) 
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x



# class ReplayBufferHER:
#     def __init__(self, capacity, goal_func): #k=4
#         self.buffer = deque(maxlen=capacity)
#         #self.k = k  # Numero di esperienze HER per ogni transizione reale
#         self.goal_func = goal_func  # Funzione per estrarre obiettivi alternativi

#     def push(self, state, action, reward, next_state, done, goal):
#         self.buffer.append((state, action, reward, next_state, done, goal))
#         # achieved_goal = next_state
#         # self.buffer.append((state, action, reward, next_state, done, goal))
class ReplayBufferHER:
    def __init__(self, capacity, goal_func, k=4, strategy='future'):
        self.buffer = deque(maxlen=capacity)
        self.goal_func = goal_func  # Function to extract achieved_goal from state
        self.k = k                  # Number of HER goals per transition
        self.strategy = strategy    # 'future', 'final', etc.

    def add_episode(self, episode):
        # Process an entire episode to generate HER transitions
        print(len(episode))
        for t in range(len(episode)):
            state, action, reward, next_state, done, original_goal = episode[t]
            
            # Add original transition
            self.buffer.append((state, action, reward, next_state, done, original_goal))
            
            # Extract achieved goals from the episode
            achieved_goals = [self.goal_func(trans[3]) for trans in episode]  # trans[3] is next_state
            
            # Select HER goals based on strategy
            if self.strategy == 'future':
                possible_indices = list(range(t, len(episode)))
                selected_indices = np.random.choice(possible_indices, size=self.k, replace=True)
            elif self.strategy == 'final':
                selected_indices = [len(episode)-1] * self.k
            else:
                raise ValueError(f"Unknown HER strategy: {self.strategy}")
            for idx in selected_indices:
                new_goal = achieved_goals[idx]
                # Compute reward for the new goal
                achieved_goal_current = self.goal_func(next_state)
                distance = -np.linalg.norm(achieved_goal_current - new_goal)
                new_reward = 0.0 if distance > -0.25 else -1.0  # Adjust threshold as needed
                # Add HER transition
                self.buffer.append((state, action, new_reward, next_state, done, new_goal))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, goals = zip(*batch)
        states = np.stack(states)  # (batch_size, state_dim)
        actions = np.stack(actions)  # (batch_size, action_dim)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)  # (batch_size, 1)
        next_states = np.stack(next_states)  # (batch_size, state_dim)
        dones = np.array(dones, dtype=np.float32).reshape(-1, 1)  # (batch_size, 1)
        goals = np.stack(goals)
        return states, actions, rewards, next_states, dones, goals
    
    # def her_relabel(self,pos_oggetto, batch_size=500):
    #     augmented_buffer = []
    #     # batch = list(self.buffer)[-batch_size:]
    #     batch = random.sample(self.buffer, batch_size)
    #     for state, action, reward, next_state, done, goal in batch:
    #         new_goal = self.goal_func(next_state)
    #         new_reward = - np.linalg.norm(next_state[:, 0:3] - pos_oggetto) 
    #         if new_reward > -0.2:
    #             augmented_buffer.append((state, action, new_reward, next_state, False, goal))
    #     self.buffer.extend(augmented_buffer)
    #             # print("Aug")

    def __len__(self):
         return len(self.buffer)

        
    
class SACAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, tau=0.95, alpha=0.2):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.q1 = Critic(state_dim + action_dim)
        self.q2 = Critic(state_dim + action_dim)
        self.policy = Actor(state_dim, action_dim)
        self.q1_target = Critic(state_dim + action_dim)
        self.q2_target = Critic(state_dim + action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=1e-4)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.replay_buffer = None  # Il buffer viene assegnato esternamente

        # Aggiungi queste linee per l'auto-entropy adjustment
        self.target_entropy = -action_dim  # -dim(A)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
    
    def update(self, batch_size=128):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q1.to(device)
        self.q2.to(device)
        self.q1_target.to(device)
        self.q2_target.to(device)
        self.policy.to(device)

        for _ in range(40):
            states, actions, rewards, next_states, dones, goals = self.replay_buffer.sample(batch_size)
            #print(rewards)
            states = torch.FloatTensor(states).squeeze(dim=1)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards)#.unsqueeze(1)
            next_states = torch.FloatTensor(next_states).squeeze(dim=1)
            dones = torch.FloatTensor(dones)#.unsqueeze(1)
            goals = torch.FloatTensor(goals)
            
            with torch.no_grad():
                next_actions, log_probs = self.policy.get_action(next_states)
                # next_actions = next_actions.squeeze(dim=1)
                # No need to convert to tensor; actions are already tensors
                q1_next = self.q1_target(next_states, next_actions)
                q2_next = self.q2_target(next_states, next_actions)
                q_next = torch.min(q1_next, q2_next) - self.alpha * log_probs # - 
                q_target = rewards + self.gamma * (1 - dones) * q_next  #target_value = reward + gamma * Q_target(next_state, policy(next_state)) 
                min_target = -1.0 / (1 - self.gamma)
                max_target = 0.0
                q_target = torch.clamp(q_target, min=min_target, max=max_target)
            
            # print(states.shape,next_states.shape,actions.shape,next_actions.shape)
            q1_pred = self.q1(states, actions)
            q2_pred = self.q2(states, actions)
            #print(q1_pred[1].item(), q2_pred[1].item(), q_next[1].item(),q_target[1].item(),log_probs[1].item())
            q_loss = F.mse_loss(q_target,q1_pred) + F.mse_loss(q_target,q2_pred)
            #print("Value_loss",q_loss)
            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_norm=0.2)
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_norm=0.2)
            self.q_optimizer.step()
            

            new_actions, log_probs_new = self.policy.get_action(states)
            q1_new = self.q1(states, new_actions)
            q2_new = self.q2(states, new_actions)
            policy_loss = (torch.min(q1_new, q2_new)-self.alpha * log_probs_new ).mean()
            
            print("Policy_loss",policy_loss)
            print("Value_loss",q_loss)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            alpha_loss = -(self.log_alpha * (log_probs_new + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().detach()

            self.policy_optimizer.step()

    def soft_update_targets(self):
        with torch.no_grad():
            for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
                target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)
            for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
                target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)

    
