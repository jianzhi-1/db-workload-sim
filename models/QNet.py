import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, N, T, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.N = N
        self.T = T
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(N*N*(2*T+1) + N, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N+1)
        )
        self.epsilon = 0.1

    def obtain_schedule(self, conflict_matrix):
        N, T = self.N, self.T # convenience
        assert conflict_matrix.shape == (1, N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"
        conflict_matrix = conflict_matrix.squeeze()
        assert conflict_matrix.shape == (N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"
        mask = torch.zeros(self.N)
        assert mask.shape == (self.N,), f"mask.shape = {mask.shape}"

        skipped = []
        scheduled = []

        for i in range(N):
            if conflict_matrix[i].sum() == 0:
                scheduled.append((i,0))
                mask[i] = 1

        for _ in range(self.N):
            state_tensor = self.get_state_tensor(conflict_matrix, mask)
            assert state_tensor.shape == (N*N*(2*T+1)+N,), f"state_tensor.shape = {state_tensor.shape}"
            action = self.select_action(state_tensor, mask)
            #assert 0 <= action and action < N, f"action = {action}"

            if action >= N:
                done = True
            else:
                pos = self.get_position(action, scheduled, conflict_matrix, mask)
                scheduled.append((action, pos))
                mask[action] = 1

                done = (mask.sum() == self.N)
                next_state_tensor = self.get_state_tensor(conflict_matrix, mask)
                assert next_state_tensor.shape == (N*N*(2*T+1)+N,), f"next_state_tensor.shape = {state_tensor.shape}"
            if done: break

        return skipped + scheduled, mask
    
    def select_action(self, state_tensor, mask):
        # action = arg max_a(Q(s, a))
        # unless epsilon, then explore a bit
        assert state_tensor.shape == (self.N*self.N*(2*self.T+1)+self.N,), f"state_tensor.shape = {state_tensor.shape}"
        assert mask.shape == (self.N,), f"mask.shape = {mask.shape}"
        with torch.no_grad():
            q_values = self.model(state_tensor)
            adjusted_mask = torch.cat([mask, torch.tensor([0])])
            # Convert mask to boolean tensor for indexing
            mask_bool = (adjusted_mask == 1)
            q_values[mask_bool] = float('-inf')  # transactions that are already scheduled are masked out
            if random.random() < self.epsilon: # epsilon-greedy exploration
               valid_actions = torch.where(mask == 0)[0].tolist()
               if len(valid_actions) == 0:
                   return self.N+1
               return random.choice(valid_actions)
            return torch.argmax(q_values).item()
        assert False # should be unreachable

    def get_position(self, new_action, scheduled, conflict_matrix, mask):
        N, T = self.N, self.T # convenience
        #assert 0 <= new_action and new_action < N, f"new_action = f{new_action}"
        assert conflict_matrix.shape == (N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"

        success, pos = False, -1 # whether can actually schedule i and the earliest it can be scheduled

        for t in range(0, T+1, 1):
            # train to schedule transaction i at time t
            temp_success = True
            for j, tj in scheduled:
                if conflict_matrix[new_action][j][tj - t + T] == 1: # conflict
                    temp_success = False
                    break
            if temp_success: # assign at earliest
                success, pos = True, t
                break
        if success:
            return pos
        return -1
    
    def get_state_tensor(self, conflict_matrix, mask):
        # essentially flatten and concatenate the conflict matrix and the mask
        N, T = self.N, self.T # convenience
        assert conflict_matrix.shape == (N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"
        assert mask.shape == (N,), f"mask.shape = {mask.shape}"
        res = torch.cat([conflict_matrix.flatten(), mask], dim=0).float().to(self.device)
        assert res.shape == (N*N*(2*T+1)+N,), f"res.shape = {res.shape}"
        return res

class Trainer:
    def __init__(self, N, T, lr=1e-3, gamma=0.9, epsilon=0.1, q_net=None):
        self.N = N
        self.T = T
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if q_net is None:
            self.q_net = QNetwork(N, T).to(self.device) # Q function
        else:
            self.q_net = q_net
        # state space is of dimension (N*N*(2*T+1)+N,)
        # action space is of dimension (N,)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma # discount rate
        self.epsilon = epsilon # epsilon-greedy

    def train_step(self, state_tensor, action, reward, next_state_tensor, done):
        assert state_tensor.shape == (self.N*self.N*(2*self.T+1)+self.N,), f"state_tensor.shape = {state_tensor.shape}"
        assert next_state_tensor.shape == (self.N*self.N*(2*self.T+1)+self.N,), f"next_state_tensor.shape = {next_state_tensor.shape}"
        q_values = self.q_net.model(state_tensor)
        next_q_values = self.q_net.model(next_state_tensor)

        target = reward
        if not done: target += self.gamma * next_q_values.max().item()
        loss = nn.MSELoss()(q_values[action], torch.tensor(target).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_episode(self, conflict_matrix, episode_number=0):
        N, T = self.N, self.T # convenience
        assert conflict_matrix.shape == (1, N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"
        conflict_matrix = conflict_matrix.squeeze()
        assert conflict_matrix.shape == (N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"
        mask = torch.zeros(self.N)
        assert mask.shape == (self.N,), f"mask.shape = {mask.shape}"

        total_reward = 0
        scheduled = []

        self.epsilon = self.epsilon * 0.9

        skipped = []
        for i in range(N):
            if conflict_matrix[i].sum() == 0:
                skipped.append((i, 0))
                mask[i] = 1
        if episode_number >= 99:
            print(f'skipped: {sorted(skipped)}', flush=True)
        
        if len(skipped) == self.N:
            return None

        for step in range(self.N - len(skipped)):
            state_tensor = self.q_net.get_state_tensor(conflict_matrix, mask)
            assert state_tensor.shape == (N*N*(2*T+1)+N,), f"state_tensor.shape = {state_tensor.shape}"
            action = self.q_net.select_action(state_tensor, mask)
            #assert 0 <= action and action < N, f"action = {action}"

            reward, pos = self.compute_reward(action, scheduled, conflict_matrix, mask)
            total_reward += reward

            if action >= N:
                done = True
            else:
                scheduled.append((action, pos))
                mask[action] = 1

                done = (mask.sum() == self.N)
                next_state_tensor = self.q_net.get_state_tensor(conflict_matrix, mask)
                assert next_state_tensor.shape == (N*N*(2*T+1)+N,), f"next_state_tensor.shape = {state_tensor.shape}"

                self.train_step(state_tensor, action, reward, next_state_tensor, done)

            if done: break
        if episode_number >= 99:
            print(f'scheduled: {sorted(scheduled)}', flush=True)
            print(mask)
        return total_reward

    def compute_reward(self, new_action, scheduled, conflict_matrix, mask):
        N, T = self.N, self.T # convenience
        #assert 0 <= new_action and new_action < N, f"new_action = f{new_action}"
        assert conflict_matrix.shape == (N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"

        success, pos = False, -1 # whether can actually schedule i and the earliest it can be scheduled

        if new_action >= N:
            # total_reward = 0
            conflict_points = 0
            pos_reward = 0
            neg_reward = 0.5
            # counter = 0
            for i in range(N):
                increment = 0.5
                if mask[i] == 0:
                    for t in range(0, T+1, 1):
                        # train to schedule transaction i at time t
                        temp_success = True
                        for j, tj in scheduled:
                            # conflict_points += sum(conflict_matrix[i][j])
                            if conflict_matrix[i][j][tj - t + T] == 1: # conflict
                                temp_success = False
                                break
                        if temp_success: # assign at earliest
                            success = True
                            break
                    if success: #ended when could've scheduled
                        neg_reward -= increment
                        increment = increment * 1.5
                    else: #would've been a conflict, reward for stopping
                        pos_reward += 2
                        # increment = 0
            # conflict_points = max(0, conflict_points - 30)
            # if neg_reward >= 0:
                # pos_reward += 10
            # print(conflict_points // 10, pos_reward, neg_reward, flush=True)
            # conflict_points = min(conflict_points // 10, 6)
            
            return pos_reward + neg_reward, -1

        # difficulty = 1
        neg_reward = -3.
        for t in range(0, T+1, 1):
            # train to schedule transaction i at time t
            temp_success = True
            for j, tj in scheduled:
                if conflict_matrix[new_action][j][tj - t + T] == 1: # conflict
                    temp_success = False
                    neg_reward *= 1.5
                    break
            if temp_success: # assign at earliest
                success, pos = True, t
                break
        if success:
            reward_curve = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            # print('success', reward_curve[pos], difficulty, flush=True)
            return reward_curve[pos], pos
        # print('fail', difficulty, flush=True)
        return neg_reward, -1
    
    def print_conflict_matrix(conflict_tensor):
        conflict_matrix = conflict_tensor[0].cpu().numpy()
        for i in range(len(conflict_matrix)):
            conflicts = []
            for j in range(len(conflict_matrix[i])):
                if np.any(conflict_matrix[i][j]):
                    conflicts.append(j)
            print(f'{i}: {conflicts}', flush=True)