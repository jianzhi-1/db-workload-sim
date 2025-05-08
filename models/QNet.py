import random
import torch
import torch.optim as optim
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, N, T, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.N = N
        self.T = T
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(N*N*(2*T+1) + N, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N)
        )

    def forward(self, x):
        return self.model(x)

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

    def get_state_tensor(self, conflict_matrix, mask):
        # essentially flatten and concatenate the conflict matrix and the mask
        N, T = self.N, self.T # convenience
        assert conflict_matrix.shape == (N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"
        assert mask.shape == (N,), f"mask.shape = {mask.shape}"
        res = torch.cat([conflict_matrix.flatten(), mask], dim=0).float().to(self.device)
        assert res.shape == (N*N*(2*T+1)+N,), f"res.shape = {res.shape}"
        return res

    def select_action(self, state_tensor, mask):
        # action = arg max_a(Q(s, a))
        # unless epsilon, then explore a bit
        assert state_tensor.shape == (self.N*self.N*(2*self.T+1)+self.N,), f"state_tensor.shape = {state_tensor.shape}"
        assert mask.shape == (self.N,), f"mask.shape = {mask.shape}"
        with torch.no_grad():
            q_values = self.q_net(state_tensor.unsqueeze(0)).squeeze()
            q_values[mask == 1] = float('-inf')  # transactions that are already scheduled are masked out
            if random.random() < self.epsilon: # epsilon-greedy exploration
                valid_actions = torch.where(mask == 0)[0]
                return random.choice(valid_actions.tolist())
            return torch.argmax(q_values).item()
        assert False # should be unreachable

    def train_step(self, state_tensor, action, reward, next_state_tensor, done):
        assert state_tensor.shape == (self.N*self.N*(2*self.T+1)+self.N,), f"state_tensor.shape = {state_tensor.shape}"
        assert next_state_tensor.shape == (self.N*self.N*(2*self.T+1)+self.N,), f"next_state_tensor.shape = {next_state_tensor.shape}"
        q_values = self.q_net(state_tensor.unsqueeze(0)).squeeze()
        next_q_values = self.q_net(next_state_tensor.unsqueeze(0)).squeeze()

        target = reward
        if not done: target += self.gamma * next_q_values.max().item()
        loss = nn.MSELoss()(q_values[action], torch.tensor(target).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run_episode(self, conflict_matrix):
        N, T = self.N, self.T # convenience
        assert conflict_matrix.shape == (1, N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"
        conflict_matrix = conflict_matrix.squeeze()
        assert conflict_matrix.shape == (N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"
        mask = torch.zeros(self.N)
        assert mask.shape == (self.N,), f"mask.shape = {mask.shape}"

        total_reward = 0
        scheduled = []

        for step in range(self.N):
            state_tensor = self.get_state_tensor(conflict_matrix, mask)
            assert state_tensor.shape == (N*N*(2*T+1)+N,), f"state_tensor.shape = {state_tensor.shape}"
            action = self.select_action(state_tensor, mask)
            assert 0 <= action and action < N, f"action = {action}"

            reward, pos = self.compute_reward(action, scheduled, conflict_matrix)
            total_reward += reward

            scheduled.append((action, pos))
            mask[action] = 1

            done = (mask.sum() == self.N)
            next_state_tensor = self.get_state_tensor(conflict_matrix, mask)
            assert next_state_tensor.shape == (N*N*(2*T+1)+N,), f"next_state_tensor.shape = {state_tensor.shape}"

            self.train_step(state_tensor, action, reward, next_state_tensor, done)

            if done: break

        return total_reward

    def compute_reward(self, new_action, scheduled, conflict_matrix):
        N, T = self.N, self.T # convenience
        assert 0 <= new_action and new_action < N, f"new_action = f{new_action}"
        assert conflict_matrix.shape == (N, N, 2*T+1), f"conflict_matrix.shape = {conflict_matrix.shape}"

        reward = 0.
        success, pos = False, -1 # whether can actually schedule i and the earliest it can be scheduled
        for t in range(0, T+1, 1):
            # train to schedule transaction i at time t
            temp_success = True
            for j, tj in scheduled:
                if conflict_matrix[new_action][j][tj - t + T] == 1: # conflict
                    temp_success = False
                    break
                    #return 0., -1
            if temp_success: # assign at earliest
                success, pos = True, t
                break
        if success:
            reward_curve = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
            return reward_curve[pos], pos
            #return 1, pos
        return -5., -1