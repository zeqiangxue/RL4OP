import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

# 设置随机种子
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- 环境：共源放大器 --------------------
class CSAmplifierEnv:
    def __init__(self, target_gain=20.0):
        # 工艺参数
        self.mu_cox = 200e-6   # A/V^2
        self.vth = 0.5          # V
        self.lambd = 0.05       # V^{-1}
        self.vgs = 0.8           # 固定栅源电压
        self.rl = 10e3           # 负载电阻 10kΩ
        self.l = 1e-7            # 固定沟道长度 1μm
        self.w_min = 2e-6
        self.w_max = 100e-6
        self.target_gain = target_gain
        self.max_steps = 20      # 每个episode最大步数
        self.step_count = 0
        
    def reset(self):
        # 随机初始化宽度
        self.w = np.random.uniform(self.w_min, self.w_max)
        self.step_count = 0
        return self._get_state()
    
    def _get_state(self):
        # 状态：归一化的宽度 (0~1)
        norm_w = (self.w - self.w_min) / (self.w_max - self.w_min)
        return np.array([norm_w], dtype=np.float32)
    
    def _compute_gain(self, w):
        # 计算增益 Av = gm * (ro || RL)
        # 过驱动电压
        vov = self.vgs - self.vth
        if vov <= 0:
            return 0.0
        # 漏极电流 Id = 0.5 * μCox * (W/L) * Vov^2
        id_ = 0.5 * self.mu_cox * (w / self.l) * vov**2
        # 跨导 gm = μCox * (W/L) * Vov
        gm = self.mu_cox * (w / self.l) * vov
        # 输出电阻 ro = 1/(λ Id)
        ro = 1 / (self.lambd * id_) if id_ > 0 else 1e12
        # 等效输出电阻
        rout = 1 / (1/ro + 1/self.rl)
        gain = gm * rout
        return gain
    
    def step(self, action):
        # action 是 [-1, 1] 的连续值，表示宽度的相对变化量
        # 将动作映射为宽度的增量比例（例如最大变化 ±20%）
        delta_ratio = action * 0.2   # 单步最大调整 ±20%
        new_w = self.w * (1 + delta_ratio)
        # 确保宽度在范围内
        new_w = np.clip(new_w, self.w_min, self.w_max)
        self.w = new_w
        self.step_count += 1
        
        # 计算增益
        gain = self._compute_gain(self.w)
        # 奖励 = -|增益误差|，若误差小于1则给予正向奖励
        error = abs(gain - self.target_gain)
        reward = -error
        if error < 1.0:
            reward += 5.0   # 额外奖励
        
        # 判断终止
        done = (self.step_count >= self.max_steps) or (error < 0.1)
        
        # 返回下一个状态、奖励、是否终止、额外信息
        return self._get_state(), reward, done, {'gain': gain, 'w': self.w}

# -------------------- 网络定义 --------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()   # 输出在 [-1, 1]
        )
    
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# -------------------- 经验回放 --------------------
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(next_state).to(device),
                torch.FloatTensor(done).unsqueeze(1).to(device))
    
    def __len__(self):
        return len(self.buffer)

# -------------------- DDPG智能体 --------------------
class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        
        # 初始化目标网络参数与主网络相同
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.tau = tau
        
    def select_action(self, state, noise_scale=0.1):
        # 状态转为tensor
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        # 添加探索噪声（奥恩斯坦-乌伦贝克噪声或高斯噪声）
        action = action + np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action, -1, 1)
        return action
    
    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        
        # 采样
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # 更新Critic
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q = self.target_critic(next_state, next_action)
            target_q = reward + (1 - done) * self.gamma * target_q
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# -------------------- 训练主循环 --------------------
def train(env, agent, episodes=200, batch_size=64, buffer_capacity=10000, print_interval=20):
    replay_buffer = ReplayBuffer(buffer_capacity)
    episode_rewards = []
    best_error = float('inf')
    
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, noise_scale=0.1)  # 探索噪声
            next_state, reward, done, info = env.step(action[0])  # action是1维数组
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            
            agent.update(replay_buffer, batch_size)
        
        episode_rewards.append(ep_reward)
        
        # 记录最佳结果
        current_gain = info['gain']
        error = abs(current_gain - env.target_gain)
        if error < best_error:
            best_error = error
            best_w = info['w']
        
        if (ep+1) % print_interval == 0:
            print(f"Episode {ep+1}, Reward: {ep_reward:.2f}, Gain: {current_gain:.2f}, W: {info['w']:.2e}, Best Error: {best_error:.4f}")
    
    # 绘制奖励曲线
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DDPG Training Progress')
    plt.show()
    
    return best_w, best_error

# -------------------- 运行 --------------------
if __name__ == "__main__":
    env = CSAmplifierEnv(target_gain=20.0)
    agent = DDPGAgent(state_dim=1, action_dim=1)  # 状态：归一化W，动作：调整比例
    best_w, best_error = train(env, agent, episodes=200)
    print(f"\nOptimal W found: {best_w:.2e} m, Gain error: {best_error:.4f}")
    
    # 测试最终结果
    final_gain = env._compute_gain(best_w)
    print(f"Final gain with optimal W: {final_gain:.2f} V/V")