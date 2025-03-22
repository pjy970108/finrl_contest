import torch
import torch.nn as nn
# from TARN_MAPPO.network import ActorCritic
from TARN_MAPPO.enviroment import EnvConfig
from TARN_PPO.agent import PPO
import wandb
import torch.optim as optim
import numpy as np
from copy import deepcopy
import time
# from line_profiler import LineProfiler

# #### set device####
# print("============================================================================================")
# # set device to cpu or cuda
# device = torch.device('cpu')
# if(torch.cuda.is_available()): 
#     device = torch.device('cuda:0') 
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
# print("============================================================================================")

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
        - patience (int): 개선이 없더라도 기다릴 에포크 수.
        - min_delta (float): 개선으로 간주되는 최소 변화량.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = float('-inf')  # 초기값을 -무한대로 설정
        self.early_stop = False
        
    def __call__(self, current_score):
        # 상대적 개선 평가
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0  # 개선되면 카운터 초기화
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    # def clear(self):
    #     del self.actions[:]
    #     del self.states[:]
    #     del self.logprobs[:]
    #     del self.rewards[:]
    #     del self.state_values[:]
    #     del self.is_terminals[:]
        

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.state_values.clear()
        self.is_terminals.clear()


class MAPPO:
    """
    Multi-Agent PPO (중앙집중형 Critic, 여러 Actor)
    기존의 PPO 클래스를 확장/수정:
      - critic: CentralCriticNet (1개)
      - actors: [ActorNet1, ActorNet2, ...] (에이전트 수만큼)
      - buffer: MultiAgentRolloutBuffer (공동)
    """
    def __init__(self, config):
        self.config = config
        self.num_agents = config.num_agents
        self.agents = []
        self.buffers = []
        self.best_agent_idx = 0
        self.best_reward = float('-inf')
        self.save_path = config.save_path
        
        # CUDA 최적화 설정
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        
        for _ in range(self.num_agents):
            agent = PPO(deepcopy(config))
            self.agents.append(agent)
            self.buffers.append(agent.buffer)

    @torch.no_grad()
    def select_actions(self, state, deterministic=False):
        """
        모든 에이전트가 행동을 선택하고 그 중 최고의 에이전트의 행동을 반환
        """
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action, att_score = agent.select_action(state, deterministic)
            actions.append(action)
        
        # 학습 중일 때는 모든 에이전트의 행동을 저장
        if not deterministic:
            return actions
        # 평가 시에는 최고 성능의 에이전트 행동만 반환
        else:
            return actions[self.best_agent_idx]

        # actions = []
        # # 배치로 한 번에 처리
        # state_batch = state.to(self.config.device)
        
        # for agent_idx, agent in enumerate(self.agents):
        #     action = agent.select_action(state_batch, deterministic)
        #     actions.append(action)
        
        # if not deterministic:
        #     return actions
        # else:
        #     return actions[self.best_agent_idx]


    def store_reward(self, rewards, is_terminals):
        """
        각 에이전트의 버퍼에 보상 저장
        """
        for agent_idx in range(self.num_agents):
            self.agents[agent_idx].buffer.rewards.append(rewards[agent_idx])
            self.agents[agent_idx].buffer.is_terminals.append(is_terminals[agent_idx])
    
    def all_update(self):
        """
        모든 에이전트 업데이트 및 최고 성능 에이전트 선정
        """

        
        agent_returns = []
        
        # 각 에이전트 업데이트
        for agent_idx, agent in enumerate(self.agents):
            reward_dis = agent.update()
            
            # 각 에이전트의 평균 리턴 계산
            # returns = sum(agent.buffer.rewards) / len(agent.buffer.rewards) if agent.buffer.rewards else 0
            agent_returns.append(reward_dis[0])

            # wandb에 각 에이전트의 성능 기록
            # wandb.log({
            #     f"agent_{agent_idx}_return": agent.buffer.rewards
            #     # f"agent_{agent_idx}_loss": agent.last_loss
            # })
            
            # agent.buffer.clear()

        # 최고 성능 에이전트 선정
        agent_returns_tensor = torch.stack(agent_returns)  # 리스트를 텐서로 변환
        current_best_idx = torch.argmax(agent_returns_tensor).item()  # 인덱스 추출

        if agent_returns[current_best_idx] > self.best_reward:
            self.best_reward = agent_returns[current_best_idx]
            self.best_agent_idx = current_best_idx
            
            # 최고 성능 에이전트의 정책을 다른 에이전트들에게 부분적으로 전파
            best_agent_state = self.agents[current_best_idx].policy.state_dict()
            
            # soft update 방식 적용
            for idx, agent in enumerate(self.agents):
                if idx != current_best_idx:
                    self.soft_update(agent.policy, self.agents[current_best_idx].policy, tau=0.05)
                    self.soft_update(agent.policy_old, self.agents[current_best_idx].policy_old, tau=0.05)
                     
            # for idx, agent in enumerate(self.agents):
            #     if idx != current_best_idx:
            #         # 랜덤하게 일부 파라미터만 복사 (다양성 유지를 위해)
            #         current_state = agent.policy.state_dict()
            #         for key in current_state.keys():
            #             if np.random.random() < 0.3:  # 30% 확률로 파라미터 복사
            #                 current_state[key] = best_agent_state[key].clone()
            #         agent.policy.load_state_dict(current_state)
            #         agent.policy_old.load_state_dict(current_state)
        
        # wandb.log({
        #     "best_agent_idx": self.best_agent_idx,
        #     "best_reward": self.best_reward
        # })
        
        return agent_returns


    def soft_update(self, target, source, tau=0.05):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)



    def save(self):
        """
        최고 성능 에이전트 저장
        """
        # self.agents[self.best_agent_idx].save()
        torch.save(self.agents[self.best_agent_idx].policy_old.state_dict(), self.save_path)

        
    def load(self):
        """
        저장된 최고 성능 에이전트 로드
        """
        self.agents[self.best_agent_idx].policy_old.load_state_dict(torch.load(self.save_path, map_location=lambda storage, loc: storage))
        self.agents[self.best_agent_idx].policy.load_state_dict(torch.load(self.save_path, map_location=lambda storage, loc: storage))
        print(self.save_path + " loaded.")
        
        # self.agents[self.best_agent_idx].load()




# class PPO:
#     def __init__(self, config):
#         """
#         state_dim, action_dim: 상태 및 행동 공간의 차원 수
#         lr_actor, lr_critic: 정책 네트워크와 가치 네트워크의 학습률
#         gamma: 할인 계수
#         K_epochs: 정책을 업데이트할 에포크 수
#         eps_clip: 정책 업데이트 시 클리핑 범위 (PPO의 핵심)
#         has_continuous_action_space: 연속적 행동 공간 여부
#         action_std_init: 연속적 행동 공간에서 행동의 표준 편차 초기값
#         """
        
#         self.has_continuous_action_space = config.has_continuous_action_space
#         self.gamma = config.gamma
#         self.eps_clip = config.eps_clip
#         self.K_epochs = config.K_epochs
#         # self.state_dim = config.state_dim
#         self.feature_dim = config.feature_dim
#         self.action_dim = config.action_dim
#         self.action_std_init = config.action_std_init
#         self.device = config.device
#         self.lr_actor = config.lr_actor
#         self.lr_critic = config.lr_critic
#         self.save_path = config.save_path
#         self.hyperparameters = config.hyperparameters

#         # Initialize buffer
#         self.buffer = RolloutBuffer()

#         # Actor, Critic 네트워크 정의함. 상태 및 행동 차원, 공간유형을 받아 초기화함.
#         # self.policy = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init).to(self.device)
#         self.policy = ActorCritic(self.feature_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, self.hyperparameters, self.device).to(self.device)

#         self.optimizer = torch.optim.Adam([
#             {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
#             {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
#         ])

#         # 이전 정책 네트워크 정의함. 현재 정책 네트워크의 가중치를 복사하여 초기화함.
#         # self.policy_old = ActorCritic(self.state_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init).to(self.device)
#         self.policy_old = ActorCritic(self.feature_dim, self.action_dim, self.has_continuous_action_space, self.action_std_init, self.hyperparameters, self.device).to(self.device)

#         # policy의 가중치를 policy_old에 복사함
#         self.policy_old.load_state_dict(self.policy.state_dict())

#         # Loss function
#         self.MseLoss = nn.MSELoss()


#     # 연속된 공간에서 행동의 표준 편차 설정함
#     def set_action_std(self, new_action_std):
#         if self.has_continuous_action_space:
#             self.policy.set_action_std(new_action_std)
#             self.policy_old.set_action_std(new_action_std)
#         # else:
#         #     print("--------------------------------------------------------------------------------------------")
#         #     print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
#        #     print("--------------------------------------------------------------------------------------------")

#     # 조건을 만족할때 행동의 표준 편차 감소시킴
#     def decay_action_std(self, action_std_decay_rate, min_action_std):
#         # print("--------------------------------------------------------------------------------------------")
#         if self.has_continuous_action_space:
#             self.action_std = self.action_std - action_std_decay_rate
#             self.action_std = round(self.action_std, 4)
#             if (self.action_std <= min_action_std):
#                 self.action_std = min_action_std
#                 print("setting actor output action_std to min_action_std : ", self.action_std)
#             else:
#                 print("setting actor output action_std to : ", self.action_std)
#             self.set_action_std(self.action_std)

#         # else:
#         #     print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
#         # print("--------------------------------------------------------------------------------------------")
    
    
    
#     @torch.no_grad()
       
#     def select_action(self, state, deterministic=False):
#         # 만약 연속 행동 공간이면
#         if self.has_continuous_action_space:
#             with torch.no_grad():
#                 state = state.to(self.device, dtype=torch.float)
#                 action, action_logprob, state_val = self.policy_old.act(state, deterministic)

#         if not deterministic:  # 학습용 데이터는 stochastic일 때만 저장
#             self.buffer.states.append(state)
#             self.buffer.actions.append(action)
#             self.buffer.logprobs.append(action_logprob)
#             self.buffer.state_values.append(state_val)

#         return action.detach().cpu().numpy().flatten()



#     # 정책 업데이트 
#     def update(self):
#         # Monte Carlo 방식으로 리턴 계산

#         rewards = []
#         discounted_reward = 0
#         # buffer rewards가 들어감
#         # uptime_start = time.time()
#         for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
#             # 반약 is_terminal이 True면 discounted reward를 0으로 설정함.
#             if is_terminal:
#                 discounted_reward = 0
            
#             # 계산된 리턴을 텐서로 변환하고 보상을 정규화함. 
#             discounted_reward = reward + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)
#         # Normalizing the rewards, 성능 향상을 위해 필요함.
#         # end_uptime = time.time()
#         # print("uptime time : ", end_uptime - uptime_start)
        
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

#         # convert list to tensor
#         # 버퍼의 데이터 변환 및 분리 - 버퍼에 저장된 상태, 행동, 로그 확률, 상태 가치를 텐서로 변환하고 detach를 사용해 경사 계산에서 제외함.
#         old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).to(self.device)
#         old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).to(self.device)
#         old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
#         old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).to(self.device)

#         advantages = rewards.detach() - old_state_values.detach()

#         # Optimize policy for K epochs
#         # K_epoch 동안 정책을 반복적으로 업데이트함.
#         total_policy_loss = 0
#         total_value_loss = 0
#         total_entropy_bonus = 0
#         for _ in range(self.K_epochs):
            
#             # start_time = time.time()

#             # Evaluating old actions and values
#             # 이전 state와 action으로 로그 확률, 상태 가치, 분포의 엔트로피 계산함.
#             logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
#             # eval_time = time.time()
#             # print("eval time : ", eval_time - start_time)
#             # match state_values tensor dimensions with rewards tensor
#             # 상태 가치 텐서를 정리함.
#             state_values = torch.squeeze(state_values)
            
#             # squeeze_time = time.time()
#             # print("squeeze time : ", squeeze_time - eval_time)

#             # ratios = action / old_actions # 200step의 행동의 ratio
#             ratios = torch.exp(logprobs - old_logprobs.detach())
#             # Finding Surrogate Loss
#             # surr1 : 클리핑되지 않은 손실함수
#             surr1 = ratios * advantages
#             # surr ratio를 범위로 제한하여 손실함수
#             surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
#             policy_loss = torch.min(surr1, surr2).mean()
#             value_loss = 0.5 * self.MseLoss(state_values, rewards)
#             entropy_bonus = dist_entropy.mean()
#             loss = -policy_loss + value_loss - 0.01 * entropy_bonus
#             # loss_time = time.time()
            
#             # print("loss time : ", loss_time - squeeze_time)

#             # take gradient step
#             # profiler = LineProfiler()
#             # profiler.add_function(torch.Tensor.backward)
#             self.optimizer.zero_grad()

#             # import torch.autograd.profiler as profiler
#             # with profiler.profile(record_shapes=True) as prof:
#             loss.mean().backward()
#             # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


#             # loss.mean().backward()


#             self.optimizer.step()

            
#             # opt_time = time.time()
#             # print("opt time : ", opt_time - loss_time)
#             # Accumulate losses for logging
#             total_policy_loss += policy_loss.item()
#             total_value_loss += value_loss.item()
#             total_entropy_bonus += entropy_bonus.item()

#         #     sum_loss_time = time.time()
#         #     print("sum loss time : ", sum_loss_time - opt_time)
            
#         # start_log_time = time.time()
#         # Log losses
#         self.policy_loss = total_policy_loss / self.K_epochs
#         self.value_loss = total_value_loss / self.K_epochs,
#         self.entropy_bonus =  total_entropy_bonus / self.K_epochs,
#         self.last_loss = loss.mean().item()

#         self.policy_old.load_state_dict(self.policy.state_dict())

        
#         # end_log_time = time.time()
#         # print("log time : ", end_log_time - start_log_time)


#     def save(self):
#         # 모델 저장
#         torch.save(self.policy_old.state_dict(), self.save_path)
   
#     def load(self):
#         # 모델 불러오기
#         self.policy_old.load_state_dict(torch.load(self.save_path, map_location=lambda storage, loc: storage))
#         self.policy.load_state_dict(torch.load(self.save_path, map_location=lambda storage, loc: storage))
#         print(self.save_path + " loaded.")
        
        
        
#     def get_last_metrics(self):
#         """
#         마지막 업데이트 단계에서 기록한 메트릭들을 반환합니다.
#         """
#         if self.last_loss is None:
#             raise ValueError("No metrics available. Make sure to run update() first.")
        
#         return {
#             "loss": self.last_loss,
#             "policy_loss": self.policy_loss,
#             "value_loss": self.value_loss,
#             "entropy_bonus": self.entropy_bonus
#         }



