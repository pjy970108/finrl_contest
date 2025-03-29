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
        self.device = config.device
        
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
    
    def all_update(self, global_step):
        """
        모든 에이전트 업데이트 및 최고 성능 에이전트 선정
        """

        
        agent_returns = []
        agent_losses = []
        # 각 에이전트 업데이트
        for agent_idx, agent in enumerate(self.agents):
            reward_dis, avg_loss = agent.update(agent_idx)
            
            # 각 에이전트의 평균 리턴 계산
            # returns = sum(agent.buffer.rewards) / len(agent.buffer.rewards) if agent.buffer.rewards else 0
            wandb.log({
                f"Agent_{agent_idx}/Policy_Loss_Avg_Step": avg_loss["policy_loss"],
                f"Agent_{agent_idx}/Value_Loss_Avg_Step": avg_loss["value_loss"],
                f"Agent_{agent_idx}/Entropy_Bonus_Avg_Step": avg_loss["entropy_bonus"],
                f"Agent_{agent_idx}/Total_Loss_Avg_Step": avg_loss["total_loss"],
                f"Agent_{agent_idx}/Reward_AVG": avg_loss["total_reward"],  # ✅ 추가
                "Step": global_step
            })
            # agent_returns.append((reward_dis[0]))  # agent_returns에 policy_loss 저장
            agent_returns.append(avg_loss["total_reward"])  # ← 이전엔 reward_dis[0]이었지?

            agent_losses.append(avg_loss["total_reward"])
            # global_step += 1  # Step 증가

            # wandb에 각 에이전트의 성능 기록
            # wandb.log({
            #     f"agent_{agent_idx}_return": agent.buffer.rewards
            #     # f"agent_{agent_idx}_loss": agent.last_loss
            # })
            
            # agent.buffer.clear()
        # 최고 성능 에이전트 선정
        agent_returns_tensor = torch.tensor(agent_returns, device=self.device)  # 리스트를 텐서로 변환
        current_best_idx = torch.argmax(agent_returns_tensor).item()  # 인덱스 추출
        best_agent_loss = agent_losses[current_best_idx]
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
        
        return agent_returns, best_agent_loss


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


