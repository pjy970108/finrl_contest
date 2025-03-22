import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import numpy as np
import dynamic_portfolio as dp
import TARN_PPO.backtesting as backtest
import torch
import copy 

class EnvConfig:
    def __init__(self, states, real_df, real_numpy_data, hyperparameters):
        self.env_name = "Portfolio Allocation"
        # self.window_size = 22
        # self.rolling_step = 0

        # self.states = states[str(self.rolling_step)] # 한달 기준으로 자르는 코드
        # self.real_data = real_data[self.rolling_step]
        self.states = states # 한달 기준으로 자르는 코드
        self.real_data = real_df
        self.real_numpy_data = real_numpy_data
        self.invest_money = 10000000
        self.epochs = 1
        self.log_interval = 10
        self.update_interval = 10 # 10개의 샘플로 비교함
        self.asset_weight_dict = {tick : 0 for tick in self.real_data.tic.unique()}
        self.reward_cond = "combined"
        self.dynamic_dict = {
            "GTAA": dp.cal_gtaa, 
            "DM": dp.cal_dm,  
            "PAA": dp.cal_paa, 
            "DAA": dp.cal_daa}

        #self.state_dim = states.shape[1]
        self.action_dim = len(self.dynamic_dict)
        self.asset_dim = self.states.shape[2]
        self.gamma = 0.999
        self.lr_actor = 3e-4        
        self.lr_critic = 1e-3  # Critic 학습률
        self.K_epochs = 80  # 정책 업데이트 횟수 몇번 샘플을 반복해서 업데이트 할지
        self.eps_clip = 0.2  # PPO 클리핑 범위
        self.has_continuous_action_space = True  # 연속적 행동 공간 여부
        self.action_std_init = 0.6  # 행동 표준 편차 초기값
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.num_agents = 2
        self.gamma_reward = [0.0 for _ in range(self.num_agents)]
        self.save_path = f'./TARN_MAPPO/model/mulagent_{self.num_agents}_actor_lr{self.lr_actor}_critic_lr{self.lr_critic}_gamma{self.gamma}_epochs{self.epochs}_update{self.K_epochs}_eps_clip{self.eps_clip}_ppo_model_.pth'
        self.hyperparameters = hyperparameters
        self.patience = 10
        self.entropy_weight = 0.01


class Stock_Env(EnvConfig):
    def __init__(self, config):
        # 상속받은 EnvConfig 초기화
        super().__init__(config.states, config.real_data, config.real_numpy_data, config.hyperparameters)
        # self.current_idx = 0
        self.window_size = 20
        self.env_name = "Portfolio Allocation"
        # self.window_states = self.states[str(self.current_idx)]
        self.days = list(config.real_numpy_data.keys())
        # self.window_real_data = config.real_data.loc[self.days]
        self.max_step = len(self.days)-self.window_size
        self.update_interval = len(self.days)-self.window_size
        # self.update_interval = 10

        self.save_path = config.save_path
        self.save_dict = {}
        self.save_n_weight_dict = {}
        self.all_weight_dict = {}
        # self.rewards = [self.invest_money]
        self.gamma_reward = [0.0 for _ in range(self.num_agents)]
        self.num_agents = config.num_agents

        self.idx = 0
        self.action_std_init = config.action_std_init
        self.lr_actor = config.lr_actor
        self.lr_critic = config.lr_critic
        self.K_epochs = config.K_epochs
        self.eps_clip = config.eps_clip
        self.gamma = config.gamma
        self.reward_cond = config.reward_cond
        self.K_epochs = config.K_epochs
        self.entropy_weight = config.entropy_weight
        self.epochs = config.epochs

        # 자산/보상 관리 (에이전트마다 독립)
        self.rewards = [[config.invest_money] for _ in range(self.num_agents)]
        self.asset_weight_dict_list = [
            {tick: 0.0 for tick in config.real_data.tic.unique()}
            for _ in range(self.num_agents)
        ]
        self.invest_money_list = [10000000 for _ in range(self.num_agents)]
        
        
        
    def reset(self):
        """에이전트 수만큼 초기 상태를 반환"""
        self.idx = 0
        # 투자금/weights 초기화
        self.rewards = [[10000000] for _ in range(self.num_agents)]

        self.invest_money_list = [10000000 for _ in range(self.num_agents)]
        self.asset_weight_dict_list = [
            {tick: 0.0 for tick in self.real_data.tic.unique()}
            for _ in range(self.num_agents)
        ]
        # window_states 업데이트
        self.gamma_reward = [0.0 for _ in range(self.num_agents)]

        # 첫 상태 obs: (n_agents, feature_dim) - 모든 에이전트가 같은 관측 공유
        first_state = self.states[self.idx]   # shape (C, H, W)? or (feature_dim, ...)?
        # # 필요하다면 reshape해서 [feature_dim] 형태로
        # # 예: (C,H,W) -> flatten -> (feature_dim,)
        # if len(first_state.shape) > 1:
        #     # 예: (C,H,W) -> flatten
        #     flat_state = first_state.flatten()
        # else:
        #     flat_state = first_state  # already 1D

        # 모든 에이전트에게 동일 관측
        # obs = np.tile(first_state.cpu().numpy(), (self.num_agents, 1))
        # obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        return first_state  # shape: [n_agents, feature_dim]


    # def reset(self):
    #     # 환경 초기화 함수
    #     self.idx = 0
    #     self.save_dict = {}
    #     self.rewards = []
    #     self.gamma_reward = 0.0
    #     self.invest_money = 10000000
    #     self.rewards.append(self.invest_money)
    #     return self.window_states[self.idx]
    
    def softmax(self, actions_weights, temperature):
        
        
        actions_weights = actions_weights / temperature  # Temperature Scaling 적용

        exp_x = np.exp(actions_weights - np.max(actions_weights))  # 안정적인 계산을 위한 조정
        return exp_x / exp_x.sum()
    
    def top3_action(self, actions, temperature, select_action=False):
        # actions에서 상위 3개의 인덱스를 반환하는 함수
        top3_action = self.softmax(actions, temperature)
        # print("softmax_top_3", top3_action)
        if select_action:
            top3_indices = np.argsort(top3_action)[-3:]
            top3_action = np.zeros_like(actions)
            top3_action[top3_indices] = 1/3
        
        
        keys = list(self.dynamic_dict.keys())
        action_dict = {key: value for key, value in zip(keys, top3_action)}
        filtered_data = {key: value for key, value in action_dict.items() if value != 0}
        return filtered_data
    
    def calculate_action_return(self, top3_action, day, agent_idx):
        """
        top3 동적 자산배분 return을 구하는 코드입니다.
        """
        
        real_data_day = self.real_data.loc[day]
        future_days = self.days[self.idx+self.window_size]
        # print(self.days[self.idx+self.window_size])
        # future_days = list(self.days)[self.idx+1]
        future_data = self.real_data.loc[future_days]
        before_asset_dict = copy.deepcopy(self.asset_weight_dict_list[agent_idx])
        after_asset_dict = {tick : 0 for tick in self.real_data.tic.unique()}
        for strategy, portfolio_weight in top3_action.items():
            # print(self.dynamic_dict[strategy](real_data_day, future_data, strategy, after_asset_dict, portfolio_weight))
            after_asset_dict = self.dynamic_dict[strategy](real_data_day, after_asset_dict, portfolio_weight)
            # strategy_return = list(self.dynamic_dict[strategy](real_data_day, future_data, strategy, after_asset_dict, portfolio_weight).values())
            # imp_dict[strategy] = strategy_return
        backtesting_investment = dp.calculate_rebalancing_investment(real_data_day, future_data, before_asset_dict, after_asset_dict, self.invest_money)
        
        self.asset_weight_dict_list[agent_idx] = after_asset_dict
        
        return backtesting_investment


    def calculate_n_weight_action_return(self, n_weight, day):
        """
        n_weight 동적 자산배분 return을 구하는 코드입니다.
        """
        
        real_data_day = self.window_real_data.loc[day]
        future_days = list(self.days)[self.idx+1]
        future_data = self.window_real_data.loc[future_days]
        before_asset_dict = copy.deepcopy(self.n_weight_dict)
        after_asset_dict = {tick : 0 for tick in self.real_data.tic.unique()}
        for strategy, portfolio_weight in n_weight.items():
            after_asset_dict = self.dynamic_dict[strategy](real_data_day, after_asset_dict, portfolio_weight)
        backtesting_investment = dp.calculate_rebalancing_investment(real_data_day, future_data, 
                                                                     before_asset_dict, after_asset_dict, self.n_weight_money)
        
        self.n_weight_dict = after_asset_dict
        
        return backtesting_investment
    
    
    def calculate_all_weight_action_return(self, all_weight, day):
        """
        all weight 동적 자산배분 return을 구하는 코드입니다.
        """
        real_data_day = self.window_real_data.loc[day]
        future_days = list(self.days)[self.idx+1]
        future_data = self.window_real_data.loc[future_days]
        
        # 전략과 가중치 나옴
        for strategy, portfolio_weight in all_weight.items():
            before_asset_dict = copy.deepcopy(self.all_weight_dict[strategy])
            after_asset_dict = {tick : 0 for tick in self.real_data.tic.unique()}
            after_asset_dict = self.dynamic_dict[strategy](real_data_day, after_asset_dict, portfolio_weight)
            backtesting_investment = dp.calculate_rebalancing_investment(real_data_day, future_data, before_asset_dict, after_asset_dict, self.all_weight_invest_dict[strategy])
            self.all_weight_invest_dict[strategy] = backtesting_investment

            self.all_weight_dict[strategy] = after_asset_dict
            
        return self.all_weight_invest_dict



    def n_weight(self):
        n_weight_dict = {}
        keys = list(self.dynamic_dict.keys())    
        for key in keys:
            n_weight_dict[key] = 1/len(keys)
        return n_weight_dict
    
    
    def all_weight(self):
        all_weight_dict = {}
        keys = list(self.dynamic_dict.keys())    
        for key in keys:
            all_weight_dict[key] = 1.0
        return all_weight_dict
        
        
    def window_slice(self):
        self.window_states = self.states[str(self.current_idx)]
        self.days = list(self.real_numpy_data[self.current_idx].keys())
        self.max_step = len(self.days)-1
        self.window_real_data = self.real_data.loc[self.days]
        self.asset_weight_dict_list = [
            {tick: 0.0 for tick in self.real_data.tic.unique()}
            for _ in range(self.num_agents)
        ]
        # self.asset_weight_dict = {tick : 0 for tick in self.real_data.tic.unique()}
        self.invest_money_list = [10000000 for _ in range(self.num_agents)]
        self.gamma_reward = [0.0 for _ in range(self.num_agents)]

        return self.window_states[0]
        
    
    def step(self, actions):
        # 행동을 받아서 다음 상태, 보상, 에피소드 종료 여부를 반환하는 함수
        # idx에 해당하는 날짜를 받아옴
        day = self.days[self.idx]
        print(day)
        # state를 가져옴
        print(self.days[self.idx+self.window_size])
        state = self.states[self.idx]
        reward_dict = [{} for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            self.invest_money = self.invest_money_list[i]
            action_1d = actions[i]
            top3_action = self.top3_action(action_1d, 2.0)
            rebalance_investment = self.calculate_action_return(top3_action, day, i)
            self.rewards[i].append(rebalance_investment)
            self.invest_money_list[i] = rebalance_investment
        # asset_reward = sum(imp_dict.values())
            # self.rewards.append(self.invest_money)
        # self.save_dict[day] = rebalance_investment
        
        # done = (self.idx+1 >= self.max_step)
            
            if len(self.rewards[i]) < 2: # reward가 1개 이하인 경우 sharpe를 구할 수 없기에
                returns = backtest.calculate_return(self.rewards[i])

                reward_dict[i] = {'return': returns, 'sharpe': 0.0, 'sortino': 0.0, "calmar":0.0, 'mdd': 0.0, 'combined' : 0.0}
        
            else:
                returns = backtest.calculate_return(self.rewards[i])
                sharpe = backtest.calculate_sharpe_ratio(returns, risk_free_rate=0.00, annual_factor = 12)
                sortino = backtest.calculate_annualized_sortino_ratio(returns, risk_free_rate=0.00, annual_factor = 12)
                calmar = backtest.calculate_calmar_ratio(returns, annual_factor=12)
                mdd = backtest.calculate_max_drawdown(returns)
                combined = sharpe + sortino + calmar
                reward_dict[i] = {'return': returns, 'sharpe' : sharpe, 'sortino': sortino, 'calmar': calmar, 'mdd': mdd, 'combined' : combined}
                # reward_dict[i] = {'return': self.invest_money_list[i], 'sharpe' : backtest.calculate_sharpe_ratio(returns, risk_free_rate=0.00, annual_factor = 12), 
                #             'sortino': backtest.calculate_annualized_sortino_ratio(returns, risk_free_rate=0.00, annual_factor = 12),
                #             'calmar': backtest.calculate_calmar_ratio(returns, annual_factor=12),
                #             'mdd': backtest.calculate_max_drawdown(returns),
                #             'combined' : backtest.calculate_sharpe_ratio(returns, risk_free_rate=0.00, annual_factor = 12) + backtest.calculate_annualized_sortino_ratio(returns, risk_free_rate=0.00, annual_factor = 12) + backtest.calculate_calmar_ratio(returns, annual_factor=12)}
            
            # self.gamma_reward[i] = (reward_dict[i][self.reward_cond])  + self.gamma * self.gamma_reward[i]
        
        # 경쟁적 보상 계산
        # max_reward = max(self.gamma_reward)
        # min_reward = min(self.gamma_reward)
        # reward_range = max_reward - min_reward if max_reward != min_reward else 1
        
        # adjusted_rewards = []
        # for reward in self.gamma_reward:
        #     relative_performance = (reward - min_reward) / reward_range
        #     adjusted_reward = reward + relative_performance * 0.1  # 10% 보너스/페널티
        #     adjusted_rewards.append(adjusted_reward)

        # # 감마 보상 업데이트
        # for i in range(self.num_agents):
        #     self.gamma_reward[i] = adjusted_rewards[i] +self.gamma_reward[i]
        
        self.idx += 1
        done = self.idx >= self.max_step

        if done:
            print("Episode Done")
            reward_li = [reward_di[self.reward_cond] for reward_di in reward_dict]
            self.reset()
            return state, reward_li, done, self.invest_money_list, reward_dict

        else:
            day = list(self.days)[self.idx]
            state = self.states[self.idx]
            return state, [reward_di[self.reward_cond] for reward_di in reward_dict], done, self.invest_money_list, reward_dict
    
    
    def eval_step(self, weights):
        """test를 평가하는 함수입니다.

        Args:
            weights (_type_): _description_
        """
        # def set_seed(seed):
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)  # Multi-GPU 환경 고려
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False
        #     # print(f"Random seed set to: {seed}")
            
        # set_seed(42)
        
        day = list(self.days)[self.idx]
        # print(day)
        state = self.window_states[self.idx]
        
        
        n_weight_dict = self.n_weight()
        all_weight_dict = self.all_weight()
        
        
        top3_action = self.top3_action(weights, 1.0, select_action = True)
        
        rebalance_investment_top3 = self.calculate_action_return(top3_action, day)
        self.invest_money = rebalance_investment_top3
        
        
        rebalance_investment_n_weight = self.calculate_n_weight_action_return(n_weight_dict, day)
        self.n_weight_money = rebalance_investment_n_weight
        
        
        rebalance_invest_all_weight = self.calculate_all_weight_action_return(all_weight_dict, day)
        self.all_weight_invest_dict = rebalance_invest_all_weight
        

    
               
        self.rewards.append(self.invest_money)
        self.n_rewards.append(self.n_weight_money)
        
        for dm, all_weight_dict in self.all_weight_invest_dict.items():
            self.all_weight_invest_rewards[dm].append(all_weight_dict)
                

        if len(self.rewards) < 2:
            rl_reward_dict = {'return': self.invest_money, 'sharpe': 0.0, 'sortino': 0.0, "calmar":0.0, 'mdd': 0.0, 'combined' : 0.0}
            
        else:
            rl_returns = backtest.calculate_return(self.rewards)

            rl_reward_dict = {'return': rl_returns, 'sharpe' : backtest.calculate_sharpe_ratio(rl_returns, risk_free_rate=0.00, annual_factor = 12), 
                        'sortino': backtest.calculate_annualized_sortino_ratio(rl_returns, risk_free_rate=0.00, annual_factor = 12),
                        'calmar': backtest.calculate_calmar_ratio(rl_returns, annual_factor=12),
                        'mdd': backtest.calculate_max_drawdown(rl_returns),
                        'combined' : backtest.calculate_sharpe_ratio(rl_returns, risk_free_rate=0.00, annual_factor = 12) + backtest.calculate_annualized_sortino_ratio(rl_returns, risk_free_rate=0.00, annual_factor = 12) + backtest.calculate_calmar_ratio(rl_returns, annual_factor=12)}
            
            
        self.idx += 1
        done = self.idx >= self.max_step
        # self.gamma_reward = (reward_dict[self.reward_cond])  + self.gamma * self.gamma_reward
        if done:
            print("Episode Done")
            reward_li = self.rewards
            self.reset()
            return state, reward_li, done, self.n_rewards,  self.all_weight_invest_rewards

        else:
            # self.idx += 1
            day = list(self.days)[self.idx]
            state = self.window_states[self.idx]

            return state, self.rewards, done, self.n_rewards,  self.all_weight_invest_rewards
    
    
    
if __name__ == "__main__":
    import torch
    import copy
    import pandas as pd
    train_tensor = torch.load("/Users/pjy97/Desktop/hyu/research/RL/code/feature_extract/train_feature_extract.pt")
    train_dataset = pd.read_csv("/Users/pjy97/Desktop/hyu/research/RL/code/data/train_data.csv", index_col=0)
    test_tensor = torch.load("/Users/pjy97/Desktop/hyu/research/RL/code/feature_extract/train_feature_extract.pt")
    env = copy.deepcopy(Stock_Env(train_tensor, train_dataset))
