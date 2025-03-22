import torch
import copy
import pandas as pd
from TARN_MAPPO.enviroment import EnvConfig, Stock_Env
import numpy as np
from TARN_MAPPO.agent import MAPPO, EarlyStopping
import copy
from tqdm import tqdm
import wandb
import time
import torch
import numpy as np


def train_daily_env(train_dataset, config):
    """
    Main function to train and evaluate the competitive MAPPO agents.

    Args:
        env_config: Configuration for the Stock_Env.
        num_episodes (int): Number of training episodes.

    Returns:
        None
    """

    # Initialize environment
    env = Stock_Env(config)
    early_stopping_reward = EarlyStopping(patience=config.patience, min_delta=0.0001)

    print("학습 환경 및 에이전트 생성 완료")
    print("학습 시작")
    # Initialize agents
    agent = MAPPO(env)

    # Optimizers for each agent
# Optimizers for each agent
    # optimizer = {
    #         "actor_optimizer": torch.optim.Adam(
    #             [param for actor in agent.actors for param in actor.parameters()],
    #             lr=env.lr_actor,
    #         ),
    #         "critic_optimizer": torch.optim.Adam(
    #             list(agent.critic.parameters()),
    #             lr=env.lr_critic,
    #         ),
    #     }


    # Training loop
    update_step = 0

    for epoch in tqdm(range(config.epochs + 1)):

    # 매 epoch마다 메인 에피소드들 돌면서 학습
        state = env.reset()
        global_time_step = 0  # 전체 타임스텝 카운트
        # for main_episode in range(2):                  
        # state = env.window_slice() # 스테이트 짤림, day짤림, max_step 조정됨.
        state = state.permute(1, 2, 0)
        state = torch.tensor(state, dtype=torch.float32).to(config.device)
                  
        total_sub_rewards = [[] for _ in range(env.num_agents)]
        time_step = 0
        # sub_epoch_rewards = [0] * env.num_agents
        for day_step in range(env.max_step):
            actions = agent.select_actions(state)
            actions = torch.tensor(actions).cpu().numpy()

            next_state, rewards, done, invest_money_list, reward_dict = env.step(actions)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(config.device)
            
            agent.store_reward(rewards, [done] * env.num_agents)
            next_state = next_state.permute(1, 2, 0)
            state = torch.tensor(next_state, dtype=torch.float32).to(config.device)
            # for reward in rewards:
            #     if reward > 20:
            #         print(f"reward: {reward}")
            # sub_epoch_reward += reward.item()
            for i in range(env.num_agents):
                # sub_epoch_rewards[i] += rewards[i].item()
                total_sub_rewards[i].append(rewards[i].item())
                # wandb.log({
                #     f"agent_{i}_sub_episode_reward": rewards[i].item(),
                #     "time_step": global_time_step})
            # sub_epoch_reward += np.mean(rewards)
            time_step += 1
            global_time_step += 1

            # total_sub_reward.append(sub_epoch_reward) # 서브에피소드 별 보상 기록
            # 6) 일정 스텝마다 PPO 업데이트
            # if time_step % config.update_interval == 0:
            #     agent.update()
            #     time_step = 0

            # if done:
            #     break
            if time_step % env.update_interval == 0:
                
                torch.set_grad_enabled(True)
                agent_returns =  agent.all_update()
                torch.set_grad_enabled(False)
                
                # 각 에이전트의 손실 기록
                for i in range(env.num_agents):
                    
                    last_metrics = agent.agents[i].get_last_metrics()
                    wandb.log({
                        **{f"agent_{i}_{key}_per_sub_episode": value for key, value in last_metrics.items()},
                        "time_step": update_step
                    })
                time_step = 0
                update_step += 1

            if done:
                break
            
        # main_episode_rewards.append(agent_returns)
        # main_episode_rewards.append(total_sub_reward)


        # 전체 에이전트의 평균 보상 계산

        # avg_main_episode_rewards = []
        # for agent_idx in range(env.num_agents):
        #     # mian_episode_rewards[list] : [[sub_episode1_agent1, sub_episode2_agent1, ...], [sub_episode1_agent2, sub_episode2_agent2, ...], ...]
        #     # 에이전트 별 서브 에피소드의 reward
        #     agent_rewards = [episode[agent_idx] for episode in main_episode_rewards] 
        #     avg_reward = np.mean(agent_rewards)
        #     avg_main_episode_rewards.append(avg_reward)

        #     wandb.log({
        #         f"agent_{agent_idx}_main_episode_reward": avg_reward,
        #         "epoch": epoch
        #     })

        
        # 최고 성능 에이전트의 보상으로 early stopping 체크
        best_reward = max(agent_returns) # 에이전트 중 가장 높은 보상
        early_stopping_reward(best_reward)

            
        # avg_main_episode_reward = np.mean([np.mean(x) for x in main_episode_rewards]) # 전체 에이전트의 평균 보상
        # wandb.log({"main_episode_reward": avg_main_episode_reward, "epoch": epoch})
        
        # early_stopping_reward(avg_main_episode_reward)
        
        if early_stopping_reward.early_stop:
            print(f"Early stopping triggered by reward at epoch {epoch}. "
                  f"Best Reward: {best_reward:.2f}")
            break

        # # 주기적으로 로그 출력
        # if epoch % config.log_interval == 0:
        #     print(f"Epoch {epoch}/{config.epochs}")
        #     for i, avg_reward in enumerate(avg_main_episode_rewards):
        #         print(f"Agent {i} Average Reward: {avg_reward:.2f}")
        #     print(f"Best Agent: {agent.best_agent_idx}, Best Reward: {best_reward:.2f}")

        # 주기적으로 최고 성능 에이전트 모델 저장
        if epoch % 10 == 0:
        # if epoch % 1 == 0:

            agent.save()
            print(f"Best agent model saved at epoch {epoch}")
    agent.save()
    
    return agent


def train_competitive_ippo(env, agents, config):
    """
    CompetitiveMultiAgentEnv에서,
    n_agents개의 독립 PPO(agents)를 동시에 학습.
    """
    n_agents = config.num_agents
    max_episodes = config.epochs  # 예: 에폭=에피소드
    rollout_length = 100  # 예: 한 에피소드 or fixed step 수

    for episode in range(max_episodes):
        obs = env.reset()  # shape [n_agents, obs_dim]
        done = np.array([False]*n_agents)
        step_count = 0

        # 각 에이전트 buffer clear
        for i in range(n_agents):
            agents[i].buffer.clear()

        while not done.any() and step_count < rollout_length:
            actions = []
            for i in range(n_agents):
                # obs[i] shape [obs_dim]
                obs_i = obs[i].cpu().numpy()  
                action_i = agents[i].select_action(obs_i)  # store in agent buffer
                actions.append(action_i)

            # actions shape => [n_agents, action_dim]
            actions_t = torch.tensor(actions, dtype=torch.float32)

            next_obs, rewards, dones, info = env.step(actions_t)
            # store reward/done in each agent's buffer
            for i in range(n_agents):
                agents[i].store_reward_done(rewards[i], dones[i])

            obs = next_obs
            done = dones
            step_count += 1

        # 한 에피소드가 끝나거나 rollout_length 도달 => PPO update
        for i in range(n_agents):
            agents[i].update()

        # logging
        print(f"Episode {episode} done.")
    return agents
    

def eval_env(env, agent):
    
    """
    테스트 기간동안 평가하기 위한 코드입니다.
    """
    # print("평가 환경 생성")
    # config = EnvConfig(test_tensor, test_dataset)
    # env = copy.deepcopy(Stock_Env(test_tensor, test_dataset, config))
    # print("평가 시작")
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).to(env.device)
    rewards = []
    asset_weight_li = []
    top3_action_li = []
    agent.policy.eval()         # 평가 모드 전환
    agent.policy_old.eval()     # (필요하다면) old policy도 평가 모드로
    attn_li = []
    for _ in range(env.max_step):
        state = state.permute(1, 2, 0)
        state = torch.tensor(state, dtype=torch.float32).to(env.device)
        with torch.no_grad():
            action, attn_score = agent.select_action(state,  deterministic=True)
            attn_li.append(attn_score)
        next_state, rewards, done, n_rewards,  all_weight_invest_rewards, asset_weight_dict, top3_action = env.eval_step(action)
        asset_weight_li.append(asset_weight_dict)
        top3_action_li.append(top3_action)
        state = torch.tensor(next_state, dtype=torch.float32).to(env.device)
        # rewards.append(reward_dict["return"])
        if done:
            break
    return rewards, n_rewards, all_weight_invest_rewards, asset_weight_li, top3_action_li, attn_li


if __name__ == "__main__":
    import torch
    import copy
    import pandas as pd
    from enviroment import EnvConfig, Stock_Env
    import numpy as np
    from agent import PPO
    import pickle
    # pickle 파일을 불러오는 방법
    with open("./data/train_real_data.pkl", "rb") as f:
        train_dataset_episode = pickle.load(f)
    
    
    train_dataset = pd.read_csv("./data/train_data.csv", index_col=0)

    
    feature_dim = 35


    train_scaled_tensor = torch.load("./data/all_episodes.pt")
    
    # --- Hyperparameters ---
    def get_hyperparameters(feature_dim):
        return {
            "dcc_dropout": 0.2,
            "sac_dropout": 0.01,
            "sac_heads": 2,
            "ddc_configs": [
                {"in_channels": feature_dim, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": (3-1)//2, "dilation": 1, "sac_scale": 4**0.5, "residual_out_channels" : 8, "residual_kernal": 1},
                {"in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 2, "dilation": 2, "sac_scale": 8**0.5, "residual_out_channels" : 16, "residual_kernal": 1 },
                {"in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 4, "dilation": 4, "sac_scale": 8**0.5}
            ],
            "final_conv_config": {"in_channels" : 16, "out_channels": 8, "kernel_size": 31, "stride": 1, "padding": 0}
        }

    hyperparameters = get_hyperparameters(feature_dim)

    

    train_agent = train_main_sub_env(train_scaled_tensor, train_dataset, train_dataset_episode, hyperparameters)
    
    # rewards = eval_env(test_tensor, test_dataset, train_agent)
