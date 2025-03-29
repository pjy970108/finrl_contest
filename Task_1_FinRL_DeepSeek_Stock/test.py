import sys
import os



root_list = ["/data_preprocessing/", "/TARN_MAPPO"]
for root in root_list:
    sys.path.append(os.getcwd() + root)
sys.path.append(os.getcwd())

# root_list = ["/data_preprocessing", "/TARN_PPO"]
# for root in root_list:
#     sys.path.append(os.getcwd() + root)
# sys.path.append('/workspace')
# os.chdir('/workspace')
import pandas as pd
import numpy as np
# from data_preprocessing.preprocessing_config import *
# from preprocessing_config import *
# from utils import *
# root_list = ["/data_preprocessing", "/TARN_MAPPO"]
# for root in root_list:
#     sys.path.append(os.getcwd() + root)
# sys.path.append('/workspace')
# os.chdir('/workspace')
import dynamic_portfolio as dp
import data_preprocessing.technical_indicators as ti
import data_preprocessing.main_preprocessing as preprop
from TARN_MAPPO.run import *
from TARN_PPO.agent import PPO
import copy
from TARN_PPO.enviroment import EnvConfig, Stock_Env
import TARN_PPO.backtesting as backtest
# import wandb
import pickle
import random
import copy

import torch




if __name__ == "__main__":


    # root_list = ["/base_line_model/Task_1_FinRL_DeepSeek_Stock/data_preprocessing/", "/base_line_model/Task_1_FinRL_DeepSeek_Stock/TARN_MAPPO"]
    # for root in root_list:
    #     sys.path.append(os.getcwd() + root)
    # sys.path.append(os.getcwd())

    def set_seed(seed):
        # random.seed(seed)
        # np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)  # Multi-GPU 환경 고려
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        print(f"Random seed set to: {seed}")

    SEED = 42
    set_seed(SEED)



    test_scaled_data_path = "./base_line_model/Task_1_FinRL_DeepSeek_Stock/data/test_daily_v1.pkl"
    test_dataset_path = "base_line_model/Task_1_FinRL_DeepSeek_Stock/data/test_v1.csv"
    test_scaled_tensor_path = "base_line_model/Task_1_FinRL_DeepSeek_Stock/data/test_daily_data_v1.pt"

    with open(test_scaled_data_path, "rb") as f:
        test_data = pickle.load(f)
        
    test_dataset = pd.read_csv(test_dataset_path, index_col=0)
    test_dataset.set_index("date", inplace=True)

    test_scaled_tensor = torch.load(test_scaled_tensor_path)

    TRAIN_START_DATE = '2013-01-01'
    TRAIN_END_DATE = '2018-12-31'
    TRADE_START_DATE = '2019-01-01'
    TRADE_END_DATE = '2023-12-31'
    test_dataset = test_dataset[(test_dataset.index > TRADE_START_DATE) &(test_dataset.index < TRADE_END_DATE)]

    feature_dim = 16


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
            "final_conv_config": {"in_channels" : 16, "out_channels": 8, "kernel_size": 20, "stride": 1, "padding": 0}
        }
    hyperparameters = get_hyperparameters(feature_dim)
    
    
    model_path = "base_line_model/Task_1_FinRL_DeepSeek_Stock/TARN_MAPPO/model/low_risk_top10"
    all_files_and_folders = os.listdir(model_path)
    filtered_items = [item for item in all_files_and_folders if (item != 'old') & (item != 'fig')& (item != 'TD3')]
    # config_ex = EnvConfig(test_scaled_tensor, test_dataset, test_data, hyperparameters)
    # env_ex = Stock_Env(config_ex)
    # set_seed(SEED)
    
    config = EnvConfig(test_scaled_tensor, test_dataset, hyperparameters)
    env_test = Stock_Env(config, eval = True)
    config.save_path = model_path + "/" + filtered_items[0]
    # env_test.max_step = config.real_data.index.nunique() - env_test.window_size
    # print(env_test.max_step)
    agent = PPO(config)
    agent.load()
    rewards, n_rewards, all_weight_invest_rewards, asset_weight_li, top3_action_li, attn_li, empty_df, all_weight_df, asset_weight_df = eval_env(env_test, agent)
