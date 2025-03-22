import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from data_preprocessing.preprocessing_config import *
import dynamic_portfolio as dp
from sklearn.preprocessing import MinMaxScaler
import data_preprocessing.technical_indicators as ti

# def data_preprocessing(data, args):
#     """
#     데이터 전처리 함수 입니다.
#     input
    
#     """
#     # 전략별 수익률을 저장할 딕셔너리
#     train_data = data[data.index <= args.train_end_date]
#     test_data = data[data.index > args.train_end_date]
#     train_data.index = train_data.index.strftime("%Y-%m-%d")
#     test_data.index = test_data.index.strftime("%Y-%m-%d")
#     # strategy_60_40_dict = {}
#     # strategy_90_10_dict = {}
#     # strategy_gtaa_5_dict = {}
#     # strategy_gtaa_aggressive_dict = {}
#     # strategy_vaa_g4_dict = {}
#     # strategy_vaa_g12_dict = {}
#     # strategy_DM_dict = {}
#     # strategy_CDM_dict = {}
#     # strategy_PAA_dict = {}
#     # strategy_daa_dict = {}
#     # for day in data.index.unique():
#     #     day_data = data.loc[day]
#     #     strategy_60_40_dict = dp.cal_strategy_60_40(day_data, day, strategy_60_40_dict)
#     #     strategy_90_10_dict = dp.cal_strategy_90_10(day_data, day, strategy_90_10_dict)
#     #     strategy_gtaa_5_dict = dp.cal_gtaa_5(day_data, day, strategy_gtaa_5_dict)
#     #     strategy_gtaa_aggressive_dict = dp.cal_gtaa_aggressive(day_data, day, strategy_gtaa_aggressive_dict)
#     #     strategy_vaa_g4_dict = dp.cal_vaa_g4(day_data, day, strategy_vaa_g4_dict)
#     #     strategy_vaa_g12_dict = dp.cal_vaa_g12(day_data, day, strategy_vaa_g12_dict)
#     #     strategy_DM_dict = dp.cal_DM(day_data, day, strategy_DM_dict)
#     #     strategy_CDM_dict = dp.cal_CDM(day_data, day, strategy_CDM_dict)
#     #     strategy_PAA_dict = dp.cal_paa(day_data, day, strategy_PAA_dict)
#     #     strategy_daa_dict = dp.cal_daa(day_data, day, strategy_daa_dict)
#     # # 전략별 수익률을 DataFrame으로 변환
#     # strategy_60_40_df = _dict_to_df(strategy_60_40_dict, "60_40")
#     # strategy_90_10_df = _dict_to_df(strategy_90_10_dict, "90_10")
#     # strategy_gtaa_5_df = _dict_to_df(strategy_gtaa_5_dict, "gtaa_5")
#     # strategy_gtaa_aggressive_df = _dict_to_df(strategy_gtaa_aggressive_dict, "gtaa_aggressive")
#     # strategy_vaa_g4_df = _dict_to_df(strategy_vaa_g4_dict, "vaa_g4")
#     # strategy_vaa_g12_df = _dict_to_df(strategy_vaa_g12_dict, "vaa_g12")
#     # strategy_DM_df = _dict_to_df(strategy_DM_dict, "DM")
#     # strategy_CDM_df = _dict_to_df(strategy_CDM_dict, "CDM")
#     # strategy_PAA_df = _dict_to_df(strategy_PAA_dict, "PAA")
#     # strategy_daa_df = _dict_to_df(strategy_daa_dict, "daa")
#     # 전략별 수익률을 하나의 DataFrame으로 합치기
#     # strategy_df = pd.concat([strategy_60_40_df, strategy_90_10_df, strategy_gtaa_5_df, strategy_gtaa_aggressive_df, strategy_vaa_g4_df, strategy_vaa_g12_df, strategy_DM_df, strategy_CDM_df, strategy_PAA_df, strategy_daa_df], axis=1)
#     return train_data, test_data

# def _dict_to_df(startegy_dict, column):
#     """
#     dictionary를 DataFrame으로 변환하는 함수입니다.
    
#     Args:
#         startegy_dict : 전략별 수익률이 저장된 딕셔너리
#         column : DataFrame의 column 이름
    
#     Returns:
#         df : 전략별 수익률이 저장된 DataFrame
#     """
#     df = pd.DataFrame.from_dict(startegy_dict, orient='index', columns=[column])
#     return df


def min_max_data(train_data, test_data):
    """
    변수에 대한 min-max scaling을 수행하는 함수입니다.
    
    Args:
        train_data : 학습 데이터
        test_data : 테스트 데이터
    """
    # scaler = MinMaxScaler()
    # scaler.fit(train_data.drop(columns=["TICKER", "weekday", "RET", "PRC"]))
    
    # train_data_scaled = scaler.transform(train_data.drop(columns=["TICKER", "weekday", "RET", "PRC"]))
    # test_dataset_scaled = scaler.transform(test_data.drop(columns=["TICKER", "weekday", "RET", "PRC"]))
    # train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.drop(columns=["TICKER", "weekday", "RET", "PRC"]).columns)
    # test_dataset_scaled = pd.DataFrame(test_dataset_scaled, columns=test_data.drop(columns=["TICKER", "weekday", "RET", "PRC"]).columns)
    
    
    # # TICKER, weekday를 다시 추가합니다.
    # train_data_scaled["TICKER"] = train_data["TICKER"].values
    # train_data_scaled["weekday"] = train_data["weekday"].values
    # test_dataset_scaled["TICKER"] = test_data["TICKER"].values
    # test_dataset_scaled["weekday"] = test_data["weekday"].values
    
    # train_data_scaled.index = train_data.index
    # test_dataset_scaled.index = test_data.index
    
    # return train_data_scaled, test_dataset_scaled, scaler
    # MinMaxScaler를 VOL 열에 대해서만 적용
    scaler = MinMaxScaler()
    scaler.fit(train_data[["VOL"]])

    # VOL 열만 변환
    train_data_scaled = train_data.copy()
    test_dataset_scaled = test_data.copy()

    train_data_scaled["VOL"] = scaler.transform(train_data[["VOL"]])
    test_dataset_scaled["VOL"] = scaler.transform(test_data[["VOL"]])

    return train_data_scaled, test_dataset_scaled, scaler






def main_data_preprocessing(args, data):
    """
    전처리를 수행하는 코드입니다.
    input : data (DataFrame)
    output : data (DataFrame)
    """
    # 실제값을 가져옴
    real_train_data = data[data.date <= args.train_end_date]
    real_test_data = data[data.date > args.train_end_date]
    real_train_data.index = pd.to_datetime(real_train_data['date']).dt.strftime("%Y-%m-%d")
    real_test_data.index = pd.to_datetime(real_test_data['date']).dt.strftime("%Y-%m-%d")

    
    # 데이터 전처리
    data = cal_ratio(data)
    data = scale_price(data, args.use_col) # 대창시우 교수님의 (Re-)Imag(in)ing Price Trends 논문으로 normalize
    data["date"] = pd.to_datetime(data["date"])
    data["weekday"] = data["date"].dt.weekday
    data.set_index("date", inplace=True)
    total_df = ti.preprocess_data(data)
    
    train_dataset = total_df[total_df.index <= args.train_end_date]
    test_dataset = total_df[total_df.index > args.train_end_date]
    train_dataset.index = train_dataset.index.strftime("%Y-%m-%d")
    test_dataset.index = test_dataset.index.strftime("%Y-%m-%d")
    
    # train_dataset, test_dataset, strategy_df = data_preprocessing(total_df, args)
    
    if args.save_data:
        train_dataset.to_csv("./data/train_data_v4.csv")
        test_dataset.to_csv("./data/test_data_v4.csv")
        # strategy_df.to_csv("./data/strategy_df.csv")
        
    # 만약 저장된 파일이 있으면 불러온다.
    # if train_path is not None:
    #     train_dataset = pd.read_csv(train_path)
    #     test_dataset = pd.read_csv(test_path)
        # strategy_df = pd.read_csv(strategy_path)
            
    train_scaled, test_scaled, scaler = min_max_data(train_dataset, test_dataset)
    
    if args.save_data:
        train_scaled.to_csv("./data/train_scaled_v4.csv")
        test_scaled.to_csv("./data/test_scaled_v4.csv")
        # strategy_df.to_csv("./data/strategy
    
    return train_scaled, test_scaled, train_dataset, test_dataset, scaler


if __name__ == "__main__":
    import pandas as pd
    from preprocessing_config import *

    args = Preprocessing_Arguments()
    data = args.filter_data()
    train_scaled, test_scaled, real_train_data, real_test_data, scaler = main_data_preprocessing(args, data)
    print("pass")