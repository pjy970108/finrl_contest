# config for datapreprocessing
from utils import data_loader
import pandas as pd
import numpy as np


class Preprocessing_Arguments:
    def __init__(self):
        self.data_file = "data/etf.csv" # ETF 파일을 불러오는 경로
        self.save_file = "data/etf_preprocessed.csv"
        self.delist_tic = ['PXD', 'ROIL', "MTUM"]
        self.use_col = ["date", "TICKER",  "VOL", "RET","adjust_low", "adjust_high", "adjust_close", "adjust_open", "PRC"]
        self.data = data_loader(self.data_file)
        # Data period
        self.start_date = "2007-10-05"
        self.end_date = "2023-12-29"
        self.train_end_date = "2018-06-30"
        self.test_end_date = "2022-12-31"
        self.save_data = True
    
    def filter_data(self):
        # 필요한 column과 일자를 필터링하는 코드입니다.
        self.data = self.data[~self.data['TICKER'].isin(self.delist_tic)]
        ### 데이터 QQQQ -> QQQ 티커 변경
        self.data.loc[self.data["TICKER"]=="QQQQ", "TICKER"] = "QQQ"
        self.data.loc[self.data["RET"] == "C", "RET"] = 0
        self.data["RET"] = self.data["RET"].astype(float)
        tot_data = self.data[self.data["date"] >= self.start_date]
        tot_data = tot_data.reset_index(drop=True)
        return tot_data
    
def cal_ratio(total_data):
    ###
    # Calculate the ratio of OHL price to close price
    ###
    total_data["open_ratio"] = total_data["OPENPRC"] / total_data["PRC"]
    total_data["high_ratio"] = total_data["ASKHI"] / total_data["PRC"]
    total_data["low_ratio"] = total_data["BIDLO"] / total_data["PRC"]
    return total_data
    

def _make_price(data, rets):
    ###
    # Calculate close price from return
    
    # data: DataFrame(티커별 데이터 프레임)
    # rets: 티커벌 수익률 리스트(list)
    ###
    
    # 첫번째 close 값
    return_close = [1] # 첫날 가격 t=0
    
    for ret in rets[1:]: # 둘쨋날 정규화를 위한 수익률 RET : t=1
        price_t_1 = return_close[-1]
        price_t_1 = price_t_1 * ret
        return_close.append(price_t_1)
    return return_close    


def _adjust_ohl(data):
    ###
    # Adjust the OHL price with the ratio
    ###
    data["adjust_open"] = data["open_ratio"] * data["adjust_close"]
    data["adjust_high"] = data["high_ratio"] * data["adjust_close"]
    data["adjust_low"] = data["low_ratio"] * data["adjust_close"]
    return data
    
    
# def scale_price(data, use_col):
#     ###
#     # Scale the price data
#     # 대창시우 교수님의 (Re-)Imag(in)ing Price Trends 논문
#     ###
#     tickers = data["TICKER"].unique()
#     for tic in tickers:
#         tic_data = data[data["TICKER"] == tic]
#         return_1= 1+tic_data["RET"]
#         return_close = _make_price(tic_data, return_1)
#         # Tick 데이터를 data에 업데이트
#         data.loc[data["TICKER"] == tic, "adjust_close"] = return_close
#     data = _adjust_ohl(data)
#     return data[use_col]


def scale_price(data, use_col):
    ###
    # Scale the price data using vectorized operations
    ###
    data = data.copy()  # 원본 데이터 보호
    data["adjust_close"] = data.groupby("TICKER")["RET"].transform(lambda x: (1 + x).cumprod())
    data = _adjust_ohl(data)
    return data[use_col]


# def scale_price(data, use_col):
#     ###
#     # Scale the price data using vectorized operations
#     ###
#     data["adjust_close"] = (1 + data.groupby("TICKER")["RET"]).transform("cumprod")
#     data = _adjust_ohl(data)
#     return data[use_col]