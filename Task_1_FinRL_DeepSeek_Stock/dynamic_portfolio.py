import numpy as np
import pandas as pd
## 정적 자산 전략입니다.
# 만들어야할 코드

## 동적 자산 전략입니다.
def cal_gtaa(data, before_asset_dict, portfolio_weight):
    """GTAA 코드입니다.
    day 기준 SMA 10달전 가격과 비교하여 양수인것들중 volume이 높은 자산 top 10개를 뽑습니다.
    만약 그 이하라면 남은 weight를 cash에 투자합니다.
    Args:
        data : day기준으로 slicing된 데이터입니다
        before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
        portfolio_weight : 전략의 weight입니다.
    Returns:
        before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
    """
    data_imp = data.copy()
    data_imp = data_imp[data_imp["close"] > data_imp["SMA_10m"]]
    data_imp = data_imp.sort_values(by="volume", ascending=False).head(10)
    weights = {ticker : 1/10 for ticker in data_imp["tic"]}
    
    for ticker in weights.keys():
        before_asset_dict[ticker] += weights[ticker]*portfolio_weight
    return before_asset_dict


def cal_dm(data, before_asset_dict, portfolio_weight):
    """듀얼 모멘텀 전략 코드입니다.
    day기준 12개월 누적 수익률이 무위험 수익률보다 높은 자산에 투자함
    수익률이 높은 자산 top 10개를 뽑습니다.
    만약 그 이하라면 현금을 보유합니다. 
    Args:
        data : day기준으로 slicing된 데이터입니다
        before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
        portfolio_weight : 전략의 weight입니다.
    Returns:
        before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
        
    """
    data_imp = data.copy()
    data_imp = data_imp[data_imp["momentum_product"] > 1]
    data_imp = data_imp.sort_values(by="momentum_product", ascending=False).head(10)
    weights = {ticker : 1/10 for ticker in data_imp["tic"]}
    for ticker in weights.keys():
        before_asset_dict[ticker] += weights[ticker]*portfolio_weight
    return before_asset_dict


def cal_daa(data, before_asset_dict, portfolio_weight):
    """daa 전략 코드입니다.
    day 기준 모멘텀 스코어가 높은 자산에 투자함 수익률이 높은 자산 top 10개를 뽑습니다.
    만약 그 이하라면 현금을 보유합니다. 
    Args:
        data : day기준으로 slicing된 데이터입니다
        before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
        portfolio_weight : 전략의 weight입니다.
    Returns:
        before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
        
    """
    data_imp = data.copy()
    data_imp = data_imp[data_imp["mom_score"] > 0]
    data_imp = data_imp.sort_values(by="mom_score", ascending=False).head(10)
    weights = {ticker : 1/10 for ticker in data_imp["tic"]}
    for ticker in weights.keys():
        before_asset_dict[ticker] += weights[ticker]*portfolio_weight
    return before_asset_dict


def cal_paa(data, before_asset_dict, portfolio_weight):
    """paa 전략 코드입니다.
    day 기준 paa_score가 높은 자산에 투자함 수익률이 높은 자산 top 10개를 뽑습니다.
    만약 그 이하라면 현금을 보유합니다. 
    Args:
        data : day기준으로 slicing된 데이터입니다
        before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
        portfolio_weight : 전략의 weight입니다.
    Returns:
        before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
        
    """
    data_imp = data.copy()
    data_imp = data_imp[data_imp["paa_score"] > 0]
    data_imp = data_imp.sort_values(by="paa_score", ascending=False).head(10)
    weights = {ticker : 1/10 for ticker in data_imp["tic"]}
    for ticker in weights.keys():
        before_asset_dict[ticker] += weights[ticker]*portfolio_weight
    return before_asset_dict


def cal_return(data, future_data, save_return_dict, weights):
    """"
    미래 수익률을 구하는 코드입니다.
    """
    for ticker in weights.keys():
        now_price = data[data["tic"]==ticker]["close"][0]
        future_price = future_data[future_data["tic"]==ticker]["close"][0]
        future_return = future_price/now_price - 1
        weighted_future_return = future_return * weights[ticker]
        save_return_dict[ticker] = weighted_future_return
    return save_return_dict
    

def cal_asset_weight(ticker_weights, asset_dict, portfolio_weight):
    """
    자산별 weight를 뽑는 코드입니다.
    Args:
        ticker_weights (dict): 전략에서 나온 ticker 별 weight입니다.
        asset_dict (dict): asset별 weight를 저장할 Dictionary입니다.
        portfolio_weight (float): portfolio별 weight입니다.
    """
    for ticker, weight in ticker_weights.items():
        asset_dict[ticker] += weight*portfolio_weight
    return asset_dict


def calculate_buy_sell_dict(before_asset_dict, after_asset_dict):
    """
    매수 및 매도 비중을 구하기 위한 코드입니다.
    Args:
    - before_asset_dict (dict): 전 리밸런싱 거래 비중
    - after_asset_dict (dict): 현재 리밸런싱 거래 비중
    
    Returns:
    - buy_ticker_dict (dict): 매수 비중
    - sell_ticker_dict (dict): 매도 비중
    """
    buy_ticker_dict = {}
    sell_ticker_dict = {}

    for ticker in before_asset_dict.keys():
        before_weight = before_asset_dict.get(ticker, 0)
        after_weight = after_asset_dict.get(ticker, 0)

        if after_weight > before_weight:  # 매수
            buy_ticker_dict[ticker] = after_weight - before_weight
        elif after_weight < before_weight:  # 매도
            sell_ticker_dict[ticker] = before_weight - after_weight

    return buy_ticker_dict, sell_ticker_dict


def calculate_total_transaction_fee(today_data, buy_ticker_dict, sell_ticker_dict, invest_money):
    """
    매수 및 매도에 따른 거래 비용을 계산하는 코드입니다.
    Args:
    - today_data (pd.DataFrame): 현재 날짜의 데이터
    - buy_ticker_dict (dict): 매수 비중
    - sell_ticker_dict (dict): 매도 비중
    - invest_money (float): 총 투자 금액
    
    Returns:
    - total_transaction_fee (float): 총 거래 비용
    """
    transaction_fee_rate = 0.001  # 거래 수수료 비율
    # transaction_fee_rate = 0.0  # 거래 수수료 비율
    # print("transaction_fee_rate", transaction_fee_rate)
    total_transaction_fee = 0.0

    # 매수 거래 비용 계산
    for ticker, weight in buy_ticker_dict.items():
        price = today_data[today_data["tic"] == ticker]["close"].values[0]
        investment = invest_money * weight
        invest_count = int(investment / price)  # 정수 단위로 매수 가능한 수량
        actual_investment = invest_count * price  # 실제 투자 금액
        total_transaction_fee += actual_investment * transaction_fee_rate

    # 매도 거래 비용 계산
    for ticker, weight in sell_ticker_dict.items():
        price = today_data[today_data["tic"] == ticker]["close"].values[0]
        investment = invest_money * weight
        invest_count = int(investment / price)  # 정수 단위로 매도 가능한 수량
        actual_investment = invest_count * price  # 실제 매도 금액
        total_transaction_fee += actual_investment * transaction_fee_rate

    return total_transaction_fee


def calculate_rebalancing_investment(today_data, future_data, before_asset_dict, after_asset_dict, invest_money):
    """
    리밸런싱 후 투자 결과를 계산하는 코드입니다.
    Args:
    - today_data (pd.DataFrame): 현재 날짜의 데이터
    - future_data (pd.DataFrame): 미래 날짜의 데이터
    - before_asset_dict (dict): 전 리밸런싱 거래 비중
    - after_asset_dict (dict): 현재 리밸런싱 거래 비중
    - invest_money (float): 총 투자 금액
    
    Returns:
    - asset_invest_dict (dict): 자산별 최종 투자 금액
    - remain_money (float): 남은 잔여 투자 금액
    """
    # 매수 및 매도 비중 계산
    buy_ticker_dict, sell_ticker_dict = calculate_buy_sell_dict(before_asset_dict, after_asset_dict)
    
    # 거래 비용 계산
    total_transaction_fee = calculate_total_transaction_fee(today_data, buy_ticker_dict, sell_ticker_dict, invest_money)

    # 거래 비용을 차감한 투자 가능 금액
    asset_invest_dict = {}
    remain_money = 0.0  # 잔여 투자 금액

    remain_invest_money = invest_money * (1-sum(after_asset_dict.values())) # 기존 투자금
    
    for ticker, weight in after_asset_dict.items():
        if weight == 0:
            # asset_invest_dict[ticker] = 0
            continue
        now_price = today_data[today_data["tic"] == ticker]["close"].values[0]
        future_price = future_data[future_data["tic"] == ticker]["close"].values[0]
        # future_return = future_price / now_price  # 미래 수익률

        # 티커별 투자 가능 금액 및 실제 투자 금액 계산
        ticker_investment = invest_money * weight
        invest_count = int(ticker_investment / now_price)  # 정수 단위로 매수 가능 수량
        actual_investment = invest_count * now_price  # 실제 투자 금액
        remain_money += ticker_investment - actual_investment  # 남은 잔여 금액
        
        future_investment = invest_count * future_price  # 미래 가치
        # 자산별 최종 금액 저장
        asset_invest_dict[ticker] = future_investment
    
    total_investment = sum(asset_invest_dict.values())
    backtesting_investment = total_investment - total_transaction_fee + remain_money + remain_invest_money

    return backtesting_investment






    
## 동적 자산 전략입니다.
# def cal_gtaa(data, before_asset_dict, portfolio_weight):
#     """
#     GTAA 코드입니다.
#     day 기준 SMA 10달전 가격과 비교하여 양수인것들중 volume이 높은 자산 top 10개를 뽑습니다.
#     만약 그 이하라면 남은 weight를 cash에 투자합니다.
#     day기준 21일 전의 수익률과 비교하여 수익률을 계산합니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """
    
#     data_imp = data.copy()
#     data_imp = data_imp[data_imp["close"] > data_imp["SMA_10m"]]
#     data_imp = data_imp.sort_values(by="volume", ascending=False).head(10)
#     weights = {ticker : 1/10 for ticker in data_imp["tic"]}
    
#     for ticker in weights.keys():
#         before_asset_dict[ticker] += weights[ticker]*portfolio_weight
#     return before_asset_dict



# def cal_strategy_60_40(data, before_asset_dict, portfolio_weight):
#     """6040 전략의 수익률을 만드는 코드입니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """
#     # save_return_dict = {}
#     weights = {'SPY': 0.6, 'AGG': 0.4}
    
#     before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)
        
#     return before_asset_dict




# def cal_strategy_90_10(data, before_asset_dict, portfolio_weight):
#     """9010 전략의 수익률을 만드는 코드입니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """
#     weights = {'SPY': 0.9, 'SHY': 0.1}
#     before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)
    
#     return before_asset_dict


# ## 동적 자산 전략입니다.
# def cal_gtaa_5(data, before_asset_dict, portfolio_weight):
#     """gtaa 전략을 짜는 코드입니다.

#     day기준 21일 전의 수익률과 비교하여 수익률을 계산합니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """

#     weights = {"IEF" : 0.2, "SPY" : 0.2, "VNQ": 0.2, "EFA": 0.2, "DBC" : 0.2}
#     data_imp = data.copy()
#     data_imp = data_imp[data_imp["TICKER"].isin(list(weights.keys()))]
    
#     for ticker in weights.keys():
#         now_ticker = data_imp[data_imp["TICKER"]==ticker]
#         if now_ticker["adjust_close"].iloc[0] < now_ticker["SMA_220"].iloc[0]:
#             before_asset_dict[ticker] += 0
#         else:
#             before_asset_dict[ticker] += weights[ticker]*portfolio_weight


#     return before_asset_dict
    
    
# def cal_gtaa_aggressive(data, before_asset_dict, portfolio_weight):
#     """gtaa aggressive 전략을 짜는 코드입니다.

#     day기준 21일 전의 수익률과 비교하여 수익률을 계산합니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """
#     # save_return_dict = {}

    
#     data_imp = data.copy()
#     ticker_list= ["IWD", "IWN", "IWM", "PDP", "EFA", "EEM", "IEF", "TLT", "LQD", "BWX", "DBC", "GLD", 'VNQ']
#     data_imp = data_imp[data_imp["TICKER"].isin(ticker_list)]
#     momentum_scores = data_imp.groupby("TICKER")["mom_score"].mean().to_dict()
#     top_3_assets = sorted(momentum_scores, key=momentum_scores.get, reverse=True)[:3]
#     weights = {top_3_assets[0]: 1/3, top_3_assets[1]: 1/3, top_3_assets[2]: 1/3}
    
    
    
#     for ticker in weights.keys():
#         now_ticker = data_imp[data_imp["TICKER"]==ticker]
#         if now_ticker["adjust_close"].iloc[0] < now_ticker["SMA_220"].iloc[0]:
#             before_asset_dict[ticker] += 0
            
#         else:
#             before_asset_dict[ticker] += weights[ticker]*portfolio_weight

#     return before_asset_dict


# def cal_vaa_g4(data, before_asset_dict, portfolio_weight):
#     """vaa_g4 전략을 짜는 코드입니다.
    
#     day기준 21일 전의 수익률과 비교하여 수익률을 계산합니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """
#     aggressive_assets = ['SPY', 'VEA', 'VWO', 'BND']  # 공격형 자산
#     defensive_assets = ['LQD', 'IEF', 'SHY']  # 방어형 자산
#     data_imp = data.copy()
#     aggressive_data = data_imp[data_imp["TICKER"].isin(aggressive_assets)]
#     defensive_data = data_imp[data_imp["TICKER"].isin(defensive_assets)]

        
#     if all(aggressive_data["mom_score"] >= 0): #만약 모든 공격형 자산의 모멘텀 점수가 0보다 크거나 같다면
#         # 최대 모멘텀 가져오기
#         max_mom_score = aggressive_data[aggressive_data["mom_score"] == aggressive_data["mom_score"].max()]
#     else:
#         # 방어형 자산으로 대체
#         max_mom_score = defensive_data[defensive_data["mom_score"] == defensive_data["mom_score"].max()]
    
#     weights = {asset: 1 for asset in max_mom_score["TICKER"].unique()}
#     before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#     return before_asset_dict


# def cal_vaa_g12(data, before_asset_dict, portfolio_weight):
#     """
#     vall_g12 전략을 짜는 코드입니다.
    
#     day기준 21일 전의 수익률과 비교하여 수익률을 계산합니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """

#     aggressive_assets = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "GSG", "GLD", "TLT", "LQD", "HYG"]
#     defensive_assets = ["SHY", "IEF", "LQD"]
#     data_imp = data.copy()    
#     aggressive_data = data_imp[data_imp["TICKER"].isin(aggressive_assets)]
#     defensive_data = data_imp[data_imp["TICKER"].isin(defensive_assets)]
    
#     decline_momentum = aggressive_data[aggressive_data["mom_score"] < 0]["TICKER"].nunique()
#     top_5_assets = aggressive_data.groupby("TICKER")["mom_score"].mean().sort_values(ascending=False).head(5).index
#     safe_asset_max = defensive_data[defensive_data["mom_score"] == defensive_data["mom_score"].max()]

#     if decline_momentum == 0: #하락자산이 0개
#         # 모든 위험 자산이 모멘텀이 0보다 크다면
#         # 모멘텀 상위 5개 자산에 균등 투자
#         # 모멘텀 상위 5개 자산 가져오기
#         weights = {asset: 1/5 for asset in top_5_assets}
#         # 5개 자산에 균등 투자
#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # mom_score가 0보다 큰 개수 찾기

#     elif decline_momentum == 1: # 하락자산이 1개
#         # 75% 투자
#         weights = {asset: (1/5) * 0.75 for asset in top_5_assets}
#         weights[safe_asset_max["TICKER"].unique()[0]] = 0.25
       
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)

            
#     elif decline_momentum == 2: # 하락자산이 2개
#         weights = {asset: (1/5) * 0.5 for asset in top_5_assets}
#         weights[safe_asset_max["TICKER"].unique()[0]] = 0.5

#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)


#     elif decline_momentum == 3: # 하락자산이 3개
#         weights = {asset: (1/5) * 0.25 for asset in top_5_assets}
#         weights[safe_asset_max["TICKER"].unique()[0]] = 0.75

#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

# #        save_return_dict = cal_return(data, future_data, save_return_dict, weights)

#     else:
#         weights = {asset: 1 for asset in safe_asset_max["TICKER"].unique()}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)

        
#     # weight_dict = {strategy : sum(save_return_dict.values())}
        
#     return before_asset_dict


# def cal_daa(data, before_asset_dict, portfolio_weight):
#     """
#     daa 전략을 짜는 코드입니다.
#     day기준 21일 전의 수익률과 비교하여 수익률을 계산합니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """

#     aggresive_ticker = ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "GSG", "GLD", "TLT", "HYG", "LQD"]
#     defensive_ticker = ["SHY", "IEF", "LQD"]
#     canaria_ticker = ["VWO", "BND"]
#     data_imp = data.copy()

#     aggressive_data = data_imp[data_imp["TICKER"].isin(aggresive_ticker)]
#     defensive_data = data_imp[data_imp["TICKER"].isin(defensive_ticker)]
#     canaria_data = data_imp[data_imp["TICKER"].isin(canaria_ticker)]
#     canaria_momentum = canaria_data["mom_score"] > 0
    
#     top_6_assets = aggressive_data.groupby("TICKER")["mom_score"].mean().sort_values(ascending=False).head(6).index
#     defensive_data_assets = defensive_data.groupby("TICKER")["mom_score"].mean().sort_values(ascending=False).head(1).index


#     if canaria_momentum.sum == 2:
#         # 카나리아 모멘텀이 2개 이상 양수면 위험자산에 100% 투자함
#         weights = {asset: 1/6 for asset in top_6_assets}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)

            
#     elif canaria_momentum.sum == 1:
#         # 카나리아 모멘텀이 1개 이면 위험자산에 50% 투자함
#         aggresive_weights = {asset: 0.5/6 for asset in top_6_assets}
#         defensive_weights = {defensive_data_assets[0]: 0.5}
#         before_asset_dict = cal_asset_weight(aggresive_weights, before_asset_dict, portfolio_weight)
#         before_asset_dict = cal_asset_weight(defensive_weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, aggresive_weights)
#         # save_return_dict = cal_return(data, future_data, save_return_dict, defensive_weights)

#     else:
#         # 카나리아 모멘텀이 0개 이면 안전자산에 100% 투자함
#         weights = {defensive_data_assets[0]: 1}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)
    
#     # weight_dict = {strategy : sum(save_return_dict.values())}
        
#     return before_asset_dict


# def cal_DM(data, before_asset_dict, portfolio_weight):
#     """
#     DM 전략을 짜는 코드입니다.
    
#     day기준 21일 전의 수익률과 비교하여 수익률을 계산합니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """
#     aggressive_assets = ["SPY", "EFA"]
#     defensive_assets = ["AGG"]
#     standard_assets = ["BIL"]
#     compare_assets = ["SPY"]
#     data_imp = data.copy()
#     aggressive_data = data_imp[data_imp["TICKER"].isin(aggressive_assets)]
#     defensive_data = data_imp[data_imp["TICKER"].isin(defensive_assets)]
#     # BIL 자산
#     standard_data = data_imp[data_imp["TICKER"].isin(standard_assets)]
#     # SPY 자산
#     compare_data = data_imp[data_imp["TICKER"].isin(compare_assets)]
    
#     if compare_data["return_12m"].values >= standard_data["return_12m"].values:
#         # 만약 spy가 bil보다 수익률이 높다면 공격형 자산에 투자
#         top_1_assets = aggressive_data[aggressive_data["return_12m"] == aggressive_data["return_12m"].max()].TICKER.values
        
#         weights = {asset: 1 for asset in top_1_assets}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)

#     else:
#         # 만약 spy가 bil보다 수익률이 낮다면 안전자산에 투자
#         top_1_assets = defensive_data[defensive_data["return_12m"] == defensive_data["return_12m"].max()].TICKER.values
#         weights = {asset: 1 for asset in top_1_assets}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)
        
#     # weight_dict = {strategy : sum(save_return_dict.values())}
#     return before_asset_dict
#     # return weight_dict
        

# def check_momentum(data, standard_data,  before_asset_dict, portfolio_weight):
#     """
#     day기준 21일 전의 수익률과 비교하여 수익률을 계산합니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """
#     check_mom = data["return_12m"].values >= standard_data["return_12m"].values
#     count_mom = check_mom.sum()
#     if count_mom >= 1:
#         # 만약 SHY보다 2개 자산이 높다면 공격형 자산에 투자
#         top_1_assets = data[data["return_12m"] == data["return_12m"].max()].TICKER.values
#         weights = {asset: 0.25 for asset in top_1_assets}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(data, future_data, save_return_dict, weights)

#     else:
#         # 만약 SHY보다 낮다면 방어형 자산에 투자
#         slice_data = standard_data.copy()
#         weights = {asset: 0.25 for asset in slice_data["TICKER"].values}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

#         # save_return_dict = cal_return(slice_data, future_data, save_return_dict, weights)

#     return before_asset_dict 
        

# def cal_CDM(data, before_asset_dict, portfolio_weight):
#     """
#     CDM 전략을 짜는 코드입니다.
#     Args:
#         data : day기준으로 slicing된 데이터입니다
#         before_asset_dict : 자산별 weight를 저장한 Dictionary입니다.
#         portfolio_weight : 전략의 weight입니다.
#     Returns:
#         before_asset_dict (dict): 자산별 weight를 저장한 Dictionary입니다.
#     """
#     stock_assets = ["SPY", "EFA"]
#     bond_assets = ["LQD", "HYG"]
#     reit_assets = ["VNQ", "REM"]
#     crisis_assets = ["GLD", "TLT"]
#     standard_assets = ["SHY"]
    
#     stock_data = data[data["TICKER"].isin(stock_assets)]
#     bond_data = data[data["TICKER"].isin(bond_assets)]
#     reit_data = data[data["TICKER"].isin(reit_assets)]
#     crisis_data = data[data["TICKER"].isin(crisis_assets)]
#     standard_data = data[data["TICKER"].isin(standard_assets)]
    
#     before_asset_dict = check_momentum(stock_data, standard_data, before_asset_dict, portfolio_weight)
#     before_asset_dict = check_momentum(bond_data,  standard_data, before_asset_dict, portfolio_weight)
#     before_asset_dict = check_momentum(reit_data,  standard_data, before_asset_dict, portfolio_weight)
#     before_asset_dict = check_momentum(crisis_data, standard_data, before_asset_dict, portfolio_weight)
#     # total_sum = sum(value for d in [stock_dict, bond_dict, reit_dict, crisis_dict] for value in d.values())

#     # weight_dict = {strategy : total_sum}


#     return before_asset_dict

    

# def paa_score(data):
#     data["PAA_score"] = (data["adjust_close"] / data["SMA_264"])-1
#     return data
    

# def cal_paa(data, before_asset_dict, portfolio_weight):
#     """
#     paa 전략를 짜는 코드입니다.
#     data : 데이터
#     day : 날짜
#     save_dict : 저장할 딕셔너리
    
#     return : save_dict
#     """
#     aggresive_ticker = ["SPY", "QQQ", "IWM", "VGK", "EWJ", "EEM", "IYR", "GSG", "GLD", "HYG", "LQD", "TLT"]
#     defensive_ticker = ["IEF"]
#     data_imp = data.copy()
#     aggressive_data = data_imp[data_imp["TICKER"].isin(aggresive_ticker)]
#     aggressive_data = paa_score(aggressive_data)
#     defensive_data = data_imp[data_imp["TICKER"].isin(defensive_ticker)]
#     top_6_assets = aggressive_data.groupby("TICKER")["PAA_score"].mean().sort_values(ascending=False).head(6).index

#     # 현재 상승 모멘텀인 경우
#     momentum_compare = aggressive_data["adjust_close"] > aggressive_data["SMA_264"]

#     if momentum_compare.sum() == 12: # 상승 모멘텀이 전부인 경우 100%를 위험자산에 투자
#         weights = {asset: 1/6 for asset in top_6_assets}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)

        
#     elif momentum_compare.sum() == 11: # 상승 모멘텀이 11개인 경우 83.33%를 위험자산에 투자
#         weights = {asset: 0.8333/6 for asset in top_6_assets}
#         weights["IEF"] = 0.1667
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)


#     elif momentum_compare.sum() == 10: # 상승 모멘텀이 10개인 경우 66.67%를 위험자산에 투자
#         weights = {asset: 0.6667/6 for asset in top_6_assets}
#         weights["IEF"] = 0.3333
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)
            
        
#     elif momentum_compare.sum() == 9: # 상승 모멘텀이 9개인 경우 50%를 위험자산에 투자
#         weights = {asset: 0.5/6 for asset in top_6_assets}
#         weights["IEF"] = 0.5

#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)
        
#     elif momentum_compare.sum() == 8: # 상승 모멘텀이 8개인 경우 33.33%를 위험자산에 투자
#         weights = {asset: 0.3333/6 for asset in top_6_assets}
#         weights["IEF"] = 0.6667
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)


                    
#     elif momentum_compare.sum() == 7: # 상승 모멘텀이 7개인 경우 16.67%를 위험자산에 투자
#         weights = {asset: 0.1667/6 for asset in top_6_assets}
#         weights["IEF"] = 0.8333
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)
        
#     else: # 상승 모멘텀이 6개 이하인 경우 안전자산에 투자
#         weights = {"IEF": 1}
#         before_asset_dict = cal_asset_weight(weights, before_asset_dict, portfolio_weight)
#         # save_return_dict = cal_return(data, future_data, save_return_dict, defensive_weights)
       
#     return before_asset_dict
