### 기술 지표를 생성하는 파일입니다.###
import pandas as pd


def sma(data, period):
    ### 이동평균 계산 ###
    return data.rolling(window=period).mean()


def ema(data, period):
    """
    지수이동평균을 계산합니다.

    Args:
        data (df): 데이터 프레임
        period (int): 몇일 이동 평균을 계산할지

    Returns:
        지수이동평균: _description_
    """
    return data.ewm(span=period, adjust=False).mean()


def macd(data, short_period, long_period, signal_period):
    """
    MACD를 계산합니다.

    Args:
        data (df): 데이터 프레임
        short_period (int): 단기 이동평균 기간
        long_period (int): 장기 이동평균 기간
        signal_period (int): 신호선 기간

    Returns:
        MACD: _description_
    """
    
    short_ema = ema(data, short_period) # 12일
    long_ema = ema(data, long_period) # 26일
    macd = short_ema - long_ema # 12일 지수 이동평균 - 26일 지수 이동평균
    signal = ema(macd, signal_period) # MACD의 9일 지수 이동 평균
    # macd_histogram = macd - signal
    return signal


def bollinger_bands(data, period=20, std=2):
    sma_20 = sma(data, period) # 단순 이동 평균선
    std_dev = data.rolling(window=period).std()    
    upper_band = sma_20 + (std_dev * std)
    lower_band = sma_20 - (std_dev * std)
    return pd.DataFrame({'boll_ub': upper_band, 'boll_lb': lower_band})

def rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean() # 양수 값만 남기고 나머지는 0으로 대체
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean() # 음수만 남기고 나머지는 0으로 대체 음수를 양수로 변환
    epsilon = 1e-10 
    rs = gain / (loss + epsilon)

    rsi_li = 100 - (100 / (1 + rs))

    return rsi_li


def cci(high, low, close, period=14):
    tp = (high + low + close) / 3
    sma_tp = sma(tp, period) # tp의 14일 이동평균
    mad = abs(tp - sma_tp).rolling(window=period).mean()
    epsilon = 1e-10 
    return (tp - sma_tp) / (0.015 * mad + epsilon)


def sma_rolling(data):
    data_imp = data.copy()
    for period in [30, 60, 220, 264]:  # 30 days, 60 days, 10 months, 12 months
        data_imp[f'SMA_{period}'] = sma(data_imp['adjust_close'], period)
    return data_imp[['SMA_30', 'SMA_60', 'SMA_220', 'SMA_264']]


def close_and_return_momentum(data):
    data_imp = data.copy()
    # 한달 21일로 계산
    # periods = [21, 42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252]
    periods = [22, 44, 66, 88, 110, 132, 154, 176, 198, 220, 242, 264]
    # periods = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]

    columns_list = ["return_1m", "return_3m", "return_6m", "return_12m", "return_avg", "mom_12m", "mom_score"]
    
    for idx, period in enumerate(periods):
        data_imp[f'close_{idx+1}_month'] = data_imp['adjust_close'].shift(period)
        columns_list.append(f'close_{idx+1}_month')
        
    data_imp['return_1m'] = (data_imp['adjust_close'] - data_imp['close_1_month']) / data_imp['close_1_month']
    data_imp['return_3m'] = (data_imp['adjust_close'] - data_imp['close_3_month']) / data_imp['close_3_month']
    data_imp['return_6m'] = (data_imp['adjust_close'] - data_imp['close_6_month']) / data_imp['close_6_month']
    data_imp['return_12m'] = (data_imp['adjust_close'] - data_imp['close_12_month']) / data_imp['close_12_month']
    data_imp['return_avg'] = data_imp[['return_1m', 'return_3m', 'return_6m', 'return_12m']].mean(axis=1)
    
    data_imp["mom_12m"] = data_imp["adjust_close"] / data_imp["close_12_month"] - 1
    
    data_imp["mom_score"] = (12 * data_imp["return_1m"] + 4 * data_imp["return_3m"] + 2 * data_imp["return_6m"] + 1 * data_imp["return_12m"]) / 19

    return data_imp[columns_list]


def preprocess_data(data):
    """기술지표를 만드는 함수

    Args:
        data (df): etf data
    output :
        data(df) : 기술지표 추가된 데이터
    """
    concat_data = pd.DataFrame()
    dow_30 = data[data["TICKER"] == "DIA"]["adjust_close"]
    dow_30 = pd.DataFrame({"dow_30": dow_30})
    
    for ticker in data.TICKER.unique():
        imp = data[data['TICKER'] == ticker].copy()
        # macd 만들기
        macd_data = macd(imp['adjust_close'], 12, 26, 9)
        macd_data = pd.DataFrame({'macd': macd_data})
        # print(macd_data.plot( title = ticker))
        # plt.show()
        # bollinger band 만들기
        boll_band = bollinger_bands(imp["adjust_close"], period=20, std=2)
        # bollinger_bands(data["adjust_close"], period=20, std=2).plot(title=ticker)
        # rsi 만들기
        rsi_df = rsi(imp["adjust_close"], period=14)
        rsi_df = pd.DataFrame({'rsi': rsi_df})
        # rsi_df.plot(title=ticker)
        # plt.show()
        # cci 만들기
        cci_df = cci(imp["adjust_high"], imp["adjust_low"], imp["adjust_close"], period=14)
        cci_df = pd.DataFrame({'cci': cci_df})
        
        # 이동평균 만들기
        sma_df = sma_rolling(imp)
        
        # close_and_return
        close_and_return_df = close_and_return_momentum(imp)
        
        # 모멘텀 계산
        # momentums= calculate_momentum(imp)
        tot_imp = pd.concat([imp, macd_data, boll_band, rsi_df, cci_df, sma_df, close_and_return_df, dow_30], axis=1)
        
        concat_data = pd.concat([concat_data, tot_imp], axis=0)
    concat_data.dropna(inplace=True)
    concat_data.sort_index(inplace=True)
        
    return concat_data