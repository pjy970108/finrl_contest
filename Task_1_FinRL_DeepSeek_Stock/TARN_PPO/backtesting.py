import pandas as pd
import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.02, annual_factor=12):
    """
    Sharpe Ratio = (Mean Portfolio Return - Risk-Free Rate) / Standard Deviation of Portfolio Return
    """
    returns = np.array(returns, dtype=np.float64)
    geo_return = np.prod(1 + returns) ** (1 / len(returns)) - 1
    risk_free_rate /= annual_factor
    excess_geometric_return = geo_return - risk_free_rate  # 초과 기하평균 수익률

    std_dev = np.std(returns, ddof=1)  # 표준편차 계산
    return (excess_geometric_return * annual_factor) / (std_dev * np.sqrt(annual_factor)) if std_dev != 0 else 0.0


def calculate_annualized_sortino_ratio(returns, risk_free_rate=0.02, annual_factor=12):
    """
    Annualized Sortino Ratio = (Mean Portfolio Return - Risk-Free Rate) * Annual Factor / Downside Deviation
    """
    returns = np.array(returns, dtype=np.float64)
    geo_return = np.prod(1 + returns) ** (1 / len(returns)) - 1
    risk_free_rate /= annual_factor
    excess_returns = geo_return - risk_free_rate

    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0

    return (excess_returns * annual_factor) / (downside_std * np.sqrt(annual_factor)) if downside_std != 0 else 0.0


def calculate_cagr(returns, annual_factor=12):
    """
    개별 기간별 수익률을 기반으로 연간 수익률 계산
    :param returns: 수익률 리스트 (예: 월별 수익률)
    :param annual_factor: 연간 데이터 수 (월별 수익률이면 12, 주간 수익률이면 52)
    :return: 연간 수익률
    """
    returns = np.array(returns, dtype=np.float64)
    cumulative_return = np.prod(1 + returns) - 1  # 누적 수익률
    years = len(returns) / annual_factor  # 연 단위 기간
    return (1 + cumulative_return) ** (1 / years) - 1


def calculate_max_drawdown(returns):
    """
    Maximum Drawdown (MDD) = max(1 - (Cumulative Value at Time t / Maximum Cumulative Value Up to Time t))
    """
    returns = np.array(returns, dtype=np.float64)
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - running_max) / running_max
    return abs(np.min(drawdowns)) * 100  # 음수값을 절대값으로 변환
    

def calculate_calmar_ratio(returns, annual_factor=12):
    """
    Calmar Ratio = Annualized Return / Maximum Drawdown
    """
    returns = np.array(returns, dtype=np.float64)
    annualized_return = calculate_cagr(returns, annual_factor)
    max_drawdown = calculate_max_drawdown(returns)
    
    return annualized_return / max_drawdown if max_drawdown != 0 else 0.0  # MDD가 0이면 0 반환


def calculate_cumulative_return(returns):
    """
    개별 기간별 수익률을 기반으로 누적 수익률을 계산
    """
    returns = np.array(returns, dtype=np.float64)  # 데이터가 리스트로 주어질 경우 처리
    return np.prod(1 + returns) - 1


def calculate_return(invest_money):
    """
    투자금액을 기반으로 수익률 계산 (NumPy 최적화)
    """
    invest_money = np.array(invest_money, dtype=np.float64)  # 리스트를 NumPy 배열로 변환
    invest_money_shifted = np.roll(invest_money, 1)  # 배열을 한 칸씩 뒤로 이동
    invest_money_shifted[0] = np.nan  # 첫 번째 값은 NaN 처리 (기존 코드에서 0으로 설정)

    returns_ratio = (invest_money / invest_money_shifted) - 1
    returns_ratio[0] = 0  # NaN을 0으로 변경

    return returns_ratio