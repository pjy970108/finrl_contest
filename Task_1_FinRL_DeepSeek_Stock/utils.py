### data loading 및 기타 유용한 함수들을 정의한 파일입니다.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_loader(path):
    ### 
    # Load the data from the path and return the data
    ###
    # return pd.read_csv(path,  dtype={"RET": float, "RETX": float})
    return pd.read_csv(path)


############################Visualization######################################################
def make_plot_for_eda(data, column):
    """
    데이터를 Ticker별로 plot을 그리는 함수입니다.
    """
    for i in data.TICKER.unique():
        print(i)
        temp = data[data["TICKER"] == i]

        # 여러 열을 한 번에 플롯
        plt.figure(figsize=(10, 6))
        plt.plot(temp.date, temp[column], label=column)

        plt.xticks(temp.date[::100], rotation=45)
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.title(f'{column} for Ticker {i}')
        plt.legend()
        plt.show()
        
        
def plot_loss_curve(loss_values):
    # Plot the loss curve
    plt.figure()
    plt.plot(loss_values, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve During Training')
    plt.legend()
    plt.show()
