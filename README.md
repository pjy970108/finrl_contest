# FinRL Contest 2025
This repository contains the starter kit and tutorials for the FinRL Contest 2025.

## Outline
  - [Tutorial](#tutorial)
  - [Task 1 FinRL-DeepSeek for Stock Trading](#task-1-finrl-deepseek-for-stock-trading)
  - [Task 2 FinRL-AlphaSeek for Crypto Trading](#task-2-finrl-alphaseek-for-crypto-trading)
  - [Task 3 FinLLM Leaderboard - Models with Reinforcement Fine-Tuning (ReFT)](#task-3-finllm-leaderboard---models-with-reinforcement-fine-tuning-reft)
  - [Task 4 FinLLM Leaderboard - Digital Regulatory Reporting (DRR)](#task-4-finllm-leaderboard---digital-regulatory-reporting-drr)
  - [Report Submission Requirement](#report-submission-requirement)
  - [Resources](#resources)

## Tutorial
Please explore 
* [FinRL Contest Documentation](https://finrl-contest.readthedocs.io/en/latest/) for task 1 and 2, 
* [Open FinLLM Leaderboard Documentation](https://finllm-leaderboard.readthedocs.io/en/latest/) for task 3, and 
* [Financial Regulations Documentation](https://financial-regulations.readthedocs.io/en/latest/) for task 4. 

We also welcome questions for these documentations and will update in their FAQs.

Here we also provide some demo for FinRL:
| Task | Model | Environment | Dataset | Link |
| ---- |------ | ----------- | ------- | ---- |
| Stock Trading @ [FinRL Contest 2023](https://open-finance-lab.github.io/finrl-contest.github.io/)| PPO | Stock Trading Environment | OHLCV | [Baseline solution](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/tree/main/Tutorials/FinRL_Contest_2023_Task_1_baseline_solution) |
| Stock Trading | PPO | Stock Trading Environment | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/Tutorials/FinRL_stock_trading_demo.ipynb) |
| Crypto Trading @ [FinRL Contest 2024](https://open-finance-lab.github.io/finrl-contest-2024.github.io/)| Ensemble | Crypto Trading Environment | LOB | [Baseline solution](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_1_starter_kit) |
| Stock Trading | Ensemble | Stock Trading Environment | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/Tutorials/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb) for [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)|
| Sentiment Analysis with RLMF @ [FinRL Contest 2024](https://open-finance-lab.github.io/finrl-contest-2024.github.io/) | / | Stock Sentiment Environment | OHLCV, News | [Starter-Kit](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_2_starter_kit)|
| Sentiment Analysis with Market Feedback | ChatGLM2-6B | -- | Eastmoney News | [Code](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Sentiment_Analysis_v1/FinGPT_v1.0) |
| Stock Price Prediction | Linear Regression | -- | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/Tutorials/Example_Linear_Regression.ipynb) |

## Task 1 FinRL-DeepSeek for Stock Trading
This task is about developing automated stock trading agents trained on stock prices and financial news data, by combining reinforcement learning and large language models (LLMs).

The starter kit for Task 1 is [here](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/tree/main/Task_1_FinRL_DeepSeek_Stock). It contains example code from FinRL-DeepSeek. 


## Task 2 FinRL-AlphaSeek for Crypto Trading
This task aims to develop robust and effective trading agents for cryptocurrencies through factor mining and ensemble learning. 

The starter kit for Task 2 is [here](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/tree/main/Task_2_FinRL_AlphaSeek_Crypto). It contains example code for factor selection and ensemble learning. Participants are strongly encouraged to develop your own factor mining approaches and are welcome to experiment with various ensemble configurations that yield optimal results.


## Task 3 FinLLM Leaderboard - Models with Reinforcement Fine-Tuning (ReFT)
This task encourages participants to submit their models and compete for high rankings in the  Open FinLLM Leaderboard.

The starter kit for Task 3 is [here](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/tree/main/Task_3_FinLLM_Leaderboard_Models_ReFT). We will add evaluation framework introduction soon.


## Task 4 FinLLM Leaderboard - Digital Regulatory Reporting (DRR)
This task aims to challenge the community to explore the strengths and limitations of LLMs in digital regulatory reporting: CDM, MOF, and XBRL.

The starter kit for Task 4 is [here](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/tree/main/Task_4_FinLLM_Leaderboard_DRR). It provides the summary and statistics of question datasets which will be used to evaluate the submitted LLMs. Participants can collect raw data themselves according to data sources or utilize other datasets to train or fine-tune their models.


## Resources
Useful materials and resources for contestants:
* FinRL Contests
  * FinRL Contest 2023: [Contest Website](https://open-finance-lab.github.io/finrl-contest.github.io/); [Github](https://github.com/Open-Finance-Lab/FinRL_Contest_2023)
  * FinRL Contest 2024: [Contest Website](https://open-finance-lab.github.io/finrl-contest-2024.github.io/); [Github](https://github.com/Open-Finance-Lab/FinRL_Contest_2024)
* [FinRL-DeepSeek](https://github.com/benstaf/FinRL_DeepSeek)
* Regulations Challenges at COLING 2025: [Contest Website](https://coling2025regulations.thefin.ai/); [Github](https://github.com/Open-Finance-Lab/Regulations_Challenge_COLING_2025)
* [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
* [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta)
* [FinRL Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials)
