import os
import time
import numpy as np
import pandas as pd
from openai import OpenAI

# Set your OpenAI API key and base URL
openai = OpenAI(
    api_key=  "",# Replace with your actual DeepInfra token
    base_url="https://api.deepinfra.com/v1/openai",
)

stream = False  # Set to True if you want to stream the response
model_used = 'sentiment_deepseek'  # Define the model_used variable

def get_sentiment(symbol, *texts):
    texts = [text for text in texts if text != 0]
    num_text = len(texts)
    print(f"Number of texts: {num_text}")
    text_content = " ".join([f"### News to Stock Symbol -- {symbol[idx]}: {text}" for idx, text in enumerate(texts)])

    # conversation = [
    #     {"role": "system",
    #      "content": f"Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. {num_text} summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."},
    #     {"role": "user",
    #      "content": f"News to Stock Symbol -- AAPL: Apple (AAPL) increase 22% ### News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30% ### News to Stock Symbol -- MSFT: Microsoft (MSTF) price has no change"},
    #     {"role": "assistant", "content": "5, 1, 3"},
    #     {"role": "user",
    #      "content": f"News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15 ### News to Stock Symbol -- AAPL: Apple (AAPL) will release VisonPro on Feb 2, 2024"},
    #     {"role": "assistant", "content": "4, 4"},
    #     {"role": "user", "content": text_content},
    # ]
    
    conversation = [
        {"role": "system",
         "content": (
             f"Forget all previous instructions. You are now a financial expert giving investment advice. I'll give you a news summary, {num_text} summarized news will be passed in each time. and you need to answer whether this news is GOOD NEWS or BAD NEWS for the listed company.  Please choose only one option from GOOD NEWS, BAD NEWS, NOT SURE, and do not provide any additional responses. Then convert GOOD NEWS to 2, NOT SURE to 1, and BAD NEWS to 0. Only respond with numbers separated by commas (e.g., '2,1,0')."
         )},
        {"role": "user",
         "content": "News to Stock Symbol -- AAPL: Apple announced record-breaking iPhone sales ### News to Stock Symbol -- AAPL: Apple is being sued for patent infringement ### News to Stock Symbol -- AAPL: Apple will attend CES 2025 but no product is confirmed"},
        {"role": "assistant", "content": "2,0,1"},
        {"role": "user", "content": text_content},
    ]


    try:
        chat_completion = openai.chat.completions.create(
    #        model='deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
            model='deepseek-ai/DeepSeek-V3',
            messages=conversation,
            temperature=0,
            max_tokens=100,
            stream=stream,
        )

        if stream:
            content = ""
            for event in chat_completion:
                if event.choices[0].finish_reason:
                    print(event.choices[0].finish_reason,
                          event.usage['prompt_tokens'],
                          event.usage['completion_tokens'])
                else:
                    content += event.choices[0].delta.content
            print(content)
        else:
            content = chat_completion.choices[0].message.content
            print(content)
            print(chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens)

    except Exception as e:
        print(f"Error: {e}")
        return [np.nan] * num_text

    sentiments = []
    for sentiment in content.split(',')[:-1]:
        try:
            sentiment_value = int(sentiment.strip())
        except ValueError:
            print("content error")
            print("sentiment was: " + str(sentiment.strip()))
            sentiment_value = np.nan
        sentiments.append(sentiment_value)
    return sentiments

def process_csv(input_csv_path, output_csv_path, batch_size=5, chunk_size=1000):
    start_time = time.time()

    # Check if the output file exists and load the last processed row
    if os.path.exists(output_csv_path):
        output_df = pd.read_csv(output_csv_path, 
        on_bad_lines='warn',
        engine='python'
)
        last_processed_row = len(output_df)
    else:
        last_processed_row = 0

    # Read the CSV file in chunks
    chunks = pd.read_csv(input_csv_path, encoding="utf-8", chunksize=chunk_size,
    on_bad_lines='warn', 
    engine='python'     # Print a warning for each skipped line
    )

    for chunk_number, chunk in enumerate(chunks):
        # Skip already processed chunks
        if chunk_number * chunk_size < last_processed_row:
            continue

        chunk.columns = chunk.columns.str.capitalize()
        if model_used not in chunk.columns:
            chunk[model_used] = np.nan

        for i in range(0, len(chunk), batch_size):
            batch = chunk.iloc[i:i + batch_size]
            texts = batch['Lsa_summary'].tolist()
            symbol = batch['Stock_symbol'].tolist()  # Extract the stock symbol for the current batch
            sentiments = get_sentiment(symbol, *texts)

            for j, sentiment in enumerate(sentiments):
                if i + j < len(chunk):
                    chunk.loc[chunk.index[i + j], model_used] = sentiment

        # Append the processed chunk to the output file
        chunk.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

    print(f"Process completed in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    import csv
    csv.field_size_limit(10**7)  # <-- 이 줄 추가!

    model_path = "base_line_model/Task_1_FinRL_DeepSeek_Stock/data/"
    input_file_name='nasdaq_exteral_data.csv'
    input_file = model_path + input_file_name
    output_file= model_used + '_' + input_file_name
    process_csv(input_file, output_file, batch_size=100, chunk_size=10000)
