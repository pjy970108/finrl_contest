from polygon import RESTClient
from polygon.rest import models
import csv
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# Access the API key
api_key = os.getenv("POLYGON_API_KEY")

# Initialize Polygon API Client
client = RESTClient(api_key)

# Fetch aggregated bar data
aggs = client.get_aggs(
    "MMM",
    1,
    "minute",
    "2023-01-01",
    "2023-08-01",
    limit = 50000
)

# Convert Agg objects to dictionaries & format timestamp
data_dicts = [
    {
        **vars(agg),  
        "timestamp": datetime.utcfromtimestamp(agg.timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
    }
    for agg in aggs
]

# Define the CSV file name
csv_file = "aggregated_aggs.csv"

# Writing to CSV
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=data_dicts[0].keys())  # Use converted dict keys
    writer.writeheader()
    writer.writerows(data_dicts)  # Write list of dictionaries

print(f"CSV file '{csv_file}' has been created successfully!")
