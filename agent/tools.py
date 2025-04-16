import pandas as pd
import numpy as np
from langchain.tools import tool

# Simulate a historical MMM dataset
np.random.seed(42)
channels = ["Paid Search", "Paid Social", "TV", "Display", "YouTube"]
years = ["2023", "2024"]

data = []
for year in years:
    for channel in channels:
        spend = np.random.uniform(1_000_000, 5_000_000)
        roi = np.random.uniform(1.2, 2.0)
        contribution = spend * roi
        data.append({
            "year": year,
            "channel": channel,
            "spend": round(spend, 2),
            "roi": round(roi, 2),
            "contribution": round(contribution, 2)
        })

df_mmm = pd.DataFrame(data)

@tool
def get_historical_contribution(channel: str, year: str) -> str:
    row = df_mmm[(df_mmm["channel"] == channel) & (df_mmm["year"] == year)]
    if row.empty:
        return f"No data found for {channel} in {year}."
    contribution = row.iloc[0]["contribution"]
    return f"In {year}, {channel} contributed ${contribution:,.2f} in revenue."

@tool
def simulate_budget_scenario(channel: str, extra_budget: float, time_period: str) -> str:
    row = df_mmm[(df_mmm["channel"] == channel) & (df_mmm["year"] == time_period)]
    if row.empty:
        return f"No model data available for {channel} in {time_period}."
    roi = row.iloc[0]["roi"]
    projected_return = extra_budget * roi
    return f"If you invest an extra ${extra_budget:,.0f} in {channel} during {time_period}, you can expect approximately ${projected_return:,.2f} in additional revenue."
