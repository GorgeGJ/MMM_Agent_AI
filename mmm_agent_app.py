# mmm_agent_scaffold.py

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3", temperature=0)

# from langchain.tools.python import PythonREPLTool
from langchain_experimental.tools.python.tool import PythonREPLTool

import pandas as pd
import numpy as np
import arviz as az
import pickle

# Load PyMC trace
with open("data/pymc_trace.pkl", "rb") as f:
    trace = pickle.load(f)

# Generate MMM summary CSV from PyMC model results
channels = ["facebook", "paid_search", "youtube"]
period = "March 2024"
spend_map = {"facebook": 100000, "paid_search": 150000, "youtube": 80000}

records = []
for channel in channels:
    beta_key = f"beta_{channel}"
    if beta_key in trace.posterior:
        betas = trace.posterior[beta_key].values.flatten()
        mean_beta = np.mean(betas)
        mean_sales = spend_map[channel] * mean_beta
        roi = mean_sales / spend_map[channel]
        records.append({
            "channel": channel,
            "period": period,
            "spend": spend_map[channel],
            "incremental_sales": mean_sales,
            "roi": roi
        })

mmm_results_df = pd.DataFrame(records)
mmm_results_df.to_csv("data/mmm_model_summary.csv", index=False)

# Sample functions for accessing PyMC-based MMM results
def get_incremental_sales(channel: str, period: str):
    row = mmm_results_df[(mmm_results_df['channel'] == channel) & (mmm_results_df['period'] == period)]
    if not row.empty:
        sales = row.iloc[0]['incremental_sales']
        roi = row.iloc[0]['roi']
        return f"In {period}, {channel} drove an estimated ${sales:.2f} in incremental sales with an ROI of {roi:.2f}."
    return "Sorry, I couldn't find data for that channel and period."

def forecast_sales(channel: str, new_spend: float):
    beta_key = f"beta_{channel.lower()}"
    if beta_key not in trace.posterior:
        return f"No forecast data available for {channel}."

    betas = trace.posterior[beta_key].values.flatten()
    sales_samples = new_spend * betas
    mean_sales = np.mean(sales_samples)
    ci_lower, ci_upper = np.percentile(sales_samples, [2.5, 97.5])

    return (
        f"With a spend of ${new_spend:.2f} on {channel}, the forecasted incremental sales is approximately ${mean_sales:.2f}"
        f" (95% CI: ${ci_lower:.2f} to ${ci_upper:.2f})."
    )

def optimize_budget(goal: str = "maximize_sales"):
    return "To maximize sales, allocate 45% to Paid Search, 35% to Facebook, and 20% to YouTube based on posterior means."

# Define tools for the agent
tools = [
    Tool(
        name="Get Incremental Sales",
        func=lambda x: get_incremental_sales(**eval(x)),
        description="Returns incremental sales and ROI given a channel and period. Input should be a dictionary string with 'channel' and 'period'."
    ),
    Tool(
        name="Forecast Sales",
        func=lambda x: forecast_sales(**eval(x)),
        description="Forecast sales based on a new spend amount. Input should be a dictionary string with 'channel' and 'new_spend'."
    ),
    Tool(
        name="Optimize Budget",
        func=lambda x: optimize_budget(goal=x),
        description="Suggest optimal budget allocation based on goal (e.g., 'maximize_sales')."
    ),
    PythonREPLTool()
]

# Create the agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Sample queries
response1 = agent.invoke("How much incremental sales did Facebook drive in March 2024?")
response2 = agent.invoke("Assume a budget of $200,000 for Paid Search next quarter. What impact would this have on overall marketing performance?")
response3 = agent.invoke("How should I allocate budget to maximize sales?")

print(response1)
print(response2)
print(response3)
