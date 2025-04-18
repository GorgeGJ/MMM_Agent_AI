import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import pickle

# Example input data (media spend + observed revenue)
# !pip install "pandas<2.2.0"

# Set seed for reproducibility
data = pd.read_csv('data/simulated_mmm_input.csv')

# Normalize data for modeling
X = data[['facebook', 'paid_search', 'youtube']] / 100000
y = data['sales'] / 100000

# Bayesian Linear Regression with PyMC
with pm.Model() as model:
    beta_facebook = pm.Normal('beta_facebook', mu=0.03, sigma=0.01)
    beta_paid_search = pm.Normal('beta_paid_search', mu=0.05, sigma=0.015)
    beta_youtube = pm.Normal('beta_youtube', mu=0.02, sigma=0.005)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = (
        beta_facebook * X['facebook'] +
        beta_paid_search * X['paid_search'] +
        beta_youtube * X['youtube']
    )

    revenue_obs = pm.Normal('revenue_obs', mu=mu, sigma=sigma, observed=y)
    trace = pm.sample(draws=500, chains=4, tune=100, target_accept=0.9, return_inferencedata=True)

# Save trace and input data
data.to_csv("data/input_data.csv", index=False)
with open("data/pymc_trace.pkl", "wb") as f:
    pickle.dump(trace, f)
# --- Streamlit App ---
# mmm_agent_scaffold.py

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
# from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="llama3", temperature=0)

# from langchain.tools.python import PythonREPLTool
from langchain_experimental.tools.python.tool import PythonREPLTool

# Attempt to detect and load associated input data from previous modeling
import os

input_data_path = "data/input_data.csv"

if os.path.exists(input_data_path):
    input_df = pd.read_csv(input_data_path)

    # Aggregate spend per channel (mean or sum)
    spend_map = input_df[["facebook", "paid_search", "youtube"]].mean().to_dict()
    total_spend = sum(spend_map.values())

    # Re-run summary generation using dataset-based spend
    channels = list(spend_map.keys())
    period = "March 2024"

    records = []
    for channel in channels:
        beta_key = f"beta_{channel}"
        if beta_key in trace.posterior:
            betas = trace.posterior[beta_key].values.flatten()
            mean_beta = np.mean(betas)
            ci_lower, ci_upper = np.percentile(betas, [2.5, 97.5])
            spend = spend_map[channel]
            mean_sales = spend * mean_beta
            ci_sales_lower = spend * ci_lower
            ci_sales_upper = spend * ci_upper
            roi = mean_sales / spend
            records.append({
                "channel": channel,
                "period": period,
                "spend": spend,
                "spend_share": spend / total_spend,
                "incremental_sales": mean_sales,
                "incremental_sales_ci_lower": ci_sales_lower,
                "incremental_sales_ci_upper": ci_sales_upper,
                "roi": roi
            })

    mmm_results_df = pd.DataFrame(records)
    mmm_results_df["roi_normalized"] = mmm_results_df["roi"] / mmm_results_df["roi"].max()

    # Save enhanced summary again
    dataset_summary_path = "data/mmm_model_summary.csv"
    mmm_results_df.to_csv(dataset_summary_path, index=False)
    dataset_summary_path
else:
    dataset_summary_path = None

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