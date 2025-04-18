import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import pickle

# Example input data (media spend + observed revenue)
# !pip install "pandas<2.2.0"
import pandas as pd
import numpy as np

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
