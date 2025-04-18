import pandas as pd
import numpy as np
import argparse

def simulate_mmm_data(n_samples=10000, channels=None, seed=42):
    if channels is None:
        channels = {
            'facebook': {'mean': 120000, 'std': 20000, 'beta': 0.03},
            'paid_search': {'mean': 140000, 'std': 25000, 'beta': 0.05},
            'youtube': {'mean': 90000, 'std': 15000, 'beta': 0.02}
        }

    np.random.seed(seed)
    data = {}
    revenue = np.zeros(n_samples)

    for name, params in channels.items():
        spend = np.random.normal(params['mean'], params['std'], size=n_samples).clip(min=0)
        data[name] = spend
        revenue += spend * params['beta']

    # Add noise
    revenue += np.random.normal(0, 30000, size=n_samples)
    revenue = revenue.clip(min=0)
    data['revenue'] = revenue

    df = pd.DataFrame(data)
    df.to_csv("simulated_mmm_input.csv", index=False)
    print(f"✅ Saved {n_samples} records to simulated_mmm_input.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000, help="Number of rows to simulate")
    args = parser.parse_args()

    simulate_mmm_data(n_samples=args.samples)
