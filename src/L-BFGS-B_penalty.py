import pandas as pd
import numpy as np
from scipy.stats import burr12
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# Load and Clean Data
def load_and_clean_data(filepath: str):
    xls = pd.ExcelFile(filepath)
    product_df = xls.parse("Demand")
    dist_df = xls.parse("Demand_Variance")

    product_df.index = product_df.iloc[:, 0]
    product_df = product_df.drop(columns=product_df.columns[0])
    product_df = product_df.T.reset_index(drop=True)
    product_df['ProductID'] = [f'P{i}' for i in range(len(product_df))]

    product_df['Demand'] = pd.to_numeric(product_df['Demand'], errors='coerce')
    product_df['Variance group'] = pd.to_numeric(product_df['Variance group'], downcast='integer', errors='coerce')
    product_df['Margin'] = pd.to_numeric(product_df['Margin'], errors='coerce')
    product_df['COGS'] = pd.to_numeric(product_df['COGS'], errors='coerce')
    product_df['Substitutability group'] = pd.to_numeric(product_df['Substitutability group'], downcast='integer', errors='coerce')

    product_df['Capacity'] = product_df['Capacity'].astype(str).str.rstrip('%')
    product_df['Capacity'] = pd.to_numeric(product_df['Capacity'], errors='coerce')
    product_df['Capacity'] = product_df['Capacity'].apply(lambda x: x if pd.notna(x) else 10.0) # 1000%

    dist_df.columns = ['DemandVarGroup', 'distribution', 'c', 'd', 'loc', 'scale']
    product_df = product_df.rename(columns={'Variance group': 'DemandVarGroup'})
    product_df = product_df.merge(dist_df.drop(columns='distribution'), on='DemandVarGroup', how='left')

    return product_df

# Simulate Demand
def simulate_demand(product_df: pd.DataFrame, n_samples: int = 500):
    simulated_demand = []
    for _, row in product_df.iterrows():
        dist = burr12(row['c'], row['d'], loc=row['loc'], scale=row['scale'])
        simulated = row['Demand'] * dist.rvs(size=n_samples)
        simulated_demand.append(simulated)
    return np.array(simulated_demand)

# Objective Function with Penalty
def define_objective(product_df, simulated_demand, macro_cap):
    demand = product_df['Demand'].values
    margin = product_df['Margin'].values
    cogs = product_df['COGS'].values
    capacity = product_df['Capacity'].values
    max_surplus = demand * capacity
    total_demand = np.sum(demand)

    expected_actual = simulated_demand.mean(axis=1)
    overflow = np.maximum(expected_actual - demand, 0)
    group_map = product_df.groupby('Substitutability group').groups

    def objective(surplus):
        surplus = np.clip(surplus, 0, max_surplus)
        sales = np.minimum(demand[:, None] + surplus[:, None], simulated_demand)
        revenue = sales * margin[:, None]
        cost = surplus[:, None] * cogs[:, None]
        profit = revenue.sum(axis=0) - cost.sum(axis=0)
        loss = -np.mean(profit)

        penalty = 0.0
        if np.sum(surplus) > macro_cap * total_demand:
            penalty += 1e8 * (np.sum(surplus) - macro_cap * total_demand)

        for group_id, indices in group_map.items():
            idx = list(indices)
            group_surplus = np.sum(surplus[idx])
            group_need = overflow[idx].sum()
            if group_surplus < group_need:
                penalty += 1e8 * (group_need - group_surplus)

        return loss + penalty

    return objective, max_surplus, group_map, overflow

# Save and Visualize
def save_and_plot(product_df, surplus, simulated_demand, macro_cap, overflow, group_map, output_dir="lbfgsb_results"):
    os.makedirs(output_dir, exist_ok=True)

    product_df['OptimalSurplus'] = surplus
    product_df.to_csv(os.path.join(output_dir, "optimal_surplus_output.csv"), index=False)

    demand = product_df['Demand'].values
    margin = product_df['Margin'].values
    cogs = product_df['COGS'].values

    sales = np.minimum(demand[:, None] + surplus[:, None], simulated_demand)
    revenue = sales * margin[:, None]
    cost = surplus[:, None] * cogs[:, None]
    profit = revenue.sum(axis=0) - cost.sum(axis=0)

    plt.figure(figsize=(10, 5))
    plt.hist(profit, bins=30, color='green', edgecolor='black')
    plt.title('Profit Distribution')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profit_histogram.png"))
    plt.close()

    print("Saved to:", output_dir)
    print("Mean profit:", np.mean(profit))
    print("Std profit:", np.std(profit))

    print("\n=== Group Surplus vs. Group Need ===")
    for group_id, indices in group_map.items():
        idx = list(indices)
        group_surplus = product_df.loc[idx, 'OptimalSurplus'].sum()
        group_need = overflow[idx].sum()
        print(f"Group {int(group_id)}: Surplus = {group_surplus:.2f}, Needed(overflow) = {group_need:.2f}, Gap = {group_need - group_surplus:.2f}")

# Main
def main():
    filepath = "Input_SUA.xlsx"
    product_df = load_and_clean_data(filepath)
    simulated_demand = simulate_demand(product_df, n_samples=500)
    macro_cap = 0.4
    obj_fn, max_surplus, group_map, overflow = define_objective(product_df, simulated_demand, macro_cap)

    bounds = [(0, ms) for ms in max_surplus]
    x0 = np.zeros_like(max_surplus)

    result = minimize(
        obj_fn,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'disp': True, 'maxiter': 1000}
    )

    save_and_plot(product_df, result.x, simulated_demand, macro_cap, overflow, group_map)

main()
