import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import burr12
from scipy.optimize import minimize
import os

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
    product_df['Capacity'] = product_df['Capacity'].apply(lambda x: x if pd.notna(x) else np.inf)

    dist_df.columns = ['DemandVarGroup', 'distribution', 'c', 'd', 'loc', 'scale']
    product_df = product_df.rename(columns={'Variance group': 'DemandVarGroup'})
    product_df = product_df.merge(dist_df.drop(columns='distribution'), on='DemandVarGroup', how='left')

    return product_df

def simulate_demand(product_df: pd.DataFrame, n_samples: int = 500) -> np.ndarray:
    simulated_demand = []
    for _, row in product_df.iterrows():
        dist = burr12(row['c'], row['d'], loc=row['loc'], scale=row['scale'])
        simulated = row['Demand'] * dist.rvs(size=n_samples)
        simulated_demand.append(simulated)
    return np.array(simulated_demand)

def define_objective(demand, margin, cogs, simulated_demand):
    def objective(surplus):
        sales = np.minimum(demand[:, None] + surplus[:, None], simulated_demand)
        revenue = sales * margin[:, None]
        cost = surplus[:, None] * cogs[:, None]
        total_profit = revenue.sum(axis=0) - cost.sum(axis=0)
        return -np.mean(total_profit)
    return objective

def run_optimization(product_df: pd.DataFrame, simulated_demand: np.ndarray, macro_cap: float = 0.20):
    demand = product_df['Demand'].values
    margin = product_df['Margin'].values
    cogs = product_df['COGS'].values
    capacity = product_df['Capacity'].values
    max_surplus = capacity * demand
    total_demand = demand.sum()
    n_products = len(demand)

    objective_fn = define_objective(demand, margin, cogs, simulated_demand)

    constraints = [{
        'type': 'ineq',
        'fun': lambda x: macro_cap * total_demand - np.sum(x)
    }]

    bounds = [(0, ms if np.isfinite(ms) else None) for ms in max_surplus]
    x0 = np.zeros(n_products)

    result = minimize(
        objective_fn,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'disp': True}
    )

    return result, max_surplus

def save_results_and_visualize(product_df, result, max_surplus, simulated_demand, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    demand = product_df['Demand'].values
    margin = product_df['Margin'].values
    cogs = product_df['COGS'].values

    product_df['OptimalSurplus'] = result.x
    product_df['MaxSurplus'] = max_surplus

    output_csv = os.path.join(output_dir, "optimal_surplus_output.csv")
    product_df[['ProductID', 'Demand', 'Margin', 'COGS', 'Capacity', 'OptimalSurplus']].to_csv(output_csv, index=False)

    print(f"\nOptimization finished: {'Success' if result.success else 'Not converged'}")
    print("Message:", result.message)
    print("Saved to:", output_csv)

    print("Total surplus used:", result.x.sum())
    print("Macro cap limit:", 0.2 * demand.sum())
    print("Within bounds:", np.all(result.x <= max_surplus))

    # Plot surplus per product
    plt.figure(figsize=(12, 6))
    plt.bar(product_df['ProductID'], result.x, label='Optimal Surplus')
    plt.xticks([], [])
    plt.ylabel('Surplus Units')
    plt.title('Optimal Surplus per Product')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "surplus_per_product.png"))
    plt.close()

    # Plot profit distribution
    sales = np.minimum(demand[:, None] + result.x[:, None], simulated_demand)
    revenue = sales * margin[:, None]
    cost = result.x[:, None] * cogs[:, None]
    profit = revenue.sum(axis=0) - cost.sum(axis=0)

    plt.hist(profit, bins=30, color='green', edgecolor='black')
    plt.title('Histogram of Simulated Total Profits')
    plt.xlabel('Total Profit')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profit_distribution.png"))
    plt.close()

    print("Mean Profit:", np.mean(profit))
    print("Std Dev Profit:", np.std(profit))

def main():
    filepath = "Input_SUA.xlsx"
    product_df = load_and_clean_data(filepath)
    # product_df = product_df.iloc[:100]
    simulated_demand = simulate_demand(product_df, n_samples=20)
    result, max_surplus = run_optimization(product_df, simulated_demand)
    save_results_and_visualize(product_df, result, max_surplus, simulated_demand)


main()
