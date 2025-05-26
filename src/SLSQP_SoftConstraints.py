import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import burr12
from scipy.optimize import minimize
import os
np.random.seed(42)


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
    product_df['Capacity'] = product_df['Capacity'].apply(lambda x: x if pd.notna(x) else 1.0)  # instead of np.inf, I used 100%
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

def define_objective(demand, margin, cogs, simulated_demand, max_surplus):
    def objective(surplus):
        surplus = np.clip(surplus, 0, max_surplus)
        sales = np.minimum(demand[:, None] + surplus[:, None], simulated_demand)
        revenue = sales * margin[:, None]
        cost = surplus[:, None] * cogs[:, None]
        total_profit = revenue.sum(axis=0) - cost.sum(axis=0)

        if np.any(np.isnan(total_profit)) or np.any(np.isinf(total_profit)):
            return 1e10
        return -np.mean(total_profit)
    return objective

def run_optimization_with_group_constraints(product_df: pd.DataFrame, simulated_demand: np.ndarray, macro_cap: float):
    demand = product_df['Demand'].values
    margin = product_df['Margin'].values
    cogs = product_df['COGS'].values
    capacity = product_df['Capacity'].values
    max_surplus = capacity * demand
    total_demand = demand.sum()
    n_products = len(demand)

    objective_fn = define_objective(demand, margin, cogs, simulated_demand, max_surplus)

    expected_actual = simulated_demand.mean(axis=1)
    overflow = np.maximum(expected_actual - demand, 0)
    sub_groups = product_df.groupby('Substitutability group').groups

    group_constraints = []
    for group_id, indices in sub_groups.items():
        if len(indices) > 1:
            idx = list(indices)
            required = overflow[idx].sum() * ss
            group_constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=idx, req=required: np.sum(x[idx]) - req
            })

    constraints = [{'type': 'ineq', 'fun': lambda x: macro_cap * total_demand - np.sum(x)}]  + group_constraints
    bounds = [(0, ms if np.isfinite(ms) else None) for ms in max_surplus]
    x0 = 0.2 * max_surplus
    # x0 = np.zeros(n_products)


    result = minimize(
        objective_fn,
        x0,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'disp': True}
    )

    print("\n=== Group Surplus Report ===")
    for group_id, indices in product_df.groupby('Substitutability group').groups.items():
        idx = list(indices)
        group_surplus = result.x[idx].sum()
        group_need = overflow[idx].sum()
        print(f"Group {group_id}: Surplus = {group_surplus:.2f}, Needed = {group_need:.2f}, Gap = {group_need - group_surplus:.2f}")

    return result, max_surplus

def save_results_and_visualize(product_df, result, max_surplus, simulated_demand, output_dir="final_with_constraints"):
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
    print("Macro cap limit:", macro_cap * demand.sum())
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
    product_df = product_df.iloc[:100]
    simulated_demand = simulate_demand(product_df, n_samples)
    result, max_surplus = run_optimization_with_group_constraints(product_df, simulated_demand, macro_cap)
    save_results_and_visualize(product_df, result, max_surplus, simulated_demand)

if __name__ == "__main__":
    method='SLSQP'
    print("method", method)
    macro_cap = 0.4
    print("macro_cap", macro_cap)
    n_samples = 100
    print("n_samples", n_samples)
    ss = 0.3
    print("ss", ss)
    main()
    