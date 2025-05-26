import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import burr12
import pyswarms as ps
import os

np.random.seed(42)

# Data Loading
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
    product_df['Capacity'] = product_df['Capacity'].apply(lambda x: x if pd.notna(x) else 100.0)

    dist_df.columns = ['DemandVarGroup', 'distribution', 'c', 'd', 'loc', 'scale']
    product_df = product_df.rename(columns={'Variance group': 'DemandVarGroup'})
    product_df = product_df.merge(dist_df.drop(columns='distribution'), on='DemandVarGroup', how='left')

    return product_df

# Simulate Demand
def simulate_demand(product_df: pd.DataFrame, n_samples: int = 500) -> np.ndarray:
    simulated_demand = []
    for _, row in product_df.iterrows():
        dist = burr12(row['c'], row['d'], loc=row['loc'], scale=row['scale'])
        simulated = row['Demand'] * dist.rvs(size=n_samples)
        simulated_demand.append(simulated)
    return np.array(simulated_demand)

# Objective with Penalty
def make_fitness_function(product_df, simulated_demand, macro_cap):
    demand = product_df['Demand'].values
    margin = product_df['Margin'].values
    cogs = product_df['COGS'].values
    capacity = product_df['Capacity'].values
    max_surplus = demand * capacity
    total_demand = np.sum(demand)

    expected_actual = simulated_demand.mean(axis=1)
    overflow = np.maximum(expected_actual - demand, 0)
    group_map = product_df.groupby('Substitutability group').groups

    def fitness(surplus_matrix):
        n_particles = surplus_matrix.shape[0]
        fitness_vals = []

        for i in range(n_particles):
            s = surplus_matrix[i]
            s = np.clip(s, 0, max_surplus)

            sales = np.minimum(demand[:, None] + s[:, None], simulated_demand)
            revenue = sales * margin[:, None]
            cost = s[:, None] * cogs[:, None]
            profit = revenue.sum(axis=0) - cost.sum(axis=0)
            loss = -np.mean(profit)

            # Macro constraint
            penalty = 0
            if np.sum(s) > macro_cap * total_demand:
                penalty += 1e6 * (np.sum(s) - macro_cap * total_demand)

            # Group constraints
            for group_id, indices in group_map.items():
                idx = list(indices)
                group_surplus = np.sum(s[idx])
                group_need = overflow[idx].sum()
                if group_surplus < group_need:
                    penalty += 1e6 * (group_need - group_surplus)

            fitness_vals.append(loss + penalty)
        return np.array(fitness_vals)


    return fitness, max_surplus

# Optimization
def optimize_with_pso(product_df, simulated_demand, macro_cap):
    fitness_function, max_surplus = make_fitness_function(product_df, simulated_demand, macro_cap)

    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    bounds = (np.zeros_like(max_surplus), max_surplus)

    optimizer = ps.single.GlobalBestPSO(n_particles=300, dimensions=len(max_surplus),
                                        options=options, bounds=bounds)

    cost, pos = optimizer.optimize(fitness_function, iters=150)

    return cost, pos, optimizer.cost_history

# Save and Visualize
def save_results(product_df, pos, cost_history, simulated_demand, output_dir="pso_results"):
    os.makedirs(output_dir, exist_ok=True)

    product_df['OptimalSurplus'] = pos
    product_df.to_csv(os.path.join(output_dir, "optimal_surplus_output.csv"), index=False)

    # Plot cost history
    plt.figure(figsize=(10, 5))
    plt.plot(cost_history, label='Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('PSO Optimization Cost Over Iterations')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_history.png"))
    plt.close()

    # Profit histogram
    demand = product_df['Demand'].values
    margin = product_df['Margin'].values
    cogs = product_df['COGS'].values
    sales = np.minimum(demand[:, None] + pos[:, None], simulated_demand)
    revenue = sales * margin[:, None]
    cost = pos[:, None] * cogs[:, None]
    profit = revenue.sum(axis=0) - cost.sum(axis=0)

    plt.hist(profit, bins=30, color='green', edgecolor='black')
    plt.title('Histogram of Simulated Profits')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "profit_distribution.png"))
    plt.close()

    print("Saved to:", output_dir)
    print("Mean profit:", np.mean(profit))
    print("Std profit:", np.std(profit))

def print_group_surplus_vs_need(product_df, simulated_demand):
    demand = product_df['Demand'].values
    expected_actual = simulated_demand.mean(axis=1)
    print("expected_actual", expected_actual)
    print(len(expected_actual))
    overflow = np.maximum(expected_actual - demand, 0)

    group_map = product_df.groupby('Substitutability group').groups

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
    simulated_demand = simulate_demand(product_df, n_samples=1000)
    cost, pos, history = optimize_with_pso(product_df, simulated_demand, macro_cap=0.4)
    save_results(product_df, pos, history, simulated_demand)
    print_group_surplus_vs_need(product_df, simulated_demand)

    
if __name__ == "__main__":
    main()
