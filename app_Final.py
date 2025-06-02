import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import burr12
from scipy.optimize import minimize
import os

# === App Header ===
st.set_page_config(page_title="Surplus Optimization", layout="wide")
st.title("Surplus Optimization under Demand Uncertainty")

# === Sidebar Inputs ===
st.sidebar.header("Input Parameters")
macro_cap = st.sidebar.slider("Macro Surplus Cap (as % of total demand)", 0.1, 0.5, 0.4, step=0.05)
n_samples = st.sidebar.number_input("Number of Demand Samples", min_value=50, max_value=1000, value=100)
ss = st.sidebar.slider("Substitutability Group Overflow Factor (ss)", 0.01, 0.5, 0.1, step=0.01)
uploaded_file = st.sidebar.file_uploader("Upload Input Excel File", type=["xlsx"])

# === Functions ===
def load_and_clean_data(xls):
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
    product_df['Capacity'] = product_df['Capacity'].apply(lambda x: x if pd.notna(x) else 1.0)

    dist_df.columns = ['DemandVarGroup', 'distribution', 'c', 'd', 'loc', 'scale']
    product_df = product_df.rename(columns={'Variance group': 'DemandVarGroup'})
    product_df = product_df.merge(dist_df.drop(columns='distribution'), on='DemandVarGroup', how='left')

    return product_df

def simulate_demand(product_df, n_samples):
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
        return -np.mean(total_profit)
    return objective

def optimize_surplus(product_df, simulated_demand, macro_cap, ss):
    demand = product_df['Demand'].values
    margin = product_df['Margin'].values
    cogs = product_df['COGS'].values
    capacity = product_df['Capacity'].values
    max_surplus = capacity * demand
    total_demand = demand.sum()

    objective_fn = define_objective(demand, margin, cogs, simulated_demand, max_surplus)
    expected_actual = simulated_demand.mean(axis=1)
    overflow = np.maximum(expected_actual - demand, 0)

    sub_groups = product_df.groupby('Substitutability group').groups
    constraints = [{'type': 'ineq', 'fun': lambda x: macro_cap * total_demand - np.sum(x)}]

    for group_id, indices in sub_groups.items():
        valid_idx = [i for i in indices if max_surplus[i] > 0]
        if len(valid_idx) > 1:
            required = overflow[valid_idx].sum() * ss
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=valid_idx, req=required: np.sum(x[idx]) - req})

    bounds = [(0, ms if np.isfinite(ms) else None) for ms in max_surplus]
    x0 = 0.2 * max_surplus
    x0[max_surplus == 0] = 0

    result = minimize(objective_fn, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'disp': False})
    return result, max_surplus, overflow, sub_groups

# === Run App ===
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    product_df = load_and_clean_data(xls)
    product_df = product_df.iloc[:100]
    simulated_demand = simulate_demand(product_df, n_samples)
    result, max_surplus, overflow, sub_groups = optimize_surplus(product_df, simulated_demand, macro_cap, ss)

    product_df['OptimalSurplus'] = result.x
    product_df['MaxSurplus'] = max_surplus

    st.success("Optimization Completed")
    st.write("### Optimal Surplus per Product")
    st.dataframe(product_df[['ProductID', 'Demand', 'Margin', 'COGS', 'Capacity', 'OptimalSurplus']])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(product_df['ProductID'], result.x, color='royalblue')
    ax.set_ylabel('Surplus Units')
    ax.set_title('Optimal Surplus Allocation')
    ax.set_xticks(np.arange(0, len(product_df), 10))
    ax.set_xticklabels(product_df['ProductID'][::10], rotation=45, ha='right')
    st.pyplot(fig)


    profit = ((np.minimum(product_df['Demand'].values[:, None] + result.x[:, None], simulated_demand) * product_df['Margin'].values[:, None]) - (result.x[:, None] * product_df['COGS'].values[:, None])).sum(axis=0)

    st.write("### Profit Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(profit, bins=30, color='green', edgecolor='black')
    ax2.set_title('Histogram of Simulated Total Profits')
    ax2.set_xlabel('Total Profit')
    ax2.set_ylabel('Frequency')
    st.pyplot(fig2)

    st.write("Mean Profit:", np.mean(profit))
    st.write("Std Dev Profit:", np.std(profit))
