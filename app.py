import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Configuration ===
RESULTS_DIR = "SLSQP_SoftConstraints"
CSV_PATH = os.path.join(RESULTS_DIR, "optimal_surplus_output.csv")
SURPLUS_PLOT = os.path.join(RESULTS_DIR, "surplus_per_product.png")
PROFIT_HIST = os.path.join(RESULTS_DIR, "profit_distribution.png")

# === Streamlit App ===
st.set_page_config(page_title="SUA Optimization Dashboard", layout="centered")
st.title("Surplus Unit Allocation Optimization Results")

# Load results
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    st.subheader("📊 Optimal Surplus per Product")
    st.dataframe(df[['ProductID', 'Demand', 'Margin', 'COGS', 'Capacity', 'OptimalSurplus']])

    # Bar Plot
    st.subheader("📈 Surplus Allocation Chart")
    if os.path.exists(SURPLUS_PLOT):
        st.image(SURPLUS_PLOT, caption="Surplus per Product", use_column_width=True)
    else:
        st.warning("Surplus plot not found.")

    # Profit Histogram
    st.subheader("💰 Profit Distribution")
    if os.path.exists(PROFIT_HIST):
        st.image(PROFIT_HIST, caption="Simulated Total Profit Distribution", use_column_width=True)
    else:
        st.warning("Profit histogram not found.")

else:
    st.error("Result CSV file not found. Please run the optimization first.")
