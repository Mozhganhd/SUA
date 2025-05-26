# Surplus Unit Allocation Optimization

## Project Overview
This project addresses a **stochastic surplus allocation optimization problem** provided by Corteva Agriscience. The task is to determine optimal surplus units to add to demand forecasts of agricultural products in order to **maximize total expected profit**, while respecting:

1. Product-level **capacity limits**
2. A global ("macro") cap on total surplus (e.g., 20%-50% of total demand)
3. Group-level **substitutability constraints**, ensuring sufficient surplus coverage within each group

I implemented and compared multiple optimization strategies to explore trade-offs between convergence, interpretability, and constraint satisfaction.

---

## Folder Structure
```
SUA_Optimization/
├── data/
│   └── Input_SUA.xlsx                 # Provided data
├── results/                           # Output folder
├── src/
│   ├── SLSQP_nogroupconstraint.py
│   ├── SLSQP_hardConstraint.py
│   ├── SLSQP_penaltyConstraints.py
│   ├── SLSQP_SoftConstraints.py
│   ├── L-BFGS-B_penalty.py
│   ├── PSO.py
├── README.md
├── requirements.txt
└── streamlit_app.py
```

---

## Optimization Strategies and Evolution

### 1. `SLSQP_nogroupconstraint.py`
- **Approach**: Started with a baseline SLSQP optimization without group-level constraints.
- **Constraints**: Applied only capacity and macro-level surplus limit.
- **Outcome**: The model converged and provided a reference for expected profit under basic feasibility.

### 2. `SLSQP_hardConstraint.py`
- **Approach**: Added strict hard constraints to ensure each substitutability group receives at least as much surplus as its expected overflow.
- **Constraint Logic**:
    - For each group: `sum(surplus in group) >= sum(expected overflow in group)`
- **Outcome**: Optimization often failed to converge, returning `Exit mode 8: Positive directional derivative for linesearch`.
- **Observation**: Likely due to overly strict or infeasible constraints across large product sets. Optimization slowed down significantly.

### 3. `SLSQP_penaltyConstraints.py`
- **Approach**: Replaced hard constraints with penalty terms added to the objective for each group under-supplied.
- **Penalty Term**: `penalty += (group_need - group_surplus)^2` when surplus < overflow.
- **Outcome**: Improved convergence on small-scale tests (first 100 products). Scaling to the full dataset again caused convergence issues.

### 4. `SLSQP_softConstraints.py`
- **Approach**: Applied a relaxed constraint strategy where groups must receive only a fraction (e.g., 30%) of their overflow.
- **Outcome**: A compromise between feasibility and constraint satisfaction. Some constraint violation still occurred, but it converged more reliably than the hard-constraint version.

### 5. `PSO.py` (Particle Swarm Optimization)
- **Approach**: Used `pyswarms` for global optimization with penalty-based constraints.
- **Advantages**: No reliance on gradient information; can escape local optima.
- **Outcome**: Convergence achieved, and profit improved. However, **some groups still showed a surplus < overflow**, resulting in unmet expected demand.

### 6. `L-BFGS-B_penalty.py`
- **Approach**: Used L-BFGS-B optimizer with capacity and macro limits as bounds, and added penalty terms to the objective for group constraint violations.
- **Observation**: Faster than SLSQP in some tests. Yet, as with PSO, some substitutability groups still experienced positive gaps(surplus < overflow).

---

## Important Concepts
- **Overflow**: For each product, overflow = max(mean(simulated demand) - demand, 0). It estimates the extra units potentially needed to satisfy stochastic demand.
- **Group Gap**: For each group: `gap = total group overflow - total group surplus`. Positive values indicate unmet expected demand in the group.

---

## Setup Instructions
```bash
# Step 1: Clone repository and navigate
$ git clone <repo_url>
$ cd SUA_Optimization

# Step 2: Create environment
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 3: Install requirements
$ pip install -r requirements.txt

# Step 4: Run final optimization
$ python src/optimization_slsqp.py
$ python src/SLSQP_nogroupconstraint.py
$ python src/SLSQP_hardConstraint.py
$ python src/SLSQP_penaltyConstraints.py
$ python src/SLSQP_SoftConstraints.py
$ python src/L-BFGS-B_penalty.py
$ python src/PSO.py
# # Optional: Launch Streamlit UI
# $ streamlit run streamlit_app.py
# ```

---

## Requirements
```
pandas
numpy
matplotlib
scipy
pyswarms
openpyxl
<!-- streamlit -->
```

---

## Final Notes
- **Hard vs Soft Constraints**: Hard constraints provide guarantees but may become infeasible under tight limits. Penalty methods offer flexibility but may not fully meet group-level goals.
- **Scalability**: Applying strict group constraints to 500+ products is challenging. Relaxed or penalty-based methods improve scalability.
- **PSO & L-BFGS-B**: Both offer valuable global or smooth convergence but need fine-tuned penalties to respect all constraints.

---

