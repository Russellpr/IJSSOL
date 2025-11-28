import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# Create model
m = pyo.ConcreteModel()

# Sets
m.I = ['P1', 'P2']  # Products
m.J = ['B1','B2']
m.T = pyo.RangeSet(3)  # Time periods

# Parameters
m.H = 5
np.random.seed(0)
demand_samples = np.random.normal(loc=50, scale=10, size=(len(m.I), len(list(m.T)))).astype(int)
m.D = {(i, t): demand_samples[idx][t - 1] for idx, i in enumerate(m.I) for t in m.T}
m.o = {(i, t): 20 for i in m.I for t in m.T}
m.h = {(i, t): 10 for i in m.I for t in m.T}
m.u = {t: 0.08 for t in m.T}
m.O = {(j, t): 15 for j in m.J for t in m.T}

# Cost and Emission Factors
m.C = {
    ('B1', 1): 400, ('B1', 2): 420, ('B1', 3): 440,
    ('B2', 1): 430, ('B2', 2): 450, ('B2', 3): 470
}
m.U = {
    ('B1', 1): 40, ('B1', 2): 38, ('B1', 3): 36,
    ('B2', 1): 35, ('B2', 2): 33, ('B2', 3): 31
}
# Variables
m.p = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
m.n = pyo.Var(m.I, m.T, domain=pyo.NonNegativeIntegers)
m.q = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
m.Q = pyo.Var(m.I, m.T, domain=pyo.NonNegativeReals)
m.P = pyo.Var(m.I, m.J, m.T, domain=pyo.NonNegativeReals)

# Objective
def ObjRule(m):
    return sum(m.o[i, t] * m.n[i, t] + m.h[i, t] * m.q[i, t] / 2 for i in m.I for t in m.T) + \
           sum(m.P[i, j, t] * m.C[j, t] + m.H * m.Q[i, t] for i in m.I for j in m.J for t in m.T)
m.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)

# Constraints
def TotalSupplyMatchesProduction(m, t):
    return sum(m.P[i, j, t] for i in m.I for j in m.J) == sum(m.p[i, t] for i in m.I)
m.Con1 = pyo.Constraint(m.T, rule=TotalSupplyMatchesProduction)

def DemandSatisfaction(m, i):
    return sum(m.n[i, t] * m.q[i, t] for t in m.T) == sum(m.D[i, t] for t in m.T)
m.Con2 = pyo.Constraint(m.I, rule=DemandSatisfaction)

def MaxWorkers(m, i, t):
    return m.n[i, t] <= 10
m.Con3 = pyo.Constraint(m.I, m.T, rule=MaxWorkers)

def InventoryBalance(m, i, t):
    if t == m.T.first():
        return m.Q[i, t] == m.p[i, t] - m.n[i, t] * m.q[i, t] / 2
    return m.Q[i, t] == m.Q[i, t - 1] + m.p[i, t] - m.n[i, t] * m.q[i, t] / 2
m.Con45 = pyo.Constraint(m.I, m.T, rule=InventoryBalance)

def SupplyGreaterThanDemand(m, i, t):
    return sum(m.n[i, tp] * m.q[i, tp] for tp in m.T if tp <= t) >= sum(m.D[i, tp] for tp in m.T if tp <= t)
m.Con6 = pyo.Constraint(m.I, m.T, rule=SupplyGreaterThanDemand)

# Solve
opt = SolverFactory('ipopt', executable='C:\\Ipopt\\bin\\ipopt.exe')
opt.solve(m)

# Output
print('-------------------------Optimal----------------------------------')
print(f"Optimal Objective Value: {pyo.value(m.obj)}")

print('---------------------Production Rate------------------------------')
for i in m.I:
    for t in m.T:
        print(f"Product: {i}, Period: {t}, Production Rate: {pyo.value(m.p[i, t])}")

print('---------------------Inventory Q[i,t]-----------------------------')
for i in m.I:
    for t in m.T:
        print(f"Product: {i}, Period: {t}, Inventory: {pyo.value(m.Q[i, t])}")

print('---------------------n[i,t] and q[i,t]----------------------------')
for i in m.I:
    for t in m.T:
        n_val = pyo.value(m.n[i, t])
        q_val = pyo.value(m.q[i, t])
        print(f"Product: {i}, Period: {t}, n: {n_val}, q: {q_val}, n*q: {n_val*q_val}")

print('----------------------Carbon Emissions----------------------------')
total_emission = 0
for i in m.I:
    for j in m.J:
        for t in m.T:
            e = pyo.value(m.P[i, j, t] * m.U[j, t])
            print(f"Product: {i}, Supplier: {j}, Period: {t}, Emission: {e}")
            total_emission += e
print(f"Total Carbon Emission (transport): {total_emission:.2f}")

prod_emission = sum(pyo.value(m.p[i, t]) * m.u[t] for i in m.I for t in m.T)
print(f"Total Carbon Emission (production): {prod_emission:.2f}")
