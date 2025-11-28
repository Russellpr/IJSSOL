# Model 01: Production Rate is Known
# Model 02: Production Rate is Unknown
# Model 03: Carbon Tax is added
# Model 04: Lean Inventory is added (Sum(Qv)=Sum(D))
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
m = pyo.ConcreteModel()
#Set
m.I = ['P1', 'P2']  # list of products
m.J = ['B1','B2']
m.T = pyo.RangeSet(3)

#Parameters
m.H =5
m.p = {(i, t): 30 + 5*t for i in m.I for t in m.T}  # dummy production values
np.random.seed(0)  # for reproducibility
demand_samples = np.random.normal(loc=50, scale=10, size=(len(m.I), len(list(m.T)))).astype(int)
m.D = {(i, t): demand_samples[idx_i][t-1] for idx_i, i in enumerate(m.I) for t in m.T}
m.o = {(i, t): 20 for i in m.I for t in m.T}
m.h = {(i, t): 10 for i in m.I for t in m.T}
m.u = {1: 0.08, 2: 0.08, 3: 0.08, 4: 0.08, 5: 0.08, 6: 0.08, 7: 0.08, 8: 0.08}
m.O = {(j, t): 15 for j in m.J for t in m.T}
m.C = {
    ('B1', 1): 400, ('B1', 2): 420, ('B1', 3): 440,
    ('B2', 1): 430, ('B2', 2): 450, ('B2', 3): 470
}

m.U = {
    ('B1', 1): 40, ('B1', 2): 38, ('B1', 3): 36,
    ('B2', 1): 35, ('B2', 2): 33, ('B2', 3): 31
}

#Variables
m.n = pyo.Var(m.I, m.T, domain=pyo.NonNegativeIntegers)
m.q = pyo.Var(m.I, m.T, domain=pyo.NonNegativeIntegers)
m.Q = pyo.Var(m.I, m.T, domain=pyo.NonNegativeIntegers)
m.P = pyo.Var(m.I, m.J, m.T, domain=pyo.NonNegativeIntegers)

#Simpilify
h = m.h
o = m.o
n = m.n
H = m.H
q = m.q
P = m.P
C = m.C
O = m.O
D = m.D
p = m.p
Q = m.Q
u = m.u
U = m.U

#Objective Function
def ObjRule(m):
    return sum(m.o[i, t]*m.n[i, t] + (m.h[i, t]*m.q[i, t] / 2) for i in m.I for t in m.T) + \
           sum(m.P[i, j, t] * m.C[j, t] + m.H * m.Q[i, t] for i in m.I for j in m.J for t in m.T)
m.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)
#Constraints
def Constraint1(m, t):
    return sum(m.P[i, j, t] for i in m.I for j in m.J) == sum(m.p[i, t] for i in m.I)
m.Con1 = pyo.Constraint(m.T, rule=Constraint1)

def Constraint2(m, i):
    return sum(m.n[i, t]*m.q[i, t] for t in m.T) == sum(m.D[i, t] for t in m.T)
m.Con2 = pyo.Constraint(m.I, rule=Constraint2)

def Constraint3(m, i, t):
    return m.n[i, t] <= 10
m.Con3 = pyo.Constraint(m.I, m.T, rule=Constraint3)

def Constraint45(m, i, t):
    if t == m.T.first():
        return m.Q[i, t] == m.p[i, t] - (m.n[i, t] * m.q[i, t]) / 2
    return m.Q[i, t] == m.Q[i, t - 1] + m.p[i, t] - (m.n[i, t] * m.q[i, t]) / 2
m.Con45 = pyo.Constraint(m.I, m.T, rule=Constraint45)

def Constraint6(m, i, t):
    total_demand = sum(m.D[i, tp] for tp in m.T if tp <= t)
    total_supply = sum(m.n[i, tp] * m.q[i, tp] for tp in m.T if tp <= t)
    return total_supply >= total_demand
m.Con6 = pyo.Constraint(m.I, m.T, rule=Constraint6)

opt = SolverFactory('ipopt', executable='C:\\Ipopt\\bin\\ipopt.exe')
opt.solve(m)

# Print the results
m.pprint()
#print values
print('-------------------------Optimal----------------------------------')
Optimal = pyo.value(m.obj)
print("Optimal Answer is:", Optimal)

print('---------------------------Q[i, t]-------------------------------')
for i in m.I:
    for t in m.T:
        print(f"Product: {i}, Period: {t}, Inventory Q: {pyo.value(m.Q[i, t])}")
print('--------------------------q[i, t]-------------------------------')
for i in m.I:
    for t in m.T:
        print(f"Product: {i}, Period: {t}, Shipment q: {pyo.value(m.q[i, t])}")
print('------------------------n[i, t]--------------------------------')
for i in m.I:
    for t in m.T:
        print(f"Product: {i}, Period: {t}, Replenishment n: {pyo.value(m.n[i, t])}")
print('--------------------------P[j, t]-------------------------------')
for i in m.I:
    for j in m.J:
        for t in m.T:
            print(f"Product: {i}, Supplier: {j}, Period: {t}, Procurement P: {pyo.value(m.P[i, j, t])}")
print('----------------------n[i, t] * q[i, t]--------------------------')
total_sum = 0
for i in m.I:
    for t in m.T:
        nq = pyo.value(m.n[i, t]) * pyo.value(m.q[i, t])
        total_sum += nq
        print(f"Product: {i}, Period: {t}, n*q: {nq}")
print("Total sum of n[i,t] * q[i,t] across all periods:", total_sum)
print('---------------------Carbon Emission------------------------------')
for i in m.I:
    for j in m.J:
        for t in m.T:
            emission = pyo.value(m.P[i, j, t] * m.U[j, t])
            print(f"Product: {i}, Supplier: {j}, Period: {t}, Emission: {emission}")
total_emission = sum(pyo.value(m.P[i, j, t] * m.U[j, t]) for i in m.I for j in m.J for t in m.T)
print("Total sum of Carbon Emission:", total_emission)
print('---------------------Carbon Emission pr---------------------------')
total_sum2 = 0
for i in m.I:
    for t in m.T:
        emission = pyo.value(m.p[i, t]) * m.u[t]
        total_sum2 += emission
        print(f"Product: {i}, Period: {t}, Emission: {emission}")
print("Total sum of Carbon Emission (production):", total_sum2)
print('-----------------------------------------------------------------')
print('-------------------------Optimal----------------------------------')
print("Optimal Answer is :",Optimal)
print("Total sum of Carbon Emission for production:", total_sum2)
print("Total sum of Carbon Emission:", total_emission)
