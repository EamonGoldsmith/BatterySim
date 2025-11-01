
import pybamm
import numpy as np
import pandas as pd

# specify simulation type
model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})

# setup cell parameters
p = pybamm.ParameterValues("Chen2020")

# update parameters to match molicel p42a as used in BATEMO simulation
p.update({
	"Nominal cell capacity [A.h]": 4.2,
	"Open-circuit voltage at 100% SOC [V]": 4.2,
	"Open-circuit voltage at 0% SOC [V]": 2.5,
	"Lower voltage cut-off [V]": 2.1,
	"Cell cooling surface area [m2]": 0.065, # typical 21700 cell
	"Cell volume [m3]": 2.56e-4, # ~25.6 cm^3 for 21700

	"Number of electrodes connected in parallel to make a cell": 1,
	"Number of cells connected in series to make a battery": 28,

	"Total heat transfer coefficient [W.m-2.K-1]": 1e6,
	"Ambient temperature [K]": 298.15,
	"Initial temperature [K]": 298.15,
})

# import drive cycle from file
data_loader = pybamm.DataLoader()
drive_cycle = pd.read_csv("lap_data.csv", comment="#", header=None).to_numpy()

print(drive_cycle)

# create interpolant - must be a function of *dimensional* time
dynamic_current_func = pybamm.Interpolant(drive_cycle[:, 0], drive_cycle[:, 1], pybamm.t)
p["Current function [A]"] = dynamic_current_func

solver = pybamm.CasadiSolver(rtol=1e-9, atol=1e-9)
sim = pybamm.Simulation(model, parameter_values=p, solver=solver)

# run simulation
try:
	sim.solve(t_eval=None)
	output_variables = ["Current [A]", "Voltage [V]"]
	sim.plot(output_variables)

except Exception as e:
	print(f"Simulation failed because of {e}")