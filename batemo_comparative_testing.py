
import pybamm
import numpy as np

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
	"Current function [A]": 120 / 3,
})

# specify simulation type
model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
solver = pybamm.CasadiSolver(rtol=1e-9, atol=1e-9)

# Scenario 1
# Simulation to devise cooling energy required to run at standard operating conditions
# (120A from 100% SOC for 20 seconds)

sim = pybamm.Simulation(model, parameter_values=p, solver=solver)
t_eval = np.linspace(0, 20, 500)

# run simulation
try:
	sim.solve(t_eval)

	# to find the cooling energy required we'll extract the averaged total heating
	heat_rate = sim.solution["Volume-averaged total heating [W.m-3]"].entries
	time = time = sim.solution["Time [s]"].entries
	cell_volume = p["Cell volume [m3]"]

	sim.plot(
		output_variables=[
			"Voltage [V]",
			"Volume-averaged total heating [W.m-3]",
			"Cell temperature [K]"
		]
	)

	# then integrate over the cell volume
	energy = (np.trapezoid(y=heat_rate, x=time) * cell_volume)

	print(f"Total thermal energy is: {energy} Joules")

except Exception as e:
	print(f"Simulation failed because of {e}")