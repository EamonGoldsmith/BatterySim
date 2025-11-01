import pybamm
import numpy as np

# specify simulation type
model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
solver = pybamm.CasadiSolver(rtol=1e-9, atol=1e-9)

p = pybamm.ParameterValues("Chen2020")

experiment = pybamm.Experiment(
    [
        "Discharge at 1 A until 3.5 V",
		"Rest for 30 minutes",
		"Charge at 1 A until 4.1 V",
		"Discharge at 2 A until 3.5 V",
		"Charge at 1 A until 4.1 V",
		"Discharge at 4 A until 3.5 V",
		"Charge at 1 A until 4.1 V",
		"Discharge at 6 A until 3.5 V",
    ]
)

sim = pybamm.Simulation(model, solver=solver, experiment=experiment)

# run simulation
try:
	sim.solve(initial_soc=1.0)
	
	output_variables = ["Terminal voltage [V]", "Volume-averaged cell temperature [K]"]
	sim.plot(output_variables=output_variables)

	print(sim.solution["Time [h]"].entries[-1])

except Exception as e:
	print(f"Simulation failed because of {e}")