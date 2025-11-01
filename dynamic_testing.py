
import pybamm
import numpy as np

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

def dynamic_current_func(A, omega):
    def current(t):
        return A * pybamm.sin(2 * np.pi * omega * t)

    return current

p["Current function [A]"] = dynamic_current_func(model.param.I_typ, 0.1)

solver = pybamm.CasadiSolver(rtol=1e-9, atol=1e-9)
sim = pybamm.Simulation(model, parameter_values=p, solver=solver)

t_eval = np.linspace(0, 360, 500)

# run simulation
try:
	sim.solve(t_eval)
	output_variables = ["Current [A]", "Voltage [V]"]
	sim.plot(output_variables)

except Exception as e:
	print(f"Simulation failed because of {e}")