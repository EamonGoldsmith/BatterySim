import pybamm
import numpy as np

# specify simulation type
model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})

# setup cell parameters
p = pybamm.ParameterValues("Chen2020")

experiment = pybamm.Experiment(
    [
        "Discharge at C/10 for 10 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour",
    ]
)

solver = pybamm.CasadiSolver(rtol=1e-9, atol=1e-9)
sim = pybamm.Simulation(model, parameter_values=p, solver=solver, experiment=experiment)

sim.solve()
sim.plot()