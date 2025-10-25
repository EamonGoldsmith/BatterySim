import os
import shutil
import pandas as pd
os.environ["PYBAMM_USE_JAX"] = "0"
import pybamm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import math
from tqdm import tqdm

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import subprocess

# Constants
TOTAL_ENERGY_Wh = 6000  # Wh
TOTAL_VOLTAGE_V = 600
CONTINUOUS_CURRENT_A = 60
SEGMENT_VOLTAGE_MAX_V = 120
SEGMENT_ENERGY_MAX_MJ = 6
SEGMENT_ENERGY_MAX_Wh = SEGMENT_ENERGY_MAX_MJ * 1000 / 3.6  # onvert to Wh

# typical values
CELL_PARAMS = {
    "NMC": {
        "nominal_voltage": 3.7,
        "capacity_ah": 3.5,
        "max_discharge_current": 35,  # 10C for 3.5Ah cell
        "energy_density_wh_kg": 250,
        "weight_kg": 0.048  # 18650 cell
    },
    "LFP": {
        "nominal_voltage": 3.2,
        "capacity_ah": 3.2,
        "max_discharge_current": 32,  # 10C for 3.2Ah cell
        "energy_density_wh_kg": 180,
        "weight_kg": 0.050
    },
    # from the MOLICEL datasheet
    # https://www.molicel.com/wp-content/uploads/INR21700P45B_1.2_Product-Data-Sheet-of-INR-21700-P45B-80109.pdf
    "MOLI": {
        "nominal_voltage": 3.6,
        "capacity_ah": 4.5,
        "max_discharge_current": 13.5,
        "energy_density_wh_kg": 242,
        "weight_kg": 0.070
    }
}

def calculate_possible_topologies():
    results = []
    
    # iterate through possible number of segments (must be at least 5 for 600V/120V)
    for n_segments in range(5, 21):
        segment_voltage = TOTAL_VOLTAGE_V / n_segments
        
        # skip if segment voltage exceeds limit
        if segment_voltage > SEGMENT_VOLTAGE_MAX_V:
            continue
            
        segment_energy_wh = TOTAL_ENERGY_Wh / n_segments
        
        # skip if segment energy exceeds limit
        if segment_energy_wh > SEGMENT_ENERGY_MAX_Wh:
            continue
            
        for chemistry, params in CELL_PARAMS.items():
            v_cell = params["nominal_voltage"]
            c_cell = params["capacity_ah"]
            max_discharge = params["max_discharge_current"]
            
            # calculate cells in series per segment
            s_segment = max(1, round(segment_voltage / v_cell))
            actual_segment_voltage = s_segment * v_cell
            
            # calculate required capacity per segment
            required_capacity_ah = segment_energy_wh / actual_segment_voltage
            
            # calculate parallel cells needed
            p_segment = max(1, round(required_capacity_ah / c_cell))
            actual_capacity_ah = p_segment * c_cell
            actual_segment_energy_wh = actual_segment_voltage * actual_capacity_ah
            
            # calculate current per parallel branch
            current_per_branch = CONTINUOUS_CURRENT_A / p_segment
            
            # check current limits
            if current_per_branch > max_discharge:
                continue
                
            # calculate total cells and pack metrics
            total_cells = n_segments * s_segment * p_segment
            pack_energy_wh = n_segments * actual_segment_energy_wh
            pack_energy_density = pack_energy_wh / (total_cells * params["weight_kg"])
            
            results.append({
                "Chemistry": chemistry,
                "Segments": n_segments,
                "Cells_per_Segment": f"{s_segment}S{p_segment}P",
                "Total_Cells": total_cells,
                "Segment_Voltage_V": round(actual_segment_voltage, 1),
                "Segment_Energy_Wh": round(actual_segment_energy_wh, 1),
                "Segment_Energy_MJ": round(actual_segment_energy_wh * 3.6 / 1000, 3),
                "Pack_Voltage_V": round(n_segments * actual_segment_voltage, 1),
                "Pack_Energy_Wh": round(pack_energy_wh, 1),
                "Current_per_Cell_A": round(current_per_branch, 1),
                "Energy_Density_Wh_kg": round(pack_energy_density, 1),
                "Max_Discharge_Rate": f"{current_per_branch/c_cell:.1f}C"
            })
    
    return pd.DataFrame(results)

def analyze_and_visualize(df):
    if df.empty:
        print("No valid topologies found with given constraints.")
        return
    
    print(f"Found {len(df)} valid topology combinations")
    print("\nTop configurations per chemistry:")
    
    # create output directory
    import os
    os.makedirs("topology_analysis", exist_ok=True)
    
    # save results
    df.to_csv("topology_analysis/all_topologies.csv", index=False)
    print("Saved all_topologies.csv")
    
    # filter and sort best options
    best_nmc = df[df["Chemistry"] == "NMC"].sort_values(
        by=["Total_Cells", "Energy_Density_Wh_kg"], 
        ascending=[True, False]
    ).head(5)
    
    best_lfp = df[df["Chemistry"] == "LFP"].sort_values(
        by=["Total_Cells", "Energy_Density_Wh_kg"], 
        ascending=[True, False]
    ).head(5)

    best_moli = df[df["Chemistry"] == "MOLI"].sort_values(
        by=["Total_Cells", "Energy_Density_Wh_kg"], 
        ascending=[True, False]
    ).head(5)
    
    # combined best
    best_combined = pd.concat([best_nmc, best_lfp, best_moli])
    best_combined.to_csv("topology_analysis/best_topologies.csv", index=False)
    print("Saved best_topologies.csv")
    
    # Visualization 1: Segment configurations
    plt.figure(figsize=(12, 8))
    for chemistry in df["Chemistry"].unique():
        subset = df[df["Chemistry"] == chemistry]
        plt.scatter(
            subset["Segment_Voltage_V"], 
            subset["Segment_Energy_Wh"], 
            s=subset["Segments"]*10,
            label=chemistry,
            alpha=0.7
        )
    
    plt.axhline(y=SEGMENT_ENERGY_MAX_Wh, color='r', linestyle='--', label='Max Segment Energy')
    plt.axvline(x=SEGMENT_VOLTAGE_MAX_V, color='g', linestyle='--', label='Max Segment Voltage')
    plt.title("Segment Voltage vs. Energy for Valid Topologies")
    plt.xlabel("Segment Voltage (V)")
    plt.ylabel("Segment Energy (Wh)")
    plt.legend()
    plt.grid(True)
    plt.savefig("topology_analysis/segment_analysis.png")
    plt.close()
    
    # Visualization 2: Pack efficiency
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    for chemistry in df["Chemistry"].unique():
        subset = df[df["Chemistry"] == chemistry]
        ax1.scatter(
            subset["Total_Cells"], 
            subset["Pack_Energy_Wh"], 
            label=chemistry,
            alpha=0.7
        )
    
    ax1.set_xlabel("Total Cells in Pack")
    ax1.set_ylabel("Pack Energy (Wh)")
    ax1.axhline(y=TOTAL_ENERGY_Wh, color='r', linestyle='--', label='Target Energy')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    for chemistry in df["Chemistry"].unique():
        subset = df[df["Chemistry"] == chemistry]
        ax2.scatter(
            subset["Total_Cells"], 
            subset["Energy_Density_Wh_kg"], 
            marker='x',
            label=f"{chemistry} Density"
        )
    
    ax2.set_ylabel("Energy Density (Wh/kg)")
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.title("Pack Energy and Density vs. Cell Count")
    plt.savefig("topology_analysis/pack_efficiency.png", bbox_inches='tight')
    plt.close()
    
    # Visualization 3: Segment configurations per chemistry
    for chemistry in df["Chemistry"].unique():
        plt.figure(figsize=(10, 6))
        subset = df[df["Chemistry"] == chemistry]
        
        # get unique configurations
        configs = subset[["Cells_per_Segment", "Segments"]].drop_duplicates()
        for _, row in configs.iterrows():
            config_subset = subset[
                (subset["Cells_per_Segment"] == row["Cells_per_Segment"]) & 
                (subset["Segments"] == row["Segments"])
            ]
            plt.scatter(
                config_subset["Segment_Voltage_V"],
                config_subset["Segment_Energy_Wh"],
                s=config_subset["Total_Cells"]/10,
                label=f"{row['Cells_per_Segment']} x {row['Segments']}"
            )
        
        plt.axhline(y=SEGMENT_ENERGY_MAX_Wh, color='r', linestyle='--', label='Max Energy')
        plt.axvline(x=SEGMENT_VOLTAGE_MAX_V, color='g', linestyle='--', label='Max Voltage')
        plt.title(f"Segment Analysis: {chemistry} Chemistry")
        plt.xlabel("Segment Voltage (V)")
        plt.ylabel("Segment Energy (Wh)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"topology_analysis/{chemistry}_segment_analysis.png")
        plt.close()
    
    print("Visualizations saved to topology_analysis folder")

def permutations_results():
    print("Calculating possible battery topologies for accumulator...")
    print(f"Constraints: {TOTAL_ENERGY_Wh}Wh, {TOTAL_VOLTAGE_V}V, {CONTINUOUS_CURRENT_A}A continuous")
    print(f"Segment limits: {SEGMENT_VOLTAGE_MAX_V}V, {SEGMENT_ENERGY_MAX_MJ}MJ ({SEGMENT_ENERGY_MAX_Wh:.1f}Wh)")
    
    df = calculate_possible_topologies()
    
    if not df.empty:
        print("\nTop 5 configurations:")
        print(df.head(5).to_string(index=False))
        
        # analysis and visualization
        analyze_and_visualize(df)
    else:
        print("No valid topologies found with current constraints. Consider relaxing constraints.")


def setup_output_dirs(base_dir="battery_simulation_results"):
    """create directory structure for results"""
    dirs = {
        "base": base_dir,
        "figures": os.path.join(base_dir, "figures"),
        "data": os.path.join(base_dir, "data"),
        "comparisons": os.path.join(base_dir, "comparisons")
    }
    
    # create directories
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    return dirs

def simulate_topology(chemistry, topology, cooling_rate, discharge_current):
    """simulate battery pack performance with given parameters"""
    # load parameter sets
    try:
        if chemistry == "NMC":
            param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
        elif chemistry == "LFP":
            param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
        elif chemistry == "MOLI":
            # we'll start with a bundled parameter set and modify it later
            param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
        else:
            param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
    except:
        # fallback to default parameters
        param = pybamm.ParameterValues("Chen2020")
    
    # set cell-level parameters for possible cells
    param.update({
        "Total heat transfer coefficient [W.m-2.K-1]": cooling_rate,
        "Cell cooling surface area [m2]": 0.035,   # typical 18650 cell
        "Cell volume [m3]": 1.7e-4,                # ~17 cm^3 for 18650
        "Ambient temperature [K]": 298.15,         # 25 C
        "Initial temperature [K]": 298.15,
        "Current function [A]": discharge_current / topology[1],  # current per cell
        "Cell capacity [A.h]": 3.5,        # 3.5 Ah typical high-performance cell
        "Number of electrodes connected in parallel to make a cell": 1,
        "Negative electrode thickness [m]": 8.5e-5,
        "Positive electrode thickness [m]": 7.5e-5
    })

    # for model with thermal effects
    # enhanced single particle model, https://www.mdpi.com/2313-0105/9/10/511
    model = pybamm.lithium_ion.SPMe({"thermal": "lumped"})
    
    # explicitly specify CasADi solver, the other ones didn't work for me
    solver = pybamm.CasadiSolver(mode="safe", rtol=1e-3, atol=1e-5)
    sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

    t_eval = np.linspace(0, 3600, 100)
    
    try:
        solution = sim.solve(t_eval)
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        return None, None, None, None

    # extract results
    time = solution["Time [s]"].entries
    current_draw = np.full_like(time, discharge_current)  # Constant 60 A pack current
    mod_current_draw(current_draw)
    pack_voltage = solution["Terminal voltage [V]"].entries * topology[0]
    temperature = solution["Volume-averaged cell temperature [K]"].entries - 273.15
    
    return time, current_draw, pack_voltage, temperature

def mod_current_draw(current):
    for i, c in enumerate(current):
        current[i] = math.sin((i/10) * 2 * math.pi)*30 + 60

    print(current)

def save_individual_plot(output_dir, config_label, time, current, voltage, temp):
    """save individual plot for this configuration"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # current plot
    axs[0].plot(time, current, 'b-')
    axs[0].set_title(f"Pack Current Draw - {config_label}")
    axs[0].set_ylabel("Current [A]")
    axs[0].grid(True)
    axs[0].set_ylim([0, max(current)*1.1])
    
    # voltage plot
    axs[1].plot(time, voltage, 'r-')
    axs[1].set_title(f"Pack Voltage - {config_label}")
    axs[1].set_ylabel("Voltage [V]")
    axs[1].grid(True)
    
    # temperature plot
    axs[2].plot(time, temp, 'g-')
    axs[2].set_title(f"Pack Temperature - {config_label}")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Temperature [°C]")
    axs[2].grid(True)
    axs[2].axhline(y=60, color='r', linestyle='--', label='Safety Limit')
    axs[2].legend()
    
    plt.tight_layout()
    safe_label = config_label.replace(" ", "_").replace("/", "_")
    fig_path = os.path.join(output_dir["figures"], f"{safe_label}.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path

def compare_topologies(output_dir):
    """compare different battery configurations"""
    # test configurations
    configurations = [
        {"chemistry": "NMC", "topology": (140, 4), "cooling": "natural", "discharge": 60},
        {"chemistry": "NMC", "topology": (140, 4), "cooling": "forced_air", "discharge": 60},
        {"chemistry": "NMC", "topology": (140, 4), "cooling": "liquid", "discharge": 60},
        {"chemistry": "LFP", "topology": (160, 6), "cooling": "natural", "discharge": 60},
        {"chemistry": "LFP", "topology": (160, 6), "cooling": "forced_air", "discharge": 60},
        {"chemistry": "LFP", "topology": (160, 6), "cooling": "liquid", "discharge": 60},
        {"chemistry": "MOLI", "topology": (120, 8), "cooling": "natural1", "discharge": 60},
        {"chemistry": "MOLI", "topology": (120, 8), "cooling": "natural2", "discharge": 60},
        {"chemistry": "MOLI", "topology": (120, 8), "cooling": "natural3", "discharge": 60},
    ]

    # cooling rates [W/m^2 * K]
    # these are estimates but can be more accurately adjusted to measured values when the accumulator is built
    cooling_rates = {
        "natural": 5,       # passive cooling
        "natural1": 5,      # different labels so figures don't overwrite themselves
        "natural2": 5,
        "natural3": 5,
        "forced_air": 50,   # active air cooling
        "liquid": 1000,     # liquid cooling
    }

    # combined comparison figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    results = []
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'b', 'b', 'b', 'b']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', '-.', '-.', '-.', '-.']
    plot_data = []

    for i, config in enumerate(tqdm(configurations, desc="Simulating")):

        cooling_rate = cooling_rates[config["cooling"]]
        config_label = (f"{config['chemistry']} {config['topology'][0]}s{config['topology'][1]}p "
                        f"{config['cooling']} cooling")

        time, current, voltage, temp = simulate_topology(
            config["chemistry"],
            config["topology"],
            cooling_rate,
            config["discharge"]
        )
        
        if time is None:
            print(f"(WARNING) Skipping {config_label} due to simulation failure")
            continue
            
        # calculate pack metrics
        nominal_voltage = np.mean(voltage)
        capacity_ah = (6000 / nominal_voltage)  # 6 kWh capacity
        energy_density = 6000 / (config["topology"][0] * config["topology"][1] * 0.048)  # Wh/kg
        
        results.append({
            "label": config_label,
            "peak_temp": np.max(temp),
            "avg_temp": np.mean(temp),
            "voltage_drop": voltage[0] - voltage[-1],
            "min_voltage": np.min(voltage),
            "capacity": capacity_ah,
            "energy_density": energy_density
        })
        
        # save individual plot
        plot_file = save_individual_plot(output_dir, config_label, time, current, voltage, temp)
        plot_data.append((time, current, voltage, temp, colors[i], linestyles[i], config_label))
        
        # add to combined plot
        axs[0].plot(time, current, color=colors[i], linestyle=linestyles[i], label=config_label)
        axs[1].plot(time, voltage, color=colors[i], linestyle=linestyles[i])
        axs[2].plot(time, temp, color=colors[i], linestyle=linestyles[i])

    # format combined plots
    axs[0].set_title("Battery Current Draw Comparison")
    axs[0].set_ylabel("Current [A]")
    axs[0].grid(True)
    axs[0].legend(loc='upper right', fontsize=8)
    
    axs[1].set_title("Pack Voltage Comparison")
    axs[1].set_ylabel("Voltage [V]")
    axs[1].grid(True)
    
    axs[2].set_title("Pack Temperature Comparison")
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel("Temperature [°C]")
    axs[2].grid(True)
    axs[2].axhline(y=60, color='r', linestyle='--', label='Safety Limit')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    comp_path = os.path.join(output_dir["comparisons"], "topology_comparison.png")
    plt.savefig(comp_path)
    plt.close()
    print(f"Saved comparison plot: {comp_path}")

    # save results to CSV
    csv_path = os.path.join(output_dir["data"], "simulation_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV data: {csv_path}")

    # print performance comparison
    if results:
        print("\nPerformance Summary:")
        print(f"{'Configuration':<45} | {'Peak':>5} | {'Avg':>5} | {'Volt':>5} | {'Min':>5} | {'Cap':>5} | {'Energy'}")
        print(f"{'':<45} | {'Temp':>5} | {'Temp':>5} | {'Drop':>5} | {'Volt':>5} | {'(Ah)':>5} | {'(Wh/kg)'}")
        print("-" * 95)
        for r in results:
            print(f"{r['label']:<45} | {r['peak_temp']:>5.1f} | {r['avg_temp']:>5.1f} | {r['voltage_drop']:>5.1f} | {r['min_voltage']:>5.1f} | {r['capacity']:>5.1f} | {r['energy_density']:>6.1f}")
    else:
        print("No simulations completed successfully")

    return results, plot_data

if __name__ == "__main__":
    pybamm.set_logging_level("ERROR")
    
    # use environment variable for output if running in container
    output_base = os.getenv("OUTPUT_DIR", "battery_simulation_results")
    output_dirs = setup_output_dirs(output_base)

    # debug, run simulation:
    results, plot_data = compare_topologies(output_dirs)