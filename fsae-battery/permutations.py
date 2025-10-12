import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Constants
TOTAL_ENERGY_Wh = 6000  # 6kWh
TOTAL_VOLTAGE_V = 600
CONTINUOUS_CURRENT_A = 60
SEGMENT_VOLTAGE_MAX_V = 120
SEGMENT_ENERGY_MAX_MJ = 6
SEGMENT_ENERGY_MAX_Wh = SEGMENT_ENERGY_MAX_MJ * 1000 / 3.6  # Convert MJ to Wh

# Cell parameters (typical values for high-performance cells)
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
        "weight_kg": 0.050  # Similar form factor
    }
}

def calculate_possible_topologies():
    results = []
    
    # Iterate through possible number of segments (must be at least 5 for 600V/120V)
    for n_segments in range(5, 21):  # From 5 to 20 segments
        segment_voltage = TOTAL_VOLTAGE_V / n_segments
        
        # Skip if segment voltage exceeds limit
        if segment_voltage > SEGMENT_VOLTAGE_MAX_V:
            continue
            
        segment_energy_wh = TOTAL_ENERGY_Wh / n_segments
        
        # Skip if segment energy exceeds limit
        if segment_energy_wh > SEGMENT_ENERGY_MAX_Wh:
            continue
            
        for chemistry, params in CELL_PARAMS.items():
            v_cell = params["nominal_voltage"]
            c_cell = params["capacity_ah"]
            max_discharge = params["max_discharge_current"]
            
            # Calculate cells in series per segment
            s_segment = max(1, round(segment_voltage / v_cell))
            actual_segment_voltage = s_segment * v_cell
            
            # Calculate required capacity per segment
            required_capacity_ah = segment_energy_wh / actual_segment_voltage
            
            # Calculate parallel cells needed
            p_segment = max(1, round(required_capacity_ah / c_cell))
            actual_capacity_ah = p_segment * c_cell
            actual_segment_energy_wh = actual_segment_voltage * actual_capacity_ah
            
            # Calculate current per parallel branch
            current_per_branch = CONTINUOUS_CURRENT_A / p_segment
            
            # Check current limits
            if current_per_branch > max_discharge:
                continue
                
            # Calculate total cells and pack metrics
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
    
    # Create output directory
    import os
    os.makedirs("topology_analysis", exist_ok=True)
    
    # Save full results
    df.to_csv("topology_analysis/all_topologies.csv", index=False)
    print("Saved all_topologies.csv")
    
    # Filter and sort best options
    best_nmc = df[df["Chemistry"] == "NMC"].sort_values(
        by=["Total_Cells", "Energy_Density_Wh_kg"], 
        ascending=[True, False]
    ).head(5)
    
    best_lfp = df[df["Chemistry"] == "LFP"].sort_values(
        by=["Total_Cells", "Energy_Density_Wh_kg"], 
        ascending=[True, False]
    ).head(5)
    
    # Combined best options
    best_combined = pd.concat([best_nmc, best_lfp])
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
        
        # Get unique configurations
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

if __name__ == "__main__":
    print("Calculating possible battery topologies for FSAE accumulator...")
    print(f"Constraints: {TOTAL_ENERGY_Wh}Wh, {TOTAL_VOLTAGE_V}V, {CONTINUOUS_CURRENT_A}A continuous")
    print(f"Segment limits: {SEGMENT_VOLTAGE_MAX_V}V, {SEGMENT_ENERGY_MAX_MJ}MJ ({SEGMENT_ENERGY_MAX_Wh:.1f}Wh)")
    
    df = calculate_possible_topologies()
    
    if not df.empty:
        print("\nTop 5 configurations:")
        print(df.head(5).to_string(index=False))
        
        # Analysis and visualization
        analyze_and_visualize(df)
        
        # Design recommendations
        print("\nDesign Recommendations:")
        print("1. Higher segment counts reduce segment energy but increase complexity")
        print("2. NMC chemistry provides better energy density but requires more thermal management")
        print("3. LFP chemistry offers safer thermal performance at the cost of density")
        print("4. Aim for 5-8 segments to balance complexity and performance")
        print("5. Verify all designs meet FSAE rules for accumulator construction")
        print("6. Consider mechanical constraints when arranging segments in the vehicle")
    else:
        print("No valid topologies found with current constraints. Consider relaxing constraints.")
