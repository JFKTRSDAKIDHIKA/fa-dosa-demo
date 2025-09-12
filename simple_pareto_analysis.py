import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Keep any point dominated by c
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            # And keep self
            is_efficient[i] = True
    return is_efficient

def analyze_pareto(output_dir):
    """
    Analyzes the experiment data to find and visualize the Pareto frontier.
    """
    results_path = Path(output_dir)
    trials_file = results_path / "trials.csv"

    try:
        df = pd.read_csv(trials_file)
    except FileNotFoundError:
        print(f"Error: The file {trials_file} was not found.")
        return

    # Extract the objectives for Pareto analysis.
    # We want to minimize both edp and area_mm2.
    costs = df[['edp', 'area_mm2']].values

    # Find the Pareto-efficient points
    pareto_mask = is_pareto_efficient(costs)
    pareto_points = df[pareto_mask]

    print("Pareto Frontier Analysis")
    print("========================")
    print(f"Total data points: {len(df)}")
    print(f"Pareto optimal points found: {len(pareto_points)}")
    print("\nPareto Optimal Points:")
    print(pareto_points)

    # Generate a plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['area_mm2'], df['edp'], c='blue', label='All Points', alpha=0.5)
    plt.scatter(pareto_points['area_mm2'], pareto_points['edp'], c='red', marker='x', s=100, label='Pareto Frontier')
    plt.xlabel('Area (mm^2)')
    plt.ylabel('EDP (Energy-Delay Product)')
    plt.title('Pareto Frontier: EDP vs. Area')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = results_path / "pareto_frontier.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "/root/fa-dosa-demo/output/pareto_example"
    
    analyze_pareto(results_dir)