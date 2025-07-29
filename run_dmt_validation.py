import json
import subprocess
import os
import itertools
import pandas as pd
import random
from dosa.utils import get_divisors

# Set up environment variables for Timeloop
os.environ['PATH'] = os.environ.get('PATH', '') + ':/root/accelergy-timeloop-infrastructure/src/timeloop/bin:/root/accelergy-timeloop-infrastructure/src/timeloop/build'
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/root/accelergy-timeloop-infrastructure/src/timeloop/lib:/root/accelergy-timeloop-infrastructure/src/timeloop/build'

# --- Define Search Spaces ---

# 1. Fused Groups to be tested (example from ResNet-18)
FUSION_GROUPS_TO_TEST = [
    {
        "group_name": "layer1.0.conv1_relu",
        "layers": ["layer1.0.conv1", "layer1.0.relu"],
        "pattern": ["Conv", "ReLU"],
        "producer_layer": "layer1.0.conv1",
        "consumer_layer": "layer1.0.relu"
    },
    # Add more groups as needed
]

# 2. Hardware Configuration Space
HW_CONFIG_SPACE = {
    "num_pes": [64, 256],
    "l2_scratchpad_size_kb": [256, 512]
}

# 3. Mapping Space (simplified, needs to be dimension-dependent)
# A more robust implementation would dynamically get divisors
MAPPING_SPACE = {
    "K": [16, 32, 64],
    "C": [16, 32, 64]
}

# 4. Workload Dimensions (example for a specific layer)
# In a real scenario, this would be loaded dynamically per layer
WORKLOAD_DIMS = {
    "layer1.0.conv1": {"N": 1, "C": 64, "K": 64, "P": 56, "Q": 56, "R": 3, "S": 3},
    "layer1.0.relu": {"N": 1, "C": 64, "K": 64, "P": 56, "Q": 56, "R": 1, "S": 1} # ReLU output dims match Conv output dims
}


def generate_dynamic_mapping_space(workload_dims, producer_layer):
    """Dynamically generate valid mapping factors based on workload dimensions."""
    dims = workload_dims[producer_layer]
    mapping_space = {}
    
    # For each dimension, get its divisors
    for dim_name, dim_size in dims.items():
        if dim_name in ['N', 'K', 'C', 'P', 'Q', 'R', 'S']:
            divisors = get_divisors(dim_size).tolist()
            # Sample a few divisors for efficiency
            if len(divisors) > 4:
                divisors = random.sample(divisors, 4)
            mapping_space[dim_name] = divisors
    
    return mapping_space

def generate_complete_dimension_factors(dim_size, num_pes_sqrt):
    """Generate complete factor decomposition for a dimension across all levels.
    
    Args:
        dim_size (int): The problem dimension size to factorize.
        num_pes_sqrt (int): Square root of the number of PEs, used to constrain spatial factors.
        
    Returns:
        dict: A dictionary with factors for each level ['spatial', 'L0', 'L1', 'L2', 'DRAM'].
    """
    if dim_size == 1:
        return {'spatial': 1, 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': 1}
    
    # Initialize factors with default values
    factors = {'spatial': 1, 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': 1}
    
    # Get all prime factors of the dimension size
    def get_prime_factors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors
    
    prime_factors = get_prime_factors(dim_size)
    
    # If no prime factors (dim_size is 1), return default factors
    if not prime_factors:
        return factors
    
    # Sort prime factors in descending order for better distribution
    prime_factors.sort(reverse=True)
    
    # First, assign spatial factors (limited by num_pes_sqrt)
    spatial_product = 1
    remaining_factors = []
    
    for factor in prime_factors:
        if spatial_product * factor <= num_pes_sqrt:
            spatial_product *= factor
        else:
            remaining_factors.append(factor)
    
    factors['spatial'] = spatial_product
    
    # If there are no remaining factors, assign 1 to all memory levels
    if not remaining_factors:
        return factors
    
    # Distribute remaining factors among memory levels
    # We'll use a more deterministic approach to ensure all factors are used
    memory_levels = ['L0', 'L1', 'L2', 'DRAM']
    level_index = 0
    
    for factor in remaining_factors:
        factors[memory_levels[level_index]] *= factor
        level_index = (level_index + 1) % len(memory_levels)
    
    # Verification
    product = 1
    for val in factors.values():
        product *= val
    
    # If verification fails, use a direct approach to ensure correctness
    if product != dim_size:
        # Reset factors
        factors = {'spatial': 1, 'L0': 1, 'L1': 1, 'L2': 1, 'DRAM': dim_size}
        
        # Try to assign a valid spatial factor
        divisors = get_divisors(dim_size).tolist()
        spatial_candidates = [d for d in divisors if d <= num_pes_sqrt]
        
        if spatial_candidates:
            factors['spatial'] = spatial_candidates[-1]  # Take the largest valid divisor
            factors['DRAM'] = dim_size // factors['spatial']
    
    # Final verification
    product = 1
    for val in factors.values():
        product *= val
    assert product == dim_size, f"Factor product {product} does not match dim_size {dim_size}"
    
    return factors

def generate_configurations():
    """Generates a stream of unique configurations with complete dimension factorization."""
    hw_keys, hw_values = zip(*HW_CONFIG_SPACE.items())
    
    # Define some typical permutation patterns
    permutation_patterns = [
        'K C N P Q S R',
        'N K C P Q S R', 
        'C K N P Q S R',
        'K N C P Q S R'
    ]
    
    for group in FUSION_GROUPS_TO_TEST:
        producer_layer = group["producer_layer"]
        dims = WORKLOAD_DIMS[producer_layer]
        
        for hw_instance in itertools.product(*hw_values):
            hardware_config = dict(zip(hw_keys, hw_instance))
            
            # Calculate PE mesh size for spatial constraints
            pe_mesh_size = int(hardware_config['num_pes'] ** 0.5)
            
            # Generate multiple mapping configurations for each hardware config
            for _ in range(2):  # Generate 2 mapping variants per hardware config
                
                # Generate complete factorization for each dimension
                dim_factors = {}
                for dim_name, dim_size in dims.items():
                    if dim_name in ['N', 'K', 'C', 'P', 'Q', 'R', 'S']:
                        dim_factors[dim_name] = generate_complete_dimension_factors(dim_size, pe_mesh_size)
                
                # Build mapping config with complete factors
                mapping_config = {
                    producer_layer: {
                        'DRAM': {
                            'temporal': {dim: factors['DRAM'] for dim, factors in dim_factors.items()},
                            'permutation': random.choice(permutation_patterns)
                        },
                        'L2_Scratchpad': {
                            'temporal': {dim: factors['L2'] for dim, factors in dim_factors.items()},
                            'permutation': random.choice(permutation_patterns)
                        },
                        'L1_Accumulator': {
                            'temporal': {dim: factors['L1'] for dim, factors in dim_factors.items()},
                            'permutation': random.choice(permutation_patterns)
                        },
                        'L0_Registers': {
                            'temporal': {dim: factors['L0'] for dim, factors in dim_factors.items()},
                            'permutation': random.choice(permutation_patterns)
                        },
                        'PE_array': {
                            'spatial': {
                                dim: factors['spatial']
                                for dim, factors in dim_factors.items()
                                if dim in ['K', 'C']
                            },
                            'permutation': random.choice(permutation_patterns)
                        }
                    }
                }
                
                # No need to adjust spatial factors as they are already constrained in generate_complete_dimension_factors

                yield {
                    "fusion_group_info": group,
                    "hardware_config": hardware_config,
                    "mapping_config": mapping_config,
                    "workload_dims": WORKLOAD_DIMS
                }

def main():
    """Main control script to run DMT validation experiments."""
    output_csv_path = "dmt_validation_results.csv"
    temp_config_path = "temp_config.json"
    all_results = []

    print("Starting DMT validation run...")

    for i, config in enumerate(generate_configurations()):
        print(f"--- Running Validation Point {i+1} ---")
        
        # 1. Write the temporary config file
        with open(temp_config_path, 'w') as f:
            json.dump(config, f, indent=4)

        # 2. Run the validation script
        try:
            # Get the absolute path to validate_dmt.py
            script_dir = os.path.dirname(os.path.abspath(__file__))
            validate_script = os.path.join(script_dir, 'validate_dmt.py')
            
            # Use 'conda run' to ensure the script executes within the correct environment
            cmd = [
                'conda',
                'run',
                '-n',
                'dosa',
                '--no-capture-output',
                'python',
                validate_script,
                '--config',
                os.path.abspath(temp_config_path)
            ]
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, check=True, timeout=300
            )
            # Extract JSON output using delimiters
            stdout_content = result.stdout.strip()
            start_marker = "---DMT_VALIDATION_RESULT_START---"
            end_marker = "---DMT_VALIDATION_RESULT_END---"
            
            start_idx = stdout_content.find(start_marker)
            end_idx = stdout_content.find(end_marker)
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("Result delimiters not found in stdout")
            
            json_output = stdout_content[start_idx + len(start_marker):end_idx].strip()
            validation_data = json.loads(json_output)
            
            # Flatten the result for CSV logging
            flat_result = {
                **config['fusion_group_info'],
                **config['hardware_config'],
                "predicted_latency": validation_data['prediction']['latency'],
                "simulated_latency": validation_data['simulation']['latency'],
                "predicted_energy": validation_data['prediction']['energy'],
                "simulated_energy": validation_data['simulation']['energy'],
            }
            all_results.append(flat_result)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to validate config {i+1}: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"[STDERR]:\n{e.stderr}")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"[STDOUT]:\n{e.stdout}")
            # Add a fallback result with -1 values to continue processing
            flat_result = {
                **config['fusion_group_info'],
                **config['hardware_config'],
                "predicted_latency": -1.0,
                "simulated_latency": -1.0,
                "predicted_energy": -1.0,
                "simulated_energy": -1.0,
            }
            all_results.append(flat_result)

    # 3. Save results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv_path, index=False)
        print(f"\nValidation complete. Results saved to {output_csv_path}")
    else:
        print("\nValidation run finished, but no results were collected.")

    # Clean up the temporary file
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

if __name__ == "__main__":
    main()