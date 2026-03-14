"""
Generate Synthetic Hygrothermal-Mechanical Coupled FEM Data
for Biological Materials (AAC, Cardboard, Bio-composites)

Simulates moisture diffusion + mechanical degradation coupling
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

class HygrothermalMechanicalSimulator:
    """
    Simulates moisture-induced mechanical property degradation 
    in biological materials following coupled diffusion-mechanics models.
    """
    
    def __init__(self):
        self.data_path = Path(__file__).parent
        
    def moisture_diffusion_model(self, t, c, D, c0, c_sat):
        """Simple diffusion model: dc/dt = D * d²c/dx² (Fick's law)"""
        # Surface moisture content (boundary condition)
        c_surface = c_sat
        # Simplified 1D diffusion: exponential saturation
        c_eq = c_sat - (c_sat - c0) * np.exp(-D * t / 0.01)  # characteristic length = 0.01 m
        return np.clip(c_eq, c0, c_sat)
    
    def modulus_degradation(self, moisture_content, E0, alpha_E):
        """
        Young's modulus degradation with moisture saturation
        E(moisture) = E0 * (1 - alpha_E * (moisture_content / 100))
        """
        degradation_factor = 1 - alpha_E * (moisture_content / 100)
        return E0 * np.maximum(degradation_factor, 0.3)  # Minimum 30% stiffness
    
    def generate_single_simulation(self, params):
        """Generate one FEM simulation outcome"""
        porosity = params['porosity']
        density = params['density']
        thermal_cond = params['thermal_cond']
        diffusivity = params['moisture_diffusivity']
        E0_dry = params['E0_dry']
        alpha_E = params['E_sensitivity_to_moisture']
        nu = params['nu']
        RH_exposure = params['RH_exposure']
        temperature = params['temperature']
        thickness = params['thickness']
        load_magnitude = params['load_magnitude']
        exposure_time_days = params['exposure_time_days']
        
        # Moisture saturation based on RH (simplified: RH=100% → 100% saturation)
        c_sat = RH_exposure  # % moisture content at saturation
        c0 = 5.0  # Initial moisture content (%)
        
        # Time-dependent moisture uptake
        t_days = exposure_time_days
        t_seconds = t_days * 24 * 3600
        
        # Moisture profile (exponential saturation)
        moisture_content = self.moisture_diffusion_model(t_seconds, c0, diffusivity, c0, c_sat)
        
        # Mechanical degradation (moisture-dependent)
        E_effective = self.modulus_degradation(moisture_content, E0_dry, alpha_E)
        
        # Stress under mechanical load (simplified: σ = F/A, accounting for stiffness)
        # Stress = load_magnitude / (porosity effects)
        sigma_max = load_magnitude / (1 - porosity + 0.1)  # Lower porosity → higher stress
        
        # Strain = σ / E
        strain = sigma_max / E_effective if E_effective > 0 else 0
        
        # Damage index (Hasselman criterion simplified)
        # Damage increases with moisture and stress
        damage_index = (moisture_content / 100) * (sigma_max / 10) * alpha_E
        
        return {
            'moisture_content_avg': moisture_content,
            'E_effective_wet': E_effective,
            'principal_stress_max': sigma_max,
            'strain_max': strain,
            'damage_index': np.clip(damage_index, 0, 1),
            'G_effective': E_effective / (2 * (1 + nu))  # Shear modulus
        }
    
    def generate_dataset(self, n_samples=250):
        """Generate synthetic FEM training dataset"""
        print(f"Generating {n_samples} synthetic FEM simulations...\n")
        
        # Parameter ranges (inspired by biological materials: AAC, cardboard, bio-composites)
        param_ranges = {
            'porosity': np.linspace(0.3, 0.7, n_samples // 5),
            'density': np.linspace(400, 1000, n_samples // 5),
            'thermal_cond': np.linspace(0.08, 0.3, n_samples // 5),
            'moisture_diffusivity': np.logspace(-8, -6, n_samples // 5),
            'E0_dry': np.linspace(2.0, 8.0, n_samples // 5),
            'E_sensitivity_to_moisture': np.linspace(0.15, 0.5, n_samples // 5),
            'nu': np.linspace(0.2, 0.35, n_samples // 5),
            'RH_exposure': np.linspace(30, 95, n_samples // 5),
            'temperature': np.linspace(15, 35, n_samples // 5),
            'thickness': np.linspace(5, 25, n_samples // 5),
            'load_magnitude': np.linspace(0.5, 5.0, n_samples // 5),
            'exposure_time_days': np.linspace(1, 60, n_samples // 5)
        }
        
        # Create Latin Hypercube Sampling for better coverage
        params_list = []
        for i in range(n_samples):
            params = {}
            for key, values in param_ranges.items():
                idx = i % len(values)
                # Add small random perturbation
                params[key] = values[idx] + np.random.normal(0, values[idx] * 0.05)
            params_list.append(params)
        
        # Run simulations
        results = []
        for i, params in enumerate(params_list):
            if (i + 1) % 50 == 0:
                print(f"  Simulation {i+1}/{n_samples} completed")
            
            sim_result = self.generate_single_simulation(params)
            row = {**params, **sim_result}
            results.append(row)
        
        df = pd.DataFrame(results)
        
        # Save dataset
        output_file = self.data_path / 'fem_coupled_hygrothermal_mechanical.csv'
        df.to_csv(output_file, index=False)
        print(f"\n✓ Dataset saved: {output_file}")
        print(f"  Shape: {df.shape}")
        print(f"\nDataset Summary:")
        print(df.describe().round(4))
        
        return df

if __name__ == "__main__":
    simulator = HygrothermalMechanicalSimulator()
    df = simulator.generate_dataset(n_samples=250)
