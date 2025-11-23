import torch
import torch.nn as nn
import math

class ImprovedPhysicsInformedNN(nn.Module):
    """
    Physics-informed neural network for aluminum alloy deformation.
    
    Combines constitutive equations with neural networks:
    - Dislocation evolution: Kocks-Mecking formulation
    - Grain growth: Thermally activated boundary migration  
    - Precipitation: JMAK kinetics
    - Stress: Taylor + Hall-Petch + precipitation hardening
    
    All rates follow Arrhenius form: K = K₀ exp(-Q/RT)
    """
    
    def __init__(self, input_dim=3, hidden_dim=64):
        super(ImprovedPhysicsInformedNN, self).__init__()

        # Constants
        self.R = 8.314  # J/mol·K
        self.T_ref = nn.Parameter(torch.tensor(400.0))  # °C
        
        # Activation energies (J/mol) - trainable
        self.Q_gen = nn.Parameter(torch.tensor(40000.0))   # Dislocation generation
        self.Q_rec = nn.Parameter(torch.tensor(50000.0))   # Recovery
        self.Q_g = nn.Parameter(torch.tensor(45000.0))     # Grain growth
        self.Q_X = nn.Parameter(torch.tensor(35000.0))     # Precipitation
        
        # Dislocation parameters
        self.K0_gen = nn.Parameter(torch.tensor(1.0e10))   # 1/s
        self.K0_rec = nn.Parameter(torch.tensor(0.01))     # 1/s
        self.rho_init = nn.Parameter(torch.tensor(1.0e12)) # 1/m²
        self.rho_sat = nn.Parameter(torch.tensor(1.0e14))  # 1/m²
        
        # Grain parameters
        self.K0_d = nn.Parameter(torch.tensor(1.0e-6))     # m²/s
        self.d_init = nn.Parameter(torch.tensor(20.0))     # µm
        self.d_ss = nn.Parameter(torch.tensor(10.0))       # µm
        
        # Precipitation parameters
        self.K0_X = nn.Parameter(torch.tensor(1.0e-4))     # 1/s
        self.X_init = nn.Parameter(torch.tensor(0.01))
        self.X_eq_coef = nn.Parameter(torch.tensor(0.1))
        
        # Stress parameters
        self.sigma_0 = nn.Parameter(torch.tensor(20.0))    # MPa
        self.k_rho = nn.Parameter(torch.tensor(0.1))       # Taylor coefficient
        self.k_d = nn.Parameter(torch.tensor(50.0))        # Hall-Petch
        self.k_X = nn.Parameter(torch.tensor(10.0))        # Precipitation
        self.k_T = nn.Parameter(torch.tensor(0.1))         # Temperature softening
        self.k1 = nn.Parameter(torch.tensor(150.0))        # Strain hardening
        self.strain_exp = nn.Parameter(torch.tensor(0.6))
        
        # Thermodynamic parameters
        self.E_disloc = nn.Parameter(torch.tensor(0.1))
        self.E_grain = nn.Parameter(torch.tensor(0.05))
        self.E_precip = nn.Parameter(torch.tensor(0.02))
        self.E_temp = nn.Parameter(torch.tensor(0.01))
        
        # Neural network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ReLU()
        )

        self.temperature_branch = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.stress_branch = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.grain_size_branch = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.delta_grain_branch = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dislocation_density_branch = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.precip_fraction_branch = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.free_energy_branch = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dissipation_branch = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def arrhenius_factor(self, temperature, activation_energy):
        """
        K = exp(-Q/RT)
        
        Args:
            temperature: Tensor [°C]
            activation_energy: Scalar [J/mol]
        
        Returns:
            K: Arrhenius factor tensor
        """
        safe_temp = torch.clamp(temperature + 273.15, min=273.15)
        exp_term = torch.clamp(-activation_energy / (self.R * safe_temp), min=-30.0, max=30.0)
        return torch.exp(exp_term)

    def calculate_physics_temperature(self, strain, time, cooling_rate):
        """
        T = T_ref - ṙ·t + ΔT_strain
        
        Args:
            strain, time, cooling_rate: Tensors
        
        Returns:
            T: Temperature tensor [°C]
        """
        base_temp = self.T_ref - cooling_rate * time
        strain_heating = 20.0 * strain * torch.exp(-0.1 * time)
        physics_temp = base_temp + strain_heating
        return torch.clamp(physics_temp, min=40.0, max=600.0)

    def calculate_physics_dislocation_density(self, strain, time, temperature):
        """
        Simplified: ρ = ρ₀ + K_gen·ε̇ - K_rec·ε
        
        Args:
            strain, time, temperature: Tensors
        
        Returns:
            ρ: Dislocation density [1/m²]
        """
        K_gen = self.K0_gen * self.arrhenius_factor(temperature, self.Q_gen)
        K_rec = self.K0_rec * self.arrhenius_factor(temperature, self.Q_rec)
        
        strain_rate = strain / (time + 0.1)
        gen_term = K_gen * strain_rate
        rec_term = K_rec * (strain + 0.1)
        rho = self.rho_init + gen_term - rec_term
        
        # Soft clamp via sigmoid
        rho_min = 1.0e10
        rho_max = 1.0e15
        rho_range = rho_max - rho_min
        rho_normalized = (rho - rho_min) / rho_range
        rho_clamped = rho_min + rho_range * torch.sigmoid(2.0 * rho_normalized)
        
        return rho_clamped

    def calculate_physics_grain_size(self, time, temperature, strain):
        """
        d = d₀ + K_d·f(T)·ln(t)·g(ε)
        
        Args:
            time, temperature, strain: Tensors
        
        Returns:
            d: Grain size [µm]
        """
        temp_factor = self.arrhenius_factor(temperature, self.Q_g)
        time_factor = torch.log1p(time + 0.1)
        strain_inhibition = 1.0 / (1.0 + 0.2 * strain)
        grain_size = self.d_init + 2.0 * temp_factor * time_factor * strain_inhibition
        return torch.clamp(grain_size, min=1.0, max=50.0)

    def calculate_physics_delta_grain(self, grain_size, time, temperature, strain):
        """
        Δd = f(T,t,d)
        
        Returns:
            Δd: Change in grain size [µm]
        """
        temp_factor = self.arrhenius_factor(temperature, self.Q_g)
        delta_d = 0.2 * temp_factor * torch.sqrt(time + 0.1) * (self.d_init / (grain_size + 0.1))
        return torch.clamp(delta_d, min=0.1, max=10.0)

    def calculate_physics_precip_fraction(self, time, temperature, strain):
        """
        X = X_eq·(1 - exp(-K_X·t))
        
        Returns:
            X: Precipitation fraction
        """
        X_eq = self.X_eq_coef * (1.0 - torch.exp(-0.01 * (self.T_ref - temperature)))
        X_eq = torch.clamp(X_eq, min=0.01, max=0.5)
        K_X = self.K0_X * self.arrhenius_factor(temperature, self.Q_X)
        precip_fraction = X_eq * (1.0 - torch.exp(-K_X * time))
        return torch.clamp(precip_fraction, min=0.0, max=1.0)

    def calculate_physics_stress(self, strain, temperature, dislocation_density=None, 
                                 grain_size=None, precip_fraction=None):
        """
        σ = σ₀ + k₁·εⁿ - k_T·T + k_ρ·√ρ + k_d/√d + k_X·X
        
        Taylor + Hall-Petch + precipitation hardening
        
        Returns:
            σ: Flow stress [MPa]
        """
        base_stress = self.sigma_0 + self.k1 * torch.pow(strain + 1e-3, self.strain_exp) - self.k_T * temperature
        
        if dislocation_density is not None:
            base_stress = base_stress + self.k_rho * torch.sqrt(dislocation_density)
            
        if grain_size is not None:
            base_stress = base_stress + self.k_d / torch.sqrt(grain_size + 0.1)
            
        if precip_fraction is not None:
            base_stress = base_stress + self.k_X * precip_fraction
        
        return torch.clamp(base_stress, min=1.0)

    def calculate_free_energy(self, strain, temperature, dislocation_density, grain_size, precip_fraction):
        """
        Ψ = E_ρ·ln(ρ/ρ₀) + E_d/√d + E_X·X - E_T·T
        
        Approximate Helmholtz free energy
        
        Returns:
            Ψ: Free energy [arbitrary units]
        """
        dislocation_energy = self.E_disloc * torch.log1p(dislocation_density / 1e12)
        grain_boundary_energy = self.E_grain / torch.sqrt(grain_size + 0.1)
        precip_energy = self.E_precip * precip_fraction
        temp_energy = self.E_temp * temperature
        
        free_energy = dislocation_energy + grain_boundary_energy + precip_energy - temp_energy
        return free_energy

    def calculate_dissipation(self, stress, strain_rate):
        """
        D = σ·ε̇ ≥ 0
        
        Second law requirement
        
        Returns:
            D: Dissipation rate
        """
        return torch.abs(stress * strain_rate)
    
    def estimate_stress_from_energy(self, strain, strain_rate, temperature):
        """
        Thermodynamic stress: σ_thermo ≈ ∂Ψ/∂ε
        
        Simplified without autograd
        """
        base_stress = self.sigma_0 + self.k1 * self.strain_exp * torch.pow(strain + 1e-3, self.strain_exp - 1.0)
        temp_effect = -self.k_T * (1.0 + 0.01 * temperature)
        strain_rate_effect = 10.0 * torch.sqrt(strain_rate + 1e-4)
        thermo_stress = torch.clamp(base_stress + temp_effect + strain_rate_effect, min=1.0)
        return thermo_stress

    def calculate_thermodynamic_consistency(self, stress, thermo_stress, dissipation, free_energy):
        """
        Check three thermodynamic requirements:
        1. σ ≈ σ_thermo (consistency)
        2. D ≥ 0 (second law)
        3. |Ψ| bounded
        
        Returns:
            Three error scalars
        """
        stress_consistency_error = torch.mean(torch.abs(stress - thermo_stress)) / (torch.mean(stress) + 1e-6)
        second_law_error = torch.mean(torch.relu(-dissipation))
        energy_balance_error = torch.mean(torch.relu(torch.abs(free_energy) - 100.0))
        
        return stress_consistency_error, second_law_error, energy_balance_error

    def forward(self, x):
        """
        Forward pass: physics + neural pathways → blended output
        
        Blending weights α:
        - Temperature: α=0.2 (neural-dominated, complex heating)
        - Stress: α=0.4 (balanced)
        - Grain size: α=0.4 (balanced)
        - Microstructure: α=0.7 (physics-dominated)
        
        Returns:
            Dict of predictions + thermodynamic errors
        """
        # Input bounds
        strain = torch.clamp(x[:, 0:1], 0.001, 0.5)
        time = torch.clamp(x[:, 1:2], 0.1, 20.0)
        cooling_rate = torch.clamp(x[:, 2:3], 1.0, 100.0)

        # Neural predictions
        features = self.feature_extractor(x)
        nn_temperature = self.temperature_branch(features)
        nn_stress = self.stress_branch(features)
        nn_grain_size = self.grain_size_branch(features)
        nn_delta_grain = self.delta_grain_branch(features)
        nn_dislocation_density = self.dislocation_density_branch(features)
        nn_precip_fraction = self.precip_fraction_branch(features)
        nn_free_energy = self.free_energy_branch(features)
        nn_dissipation = self.dissipation_branch(features)

        # Physics predictions
        physics_temperature = self.calculate_physics_temperature(strain, time, cooling_rate)
        physics_dislocation_density = self.calculate_physics_dislocation_density(strain, time, physics_temperature)
        physics_grain_size = self.calculate_physics_grain_size(time, physics_temperature, strain)
        physics_delta_grain = self.calculate_physics_delta_grain(physics_grain_size, time, physics_temperature, strain)
        physics_precip_fraction = self.calculate_physics_precip_fraction(time, physics_temperature, strain)
        
        strain_rate = strain / (time + 0.1)
        basic_physics_stress = self.calculate_physics_stress(strain, physics_temperature)
        physics_stress = self.calculate_physics_stress(
            strain, physics_temperature, physics_dislocation_density, 
            physics_grain_size, physics_precip_fraction
        )
        physics_stress = 0.7 * physics_stress + 0.3 * basic_physics_stress
        
        physics_free_energy = self.calculate_free_energy(
            strain, physics_temperature, physics_dislocation_density,
            physics_grain_size, physics_precip_fraction
        )
        physics_dissipation = self.calculate_dissipation(physics_stress, strain_rate)
        thermo_stress = self.estimate_stress_from_energy(strain, strain_rate, physics_temperature)
        
        thermo_errors = self.calculate_thermodynamic_consistency(
            physics_stress, thermo_stress, physics_dissipation, physics_free_energy
        )

        # Blend physics and neural
        temperature = 0.2 * physics_temperature + 0.8 * nn_temperature
        stress = 0.4 * physics_stress + 0.6 * nn_stress
        grain_size = 0.4 * physics_grain_size + 0.6 * nn_grain_size
        delta_grain = 0.4 * physics_delta_grain + 0.6 * nn_delta_grain
        dislocation_density = 0.7 * physics_dislocation_density + 0.3 * nn_dislocation_density
        precip_fraction = 0.7 * physics_precip_fraction + 0.3 * nn_precip_fraction
        free_energy = 0.5 * physics_free_energy + 0.5 * nn_free_energy
        dissipation = 0.5 * physics_dissipation + 0.5 * nn_dissipation

        return {
            'temperature': temperature,
            'stress': stress,
            'grain_size': grain_size,
            'delta_grain': delta_grain,
            'dislocation_density': dislocation_density,
            'precip_fraction': precip_fraction,
            'free_energy': free_energy,
            'dissipation': dissipation,
            'stress_consistency_error': thermo_errors[0],
            'second_law_error': thermo_errors[1],
            'energy_balance_error': thermo_errors[2]
        }

# Aliases
MicroMechanismInformedNN = ImprovedPhysicsInformedNN
MMIANNWithConstitutiveEquations = ImprovedPhysicsInformedNN
