import numpy,pandas,json,sys
from matplotlib import pyplot

from CdBURY.classes import Time,CadmiumMassIsotope,CadmiumFluxes,Perturbation,FluxType
from CdBURY import CdBURY

from geochemistry_helpers import Sampling,GaussianProcess

sys.path.append("./code/utility")
from load_data import load_cadmium,split_data_by_site,load_carbon


def determine_initial_organic_burial(first_three_d114Cds,data_resolution,initial_seawater,initial_fluxes,isotope_offsets):
    isotope_fluxes_0 = CadmiumFluxes(CadmiumMassIsotope(initial_fluxes.input,initial_seawater.isotope_delta+isotope_offsets.input),
                                    CadmiumMassIsotope(initial_fluxes.oxyhydroxide,initial_seawater.isotope_delta+isotope_offsets.oxyhydroxide),
                                    CadmiumMassIsotope(initial_fluxes.carbonate,initial_seawater.isotope_delta+isotope_offsets.carbonate),
                                    CadmiumMassIsotope(initial_fluxes.clay,initial_seawater.isotope_delta+isotope_offsets.clay),
                                    CadmiumMassIsotope(initial_fluxes.organic,initial_seawater.isotope_delta+isotope_offsets.organic),
                                    CadmiumMassIsotope(initial_fluxes.sulphide,initial_seawater.isotope_delta+isotope_offsets.sulphide))
    
    extra_organic_flux_0 = CdBURY.Model.calculate_organic_to_target(first_three_d114Cds[1],
                                                initial_seawater,
                                                isotope_fluxes_0,
                                                data_resolution)
    organic_flux_per_year_0 = initial_fluxes.organic+extra_organic_flux_0/data_resolution
    if organic_flux_per_year_0<0:
        organic_flux_per_year_0 = 0

    

    isotope_fluxes_1 = CadmiumFluxes(CadmiumMassIsotope(initial_fluxes.input,initial_seawater.isotope_delta+isotope_offsets.input),
                                    CadmiumMassIsotope(initial_fluxes.oxyhydroxide,initial_seawater.isotope_delta+isotope_offsets.oxyhydroxide),
                                    CadmiumMassIsotope(initial_fluxes.carbonate,initial_seawater.isotope_delta+isotope_offsets.carbonate),
                                    CadmiumMassIsotope(initial_fluxes.clay,initial_seawater.isotope_delta+isotope_offsets.clay),
                                    CadmiumMassIsotope(organic_flux_per_year_0,initial_seawater.isotope_delta+isotope_offsets.organic),
                                    CadmiumMassIsotope(initial_fluxes.sulphide,initial_seawater.isotope_delta+isotope_offsets.sulphide))

    model = CdBURY.Model(time_configuration,
                        initial_seawater,
                        isotope_fluxes_1,
                        isotope_offsets)
    
    (next_seawater_isotopes,
     adjusted_fluxes,
     next_flux_multiplier,
     gradients) = model.step_forward(data_resolution,
                                     initial_seawater,
                                     isotope_fluxes_0,
                                     isotope_offsets,
                                     0.0,
                                     True)
    
    extra_organic_flux_1 = CdBURY.Model.calculate_organic_to_target(first_three_d114Cds[2],
                                                next_seawater_isotopes,
                                                isotope_fluxes_1,
                                                data_resolution)
    organic_flux_per_year_1 = organic_flux_per_year_0+extra_organic_flux_1/data_resolution
    if organic_flux_per_year_1<0:
        organic_flux_per_year_1 = 0

    return (organic_flux_per_year_0 + organic_flux_per_year_1)/2

numpy.random.seed(2) # for repeatability

#region: Data Processing
## Data
d114cd_data = load_cadmium()
svalbard_cd_data, arctic_cd_data, northsea_cd_data, kheu_cd_data = split_data_by_site(d114cd_data)

d13c_data = load_carbon()
svalbard_c_data, arctic_c_data, northsea_c_data, kheu_c_data = split_data_by_site(d13c_data)


# Specify the number of samples to draw
number_of_samples = 10000

# Specify the bins for cd and d13c distributions
d114cd_x = numpy.arange(-10.0,10.0,0.001)
d13c_x = numpy.arange(-40.0,-20.0,0.001)

# Collate values into arrays
ages = d114cd_data["Age model"].values
d114Cds = d114cd_data["δ114Cd(‰)"].values
d114Cd_uncertainties = d114cd_data["δ114Cd (2σ)"].values/2  # Convert from 2σ to 1σ

d13c = d13c_data["δ13Corg(‰)"].values

# Keep data only where not outliers
good_indices = numpy.where(d114Cds>=-0.6)

# Create distributions
d114cd_distributions = [Sampling.Distribution(d114cd_x,"Gaussian",(cd,uncertainty*1),location=age).normalise() for cd,uncertainty,age in zip(d114Cds[good_indices],d114Cd_uncertainties[good_indices],ages[good_indices],strict=True)]
d13c_distributions = [Sampling.Distribution(d13c_x,"Gaussian",(c,0.1),location=age).normalise() for c,age in zip(d13c,d13c_data["Age model"].values,strict=True)]
#endregion: Data Processing

#region: Gaussian Processes
# Time
time_configuration = Time(55.905,0.001,55.68,units="Myr")

model_ages = [ages,time_configuration.as_array()]
data_interpolation_ages = [ages,time_configuration.as_padded_array()]

# Specify the cadmium isotope hyperparameters - 2.0‰ and 0.05Myr (50kyr)
d114cd_gp_hyperparameters = (2.0,0.05)

# Create gps for d114Cd and d13C
d114cd_gp = (GaussianProcess().constrain(d114cd_distributions)
                        .setKernel("rbf",d114cd_gp_hyperparameters)
                        .query(data_interpolation_ages)
                        .getSamples(number_of_samples)
)
d13c_gp = (GaussianProcess().constrain(d13c_distributions)
                        .setKernel("rbf",d114cd_gp_hyperparameters) # Use the same hyperparameters as d114Cd
                        .query(data_interpolation_ages)
                        .getSamples(number_of_samples)
)

#endregion: Gaussian Processes


#region: Model Setup
# Pick randomised values for the initial conditions
# The multipliers are relative to modern, whereas the input isotope is an absolute value
random_mass_multiplier = numpy.random.uniform(0.333,3.0,number_of_samples)
random_input_multiplier = numpy.random.uniform(0.333,3.0,number_of_samples)
random_input_isotopes = numpy.random.uniform(0.0,0.4,number_of_samples)

# Create some empty lists to store the results
mass_multiplier_record = []
input_isotopes_record = []
input_multiplier_record = []

organic_to_c = (100/1e3)/5287500  # GtC/mol organic
#endregion: Model Setup

#region: Model
indices = range(0,number_of_samples)
models = []

for index in indices:
    # Print the index every 100 iterations
    if index%100==0:
        print(index)

    # Calculate the absolute inventory of Cd from the mass multiplier
    initial_cd_inventory = random_mass_multiplier[index]*8.36e11

    # Calculate the absolute initial input flux of Cd from the input multiplier
    initial_input_flux = 30000000.0 * random_input_multiplier[index]

    # Calculate the residence time of Cd in the ocean (for reference)
    residence_time = initial_cd_inventory/initial_input_flux

    # Extract a single sample of d114cd
    d114cd_evolution: list[numpy.ndarray] = [d114cd_gp.samples[0][index],d114cd_gp.samples[1][index]] #type: ignore
    
    # Combine the mass and isotope ratio of seawater into a single object
    initial_seawater = CadmiumMassIsotope(initial_cd_inventory,d114cd_evolution[-1][0])

    # Compile the isotope offsets into a single object
    isotope_offsets = CadmiumFluxes(initial_seawater.isotope_delta-random_input_isotopes[index],
                                    0.0,
                                    -0.55,
                                    0.0,
                                    -0.80,
                                    0.0)
    
    # Record the input isotopes and mass multiplier
    input_isotopes_record.append(initial_seawater.isotope_delta-random_input_isotopes[index])
    input_multiplier_record.append(random_input_multiplier[index])
    mass_multiplier_record.append(random_mass_multiplier[index])

    # Combine the proportions of each flux (for steady state) into a single object
    # The proportions are relative to the input flux
    flux_proportions = CadmiumFluxes(1.0,
                                0.01,
                                0.02,
                                0.01,
                                0.104375,
                                0.84875)
    # Ensure the proportions sum exactly to 1.0
    flux_proportions.sulphide = flux_proportions.input-(flux_proportions.oxyhydroxide+flux_proportions.carbonate+flux_proportions.clay+flux_proportions.organic)

    # Compute the absolute fluxes from the relative values
    input_flux = initial_input_flux
    initial_fluxes = CadmiumFluxes(input_flux*flux_proportions.input,
                                    input_flux*flux_proportions.oxyhydroxide,
                                    input_flux*flux_proportions.carbonate,
                                    input_flux*flux_proportions.clay,
                                    input_flux*flux_proportions.organic,
                                    input_flux*flux_proportions.sulphide)
    
    # Make the steady state work isotopically as well as for mass
    # This may not be possible for all combinations of fluxes/parameters - if not, skip this iteration
    try:
        (new_organic_flux,new_sulphide_flux) = CdBURY.Model.rebalance_with_organic(initial_seawater,
                                                                                initial_fluxes,
                                                                                isotope_offsets)
    except:
        continue

    # Compile the adjusted fluxes into a single object
    adjusted_initial_fluxes = CadmiumFluxes(initial_fluxes.input,
                                            initial_fluxes.oxyhydroxide,
                                            initial_fluxes.carbonate,
                                            initial_fluxes.clay,
                                            new_organic_flux,
                                            new_sulphide_flux)
    
    # Create a model object
    model = CdBURY.Model(time_configuration,
                        initial_seawater,
                        adjusted_initial_fluxes,
                        isotope_offsets)

    # Do a single RK4 step - changing organic to match d114cd
    averaged_fluxes = model.invert_rk4_step_organic(model.get_fluxes(0),
                            initial_seawater,
                            0.0,
                            0,
                            d114cd_evolution[-1])

    # Compile the new fluxes - with non steady state organic carbon burial to drive the initial trend in d114cd
    final_initial_fluxes = CadmiumFluxes(adjusted_initial_fluxes.input,
                                        adjusted_initial_fluxes.oxyhydroxide,
                                        adjusted_initial_fluxes.carbonate,
                                        adjusted_initial_fluxes.clay,
                                        averaged_fluxes.organic.mass,
                                        adjusted_initial_fluxes.sulphide)
    
    # Make sure the organic flux is positive
    if final_initial_fluxes.organic<=0:
        continue
    
    # Create a new model object with the new fluxes
    model = CdBURY.Model(time_configuration,
                        initial_seawater,
                        final_initial_fluxes,
                        isotope_offsets)
    
    # Run the full inversion using the Runge-Kutta 4th order method
    model.invert_rk4(d114cd_evolution[-1],
                     FluxType.organic)
    # Append the model to the storage list if it is valid
    models.append(model)

#endregion: Model

#region: Save Results
# Save the GPs
d114cd_gp.toJSON("./data/output/d114cd_gp.json")
d13c_gp.toJSON("./data/output/d13c_gp.json")

original_initial_conditions = {
    "input_isotopes": input_isotopes_record,
    "mass_multiplier": mass_multiplier_record,
    "input_multiplier": input_multiplier_record,
}

# Save the original initial conditions
with open("./data/output/original_initial_conditions.json","w") as file:
    json.dump(original_initial_conditions,file,indent=4)

# The model impements a to_json method, which returns a dictionary of the model parameters
model_json = []
for model in models:
    model_json.append(model.to_json())

# Write to file
# Save as the prior
with open("./data/output/invert_organic_prior_results.json","w") as file:
    json.dump(model_json,file,indent=4)
#endregion: Save Results

