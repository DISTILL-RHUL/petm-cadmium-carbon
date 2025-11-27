import json
from matplotlib import pyplot

from CdBURY.classes import Time,CadmiumMassIsotope,CadmiumFluxes,Perturbation,FluxType
from CdBURY import CdBURY

## Run CdBURY in a forward direction for a steady state and two perturbations
# Create the intial conditions
initial_seawater = CadmiumMassIsotope(1.0*8.36e11,-0.1)

isotope_offsets = CadmiumFluxes(-0.1,
                                0.0,
                                -0.55,
                                0.0,
                                -0.80,
                                0.0)

flux_proportions = CadmiumFluxes(1.0,
                            0.01,
                            0.03,
                            0.01,
                            0.11125,
                            0.84875)
flux_proportions.sulphide = flux_proportions.input-(flux_proportions.oxyhydroxide+flux_proportions.carbonate+flux_proportions.clay+flux_proportions.organic)
input_flux = 30000000.0
initial_fluxes = CadmiumFluxes(input_flux*flux_proportions.input,
                                input_flux*flux_proportions.oxyhydroxide,
                                input_flux*flux_proportions.carbonate,
                                input_flux*flux_proportions.clay,
                                input_flux*flux_proportions.organic,
                                input_flux*flux_proportions.sulphide)

input_balanced = CdBURY.Model.calculate_balanced_input(initial_seawater,
                                                                initial_fluxes,
                                                                isotope_offsets)

adjusted_isotope_offsets = CadmiumFluxes(input_balanced.isotope_delta,
                                            isotope_offsets.oxyhydroxide,
                                            isotope_offsets.carbonate,
                                            isotope_offsets.clay,
                                            isotope_offsets.organic,
                                            isotope_offsets.sulphide)

# Set the model timing
time_configuration = Time(55.92,0.001,55.00,units="Myr")

# Create a steady state model
model_steady = CdBURY.Model(time_configuration,
                    initial_seawater,
                    initial_fluxes,
                    adjusted_isotope_offsets,
                    dynamic_mass=False)

# Simulate
model_steady.solve()

# Save
with open("./data/output/steady_state.json","w") as steady_file:
    json.dump(model_steady.to_json(),steady_file,indent=4)

# Create a perturbation model
model_perturb_down = CdBURY.Model(time_configuration,
                    initial_seawater,
                    initial_fluxes,
                    adjusted_isotope_offsets,
                    dynamic_mass=True)

# Perturb a temporary decrease in organic flux
model_perturb_down.change_flux(Perturbation(55.50,FluxType.organic,CadmiumMassIsotope(0.0,-0.8)))
model_perturb_down.change_flux(Perturbation(55.30,FluxType.organic,CadmiumMassIsotope(initial_fluxes.organic,-0.8)))

# Simulate
model_perturb_down.solve()

# Save
with open("./data/output/perturbation_down.json","w") as steady_file:
    json.dump(model_perturb_down.to_json(),steady_file,indent=4)

# Create a perturbation model
model_perturb_up = CdBURY.Model(time_configuration,
                    initial_seawater,
                    initial_fluxes,
                    adjusted_isotope_offsets,
                    dynamic_mass=True)

# Perturb a temporary increase in organic flux
model_perturb_up.change_flux(Perturbation(55.50,FluxType.organic,CadmiumMassIsotope(initial_fluxes.organic*2,-0.8)))
model_perturb_up.change_flux(Perturbation(55.30,FluxType.organic,CadmiumMassIsotope(initial_fluxes.organic,-0.8)))

# Simulate
model_perturb_up.solve()

# Save
with open("./data/output/perturbation_up.json","w") as steady_file:
    json.dump(model_perturb_up.to_json(),steady_file,indent=4)

