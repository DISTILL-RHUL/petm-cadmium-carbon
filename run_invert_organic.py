import numpy,pandas,json

from CdBURY.classes import Time,MultiTime,DynamicTime,CadmiumMassIsotope,CadmiumFluxes,Perturbation,FluxType # type: ignore
from CdBURY import CdBURY # type: ignore

from matplotlib import pyplot

from geochemistry_helpers import Sampling,GaussianProcess # type: ignore

numpy.random.seed(2)

## Data
svalbard_data = pandas.read_excel("./data/input/cadmium_data.xlsx",usecols="A:I").dropna(subset=["δ114Cd(‰)"])
arctic_data = pandas.read_excel("./data/input/cadmium_data.xlsx",usecols="K:S").dropna(subset=["δ114Cd(‰).1"])
northsea_data = pandas.read_excel("./data/input/cadmium_data.xlsx",usecols="U:AC").dropna(subset=["δ114Cd(‰).2"])
kheu_data = pandas.read_excel("./data/input/cadmium_data.xlsx",usecols="AI:AQ").dropna(subset=["δ114Cd(‰).3"])

svalbard_data["2se"] = svalbard_data["2se"]
arctic_data["2se.1"] = arctic_data["2se.1"]
northsea_data["2se.2"] = northsea_data["2se.2"]
kheu_data["2se.4"] = kheu_data["2se.4"]


ages = numpy.concatenate((svalbard_data["Age model"],arctic_data["Age model.1"],northsea_data["Age model.2"],kheu_data["Age model.3"]))
cd_x = numpy.arange(-2.0,2.0,0.001)
number_of_samples = 20

d114Cds = numpy.concatenate((svalbard_data["δ114Cd(‰)"],arctic_data["δ114Cd(‰).1"],northsea_data["δ114Cd(‰).2"],kheu_data["δ114Cd(‰).3"]))
d114Cd_uncertainties = numpy.concatenate((svalbard_data["2se"]/2,arctic_data["2se.1"]/2,northsea_data["2se.2"]/2,kheu_data["2se.4"]/2))

good_indices = numpy.where(d114Cds>=-0.6)

cd_distributions = [Sampling.Distribution(cd_x,"Gaussian",(cd,uncertainty*1),location=age).normalise() for cd,uncertainty,age in zip(d114Cds[good_indices],d114Cd_uncertainties[good_indices],ages[good_indices],strict=True)]

# Time
sigmoid = lambda a,b,c,x0,x: -a/(1+numpy.exp(-b*(x-x0)))+(c+a)
# sigmoid_test = sigmoid(0.0010,100.0,0.000005,55.83,numpy.linspace(55.91,55.68,1000))

steps = [55.91]
while steps[-1]>55.68:
    step_length = sigmoid(0.0010,100.0,0.000005,55.83,steps[-1])
    steps.append(steps[-1]-step_length)

# time_configuration = DynamicTime(55.93,0.0001,55.68,units="Myr",min_step=0.0001/10,max_step=0.0001*2)
# time_configuration = MultiTime(numpy.concatenate((high_resolution_steps,low_resolution_steps)),units="Myr")
time_configuration = MultiTime(steps,units="Myr")

# time_configuration = Time(55.93,0.001,55.68,units="Myr")
model_ages = [ages,time_configuration.as_array()]
data_interpolation_ages = [ages,time_configuration.as_padded_array()]

# Gaussian Process
gp = (GaussianProcess().constrain(cd_distributions)
                        .setKernel("rbf",(5.0,0.050))
                        .query(model_ages)
                        .getSamples(number_of_samples)
)

# Save the gp
gp.toJSON("./data/output/cadmium_gp.json")

indices = range(0,number_of_samples)

## Model
random_mass_multiplier = numpy.random.uniform(0.2,5.0,len(indices))
random_input_isotopes = numpy.random.uniform(0.0,0.4,len(indices))
random_input_multiplier = numpy.random.uniform(0.5,2.0,len(indices))

# seawaters = []
# organics = []

def determine_initial_organic_burial(first_three_d114Cds,data_resolution,initial_seawater,initial_fluxes,isotope_offsets):
    isotope_fluxes_0 = CadmiumFluxes(CadmiumMassIsotope(initial_fluxes.input,initial_seawater.isotope_delta+isotope_offsets.input),
                                    CadmiumMassIsotope(initial_fluxes.oxyhydroxide,initial_seawater.isotope_delta+isotope_offsets.oxyhydroxide),
                                    CadmiumMassIsotope(initial_fluxes.carbonate,initial_seawater.isotope_delta+isotope_offsets.carbonate),
                                    CadmiumMassIsotope(initial_fluxes.clay,initial_seawater.isotope_delta+isotope_offsets.clay),
                                    CadmiumMassIsotope(initial_fluxes.organic,initial_seawater.isotope_delta+isotope_offsets.organic),
                                    CadmiumMassIsotope(initial_fluxes.sulphide,initial_seawater.isotope_delta+isotope_offsets.sulphide))
    
    extra_organic_flux_0 = CdBURY.Model.calculate_organic_to_target(first_three_d114Cds[1],
                                                initial_seawater,
                                                isotope_fluxes_0)
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
                                                isotope_fluxes_1)
    organic_flux_per_year_1 = organic_flux_per_year_0+extra_organic_flux_1/data_resolution
    if organic_flux_per_year_1<0:
        organic_flux_per_year_1 = 0

    return (organic_flux_per_year_0 + organic_flux_per_year_1)/2


models = []

for index in indices:
    if index%100==0:
        print(index)
        
    d114Cd_evolution = [gp.samples[0][index],gp.samples[1][index]]
    
    initial_seawater = CadmiumMassIsotope(random_mass_multiplier[index]*8.36e11,d114Cd_evolution[-1][0])

    isotope_offsets = CadmiumFluxes(initial_seawater.isotope_delta-random_input_isotopes[index],
                                    0.0,
                                    -0.55,
                                    0.0,
                                    -0.80,
                                    0.0)

    flux_proportions = CadmiumFluxes(1.0,
                                0.01,
                                0.02,
                                0.01,
                                0.104375,
                                0.84875)
    flux_proportions.sulphide = flux_proportions.input-(flux_proportions.oxyhydroxide+flux_proportions.carbonate+flux_proportions.clay+flux_proportions.organic)

    input_flux = random_input_multiplier[index]*30000000.0
    initial_fluxes = CadmiumFluxes(input_flux*flux_proportions.input,
                                    input_flux*flux_proportions.oxyhydroxide,
                                    input_flux*flux_proportions.carbonate,
                                    input_flux*flux_proportions.clay,
                                    input_flux*flux_proportions.organic,
                                    input_flux*flux_proportions.sulphide)
    try:
        (new_organic_flux,new_sulphide_flux) = CdBURY.Model.rebalance_with_organic(initial_seawater,
                                                                                initial_fluxes,
                                                                                isotope_offsets)
    except:
        continue

    adjusted_initial_fluxes = CadmiumFluxes(initial_fluxes.input,
                                            initial_fluxes.oxyhydroxide,
                                            initial_fluxes.carbonate,
                                            initial_fluxes.clay,
                                            new_organic_flux,
                                            new_sulphide_flux)
    
    model = CdBURY.Model(time_configuration,
                        initial_seawater,
                        adjusted_initial_fluxes,
                        isotope_offsets)
    
    match time_configuration:
        case Time():
            initial_organic_flux = determine_initial_organic_burial(d114Cd_evolution[-1],
                                                                model.time.step_in_yr,
                                                                initial_seawater,
                                                                adjusted_initial_fluxes,
                                                                isotope_offsets)
        case DynamicTime():
            initial_organic_flux = determine_initial_organic_burial(d114Cd_evolution[-1],
                                                                model.time.initial_step*model.time.units,
                                                                initial_seawater,
                                                                adjusted_initial_fluxes,
                                                                isotope_offsets)
        case MultiTime():
            initial_organic_flux = determine_initial_organic_burial(d114Cd_evolution[-1],
                                                                model.time.initial_step_in_yr,
                                                                initial_seawater,
                                                                adjusted_initial_fluxes,
                                                                isotope_offsets)
    
    
    final_initial_fluxes = CadmiumFluxes(adjusted_initial_fluxes.input,
                                        adjusted_initial_fluxes.oxyhydroxide,
                                        adjusted_initial_fluxes.carbonate,
                                        adjusted_initial_fluxes.clay,
                                        initial_organic_flux,
                                        adjusted_initial_fluxes.sulphide)
    
    if final_initial_fluxes.organic<=0:
        continue
    
    model = CdBURY.Model(time_configuration,
                        initial_seawater,
                        final_initial_fluxes,
                        isotope_offsets)
    
    model.invert(model_ages[-1],d114Cd_evolution[-1],FluxType.organic)

    # figure,axes = pyplot.subplots(3,1,sharex=True)
    # axes[0].plot(model_ages[-1],d114Cd_evolution[-1],zorder=3,color="k")
    # axes[0].plot(model.time.as_array()[:-1],[flux.isotope_delta for flux in model.seawater[:-1]],zorder=3,color="r")
    # axes[0].scatter(model.time.as_array()[:-1],[flux.isotope_delta for flux in model.seawater[:-1]],zorder=3,color="purple")

    # axes[1].plot(model.time.as_array()[:-1],[flux.mass for flux in model.seawater[:-1]],zorder=3,color="r")

    # # axes[2].scatter(model.time.as_array(),[flux.mass for flux in model.organic_flux][0:-1],zorder=3,color="b")
    # axes[2].plot(model.time.as_array()[:-1],[flux.mass for flux in model.organic_flux[:-1]],zorder=3,color="r")

    # gp.plotConstraints(color="black",axis=axes[0])

    # axes[0].invert_xaxis()

    models.append(model)


model_json = []
for model in models:
    model_json.append(model.to_json())

# Write to file
with open("./data/output/invert_organic_prior_results.json","w") as file:
    json.dump(model_json,file,indent=4)


a = 5
