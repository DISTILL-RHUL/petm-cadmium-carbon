import json,numpy,pandas
from CdBURY import CdBURY # type: ignore

from matplotlib import pyplot

model_json = json.load(open("./data/output/invert_organic_posterior_results_timewarp.json"))

time = model_json[0]["time"]
time_step_length = numpy.abs(numpy.diff(numpy.array(time)*model_json[0]["time_multiplier"]))


carbon_over_cadmium = (100/1e3)/5287500  # GtC/mol organic

flux_multipliers = [numpy.cumprod(1.0+numpy.array(model["flux_multiplier"])) for model in model_json]
inverse_flux_multipliers = [1.0/flux_multiplier for flux_multiplier in flux_multipliers]

weights = time_step_length/sum(time_step_length)

## Relative
relative_organic_burials = [numpy.array(model["organic_mass"])/model["organic_mass"][0] for model in model_json]
average_relative_organic_burial = [numpy.sum(relative_organic_burial*weights) for relative_organic_burial in relative_organic_burials]

# Constant stoichiometry
# Total
organic_burial_constant = [numpy.array(model["organic_mass"])
                                                            *time_step_length
                                                            *carbon_over_cadmium for model in model_json]
total_organic_burial_constant = [numpy.sum(organic_burial) for organic_burial in organic_burial_constant]
# Excess
excess_organic_burial_constant = [(numpy.array(model["organic_mass"])-model["organic_mass"][0])
                                                *time_step_length
                                                *carbon_over_cadmium for model in model_json]
total_excess_organic_burial_constant = [numpy.sum(excess_organic_burial) for excess_organic_burial in excess_organic_burial_constant]

# Variable stoichiometry
# Total
organic_burial_variable = [numpy.array(model["organic_mass"])
                                                            *time_step_length
                                                            /flux_multiplier[:-2]
                                                            *carbon_over_cadmium for model,flux_multiplier in zip(model_json,flux_multipliers)]
total_organic_burial_variable = [numpy.sum(organic_burial) for organic_burial in organic_burial_variable]

organic_burial_variable_per_year = [organic_burial/time_step_length for organic_burial in organic_burial_variable]


# Excess
excess_organic_burial_variable = [(numpy.array(model["organic_mass"])-model["organic_mass"][0])
                                                *time_step_length
                                                /flux_multiplier[:-2]
                                                *carbon_over_cadmium for model,flux_multiplier in zip(model_json,flux_multipliers)]
total_excess_organic_burial_variable = [numpy.sum(excess_organic_burial) for excess_organic_burial in excess_organic_burial_variable]

excess_organic_burial_variable_per_year = [excess_organic_burial/time_step_length for excess_organic_burial in excess_organic_burial_variable]

for model, burial, excess_burial, total_burial, total_excess_burial, total_excess_burial_constant in zip(model_json,
                                                                                  organic_burial_variable,
                                                                                  excess_organic_burial_variable,
                                                                                  total_organic_burial_variable,
                                                                                  total_excess_organic_burial_variable,
                                                                                    total_excess_organic_burial_constant,
                                                                                  strict=True):
    model["organic_burial"] = list(burial)
    model["excess_organic_burial"] = list(excess_burial)
    model["total_organic_burial"] = float(total_burial)
    model["total_excess_organic_burial"] = float(total_excess_burial)
    model["total_excess_organic_burial_constant"] = float(total_excess_burial_constant)

# Save the model with the new variables
with open("./data/output/invert_organic_posterior_results_with_burial_timewarp.json","w") as file:
    json.dump(model_json,file)

# Comparison
constant_variable_offset = numpy.array(total_excess_organic_burial_constant)-numpy.array(total_excess_organic_burial_variable)


