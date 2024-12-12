import json,numpy,pandas,scipy
from CdBURY import CdBURY # type: ignore

from matplotlib import pyplot

model_json = json.load(open("./data/output/invert_organic_prior_results.json"))

time = model_json[0]["time"]
time_step_length = numpy.abs(numpy.diff(numpy.array(time)*model_json[0]["time_multiplier"]))

subset_start = 15
subset_end = -1

subset_time = time[subset_start:subset_end]

carbon_over_cadmium = (100/1e3)/5287500  # GtC/mol organic

flux_multipliers = [numpy.cumprod(1.0+numpy.array(model["flux_multiplier"])) for model in model_json]
inverse_flux_multipliers = [1.0/flux_multiplier for flux_multiplier in flux_multipliers]

weights = time_step_length/sum(time_step_length)

## Relative
relative_organic_burials = [numpy.array(model["organic_mass"][subset_start:subset_end])/model["organic_mass"][subset_start] for model in model_json]
average_relative_organic_burial = [numpy.sum(relative_organic_burial*weights[subset_start:subset_end]) for relative_organic_burial in relative_organic_burials]

# Constant stoichiometry
# Total
organic_burial_constant = [numpy.array(model["organic_mass"][subset_start:subset_end])
                                                            *time_step_length[subset_start:subset_end]
                                                            *carbon_over_cadmium for model in model_json]
total_organic_burial_constant = [numpy.sum(organic_burial) for organic_burial in organic_burial_constant]
# Excess
excess_organic_burial_constant = [(numpy.array(model["organic_mass"][subset_start:subset_end])-model["organic_mass"][subset_start])
                                                *time_step_length[subset_start:subset_end]
                                                *carbon_over_cadmium for model in model_json]
total_excess_organic_burial_constant = [numpy.sum(excess_organic_burial) for excess_organic_burial in excess_organic_burial_constant]

# Variable stoichiometry
# Total
organic_burial_variables = [numpy.array(model["organic_mass"][subset_start:subset_end])
                                                            *time_step_length[subset_start:subset_end]
                                                            /flux_multiplier[subset_start:subset_end-2]
                                                            *carbon_over_cadmium for model,flux_multiplier in zip(model_json,flux_multipliers)]
total_organic_burial_variable = [numpy.sum(organic_burial) for organic_burial in organic_burial_variables]

organic_burial_variable_per_year = [organic_burial/time_step_length[subset_start:subset_end] for organic_burial in organic_burial_variables]

# Excess
excess_organic_burial_variables = [(numpy.array(model["organic_mass"][subset_start:subset_end])-model["organic_mass"][subset_start])
                                                *time_step_length[subset_start:subset_end]
                                                /flux_multiplier[subset_start:subset_end-2]
                                                *carbon_over_cadmium for model,flux_multiplier in zip(model_json,flux_multipliers)]
total_excess_organic_burial_variable = [numpy.sum(excess_organic_burial) for excess_organic_burial in excess_organic_burial_variables]

excess_organic_burial_variable_per_year = [excess_organic_burial/time_step_length[subset_start:subset_end] for excess_organic_burial in excess_organic_burial_variables]


# Comparison
constant_variable_offset = numpy.array(total_excess_organic_burial_constant)-numpy.array(total_excess_organic_burial_variable)

kde_x = numpy.arange(-1e3,1e5,1e2)
excess_constant_kde = scipy.stats.gaussian_kde(total_excess_organic_burial_constant)
excess_variable_kde = scipy.stats.gaussian_kde(total_excess_organic_burial_variable)

figure,axes = pyplot.subplots(1,1)
handle_0 = axes.hist(total_excess_organic_burial_constant,color="#6cb7c4",zorder=1,bins=50,alpha=0.5)
handle_1 = axes.hist(total_excess_organic_burial_variable,color="#4276b1",zorder=1,bins=50,alpha=0.5)


# handle_0 = axes.fill_between(kde_x,excess_constant_kde(kde_x),color="#6cb7c4",alpha=0.3,zorder=0)
# handle_1 = axes.fill_between(kde_x,excess_variable_kde(kde_x),color="#4276b1",alpha=0.3,zorder=0)

axes.legend(["Constant stoichiometry","Variable stoichiometry"],loc="upper right")

axes.set_xlabel("Total excess organic burial (GtC)")
axes.set_ylabel("Density")

figure_2,axes_2 = pyplot.subplots(1,1)

excess_burial_per_year_upper = numpy.percentile(organic_burial_variable_per_year,97.5,axis=0)
excess_burial_per_year_lower = numpy.percentile(organic_burial_variable_per_year,2.5,axis=0)

axes_2.fill_between(subset_time[1:],excess_burial_per_year_upper,excess_burial_per_year_lower,color="#4276b1",alpha=0.3)

axes_2.plot(subset_time[1:],numpy.median(organic_burial_variable_per_year,axis=0))

axes_2.set_xlabel("Age (Ma)")
axes_2.set_ylabel("Organic burial rate (GtC/yr)")

axes_2.set_xlim((55.95,55.65))


a = 5