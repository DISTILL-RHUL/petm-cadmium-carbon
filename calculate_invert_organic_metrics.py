import json,numpy,pandas
from CdBURY import CdBURY

model_json = json.load(open("./data/output/invert_organic_prior_results.json"))
time = model_json[0]["time"]
time_step_length = numpy.abs(numpy.diff(numpy.array(time)*model_json[0]["time_multiplier"]))
timestep = time[1]-time[0]

organic_to_c = (100/1e3)/5287500  # GtC/mol organic

excess_organic_burial = [((numpy.array(model["organic_mass"])/model["organic_mass"][0])-1.0) for model in model_json]
excess_means = [numpy.mean(excess) for excess in excess_organic_burial]

flux_multipliers = [numpy.cumprod(1.0+numpy.array(model["flux_multiplier"])) for model in model_json]
organic_burial_variable_stoichiometry = [(model["organic_mass"]*time_step_length)/flux_multiplier[:-1] for model,flux_multiplier in zip(model_json,flux_multipliers,strict=True)]
total_organic_burial_variable_stoichiometry = [numpy.sum(organic_burial) for organic_burial in organic_burial_variable_stoichiometry]
total_c_burial = [organic_burial*organic_to_c for organic_burial in total_organic_burial_variable_stoichiometry]

excess_burial_variable_stoichiometry = [organic_burial-organic_burial[0] for organic_burial in organic_burial_variable_stoichiometry]
total_excess_burial = [numpy.sum(excess_burial) for excess_burial in excess_burial_variable_stoichiometry]
total_excess_c_burial = [excess_burial*organic_to_c for excess_burial in total_excess_burial]

organic_burial_constant_stoichiometry = [(model["organic_mass"]*time_step_length) for model in model_json]
total_organic_burial_constant_stoichiometry = [numpy.sum(organic_burial) for organic_burial in organic_burial_constant_stoichiometry]
total_c_burial_constant_stoichiometry = [organic_burial*organic_to_c for organic_burial in total_organic_burial_constant_stoichiometry]

excess_burial_constant_stoichiometry = [organic_burial-organic_burial[0] for organic_burial in organic_burial_constant_stoichiometry]
total_excess_burial_constant_stoichiometry = [numpy.sum(excess_burial) for excess_burial in excess_burial_constant_stoichiometry]
total_excess_c_burial_constant_stoichiometry = [excess_burial*organic_to_c for excess_burial in total_excess_burial_constant_stoichiometry]

organic_burial_constant_stoichiometry = [CdBURY.Model.integrate_organic(model["organic_mass"],model["time_multiplier"]) for model in model_json]
excess_organic_burial = [CdBURY.Model.integrate_excess_organic(model["organic_mass"],model["time_multiplier"]) for model in model_json]
    
posterior_model_json = []
for model in model_json:
    if all(numpy.array(model["organic_mass"])>0) and all(numpy.array(model["seawater_mass"])>0):
        posterior_model_json.append(model)

with open("./data/output/invert_organic_posterior_results.json","w") as file:
    json.dump(posterior_model_json,file)


useful_prior_metrics = pandas.DataFrame()
useful_prior_metrics["time"] = list(time[:-1])

seawater_masses = [model["seawater_mass"] for model in model_json]
useful_prior_metrics["seawater_mass_median"] = numpy.median(seawater_masses,axis=0)
useful_prior_metrics["seawater_mass_lower_quantile"] = numpy.percentile(seawater_masses,2.5,axis=0)
useful_prior_metrics["seawater_mass_upper_quantile"] = numpy.percentile(seawater_masses,97.5,axis=0)

seawater_d114Cds = [model["seawater_δ114Cd"] for model in model_json]
useful_prior_metrics["seawater_d114Cd_median"] = numpy.median(seawater_d114Cds,axis=0)
useful_prior_metrics["seawater_d114Cd_lower_quantile"] = numpy.percentile(seawater_d114Cds,2.5,axis=0)
useful_prior_metrics["seawater_d114Cd_upper_quantile"] = numpy.percentile(seawater_d114Cds,97.5,axis=0)

organic_flux = [model["organic_mass"] for model in model_json]
useful_prior_metrics["organic_flux_median"] = numpy.median(organic_flux,axis=0)
useful_prior_metrics["organic_flux_lower_quantile"] = numpy.percentile(organic_flux,2.5,axis=0)
useful_prior_metrics["organic_flux_upper_quantile"] = numpy.percentile(organic_flux,97.5,axis=0)

useful_prior_metrics.to_excel("./data/output/invert_organic_prior_metrics.xlsx")

useful_posterior_metrics = pandas.DataFrame()
useful_posterior_metrics["time"] = list(time[:-1])

seawater_masses = [model["seawater_mass"] for model in posterior_model_json]
useful_posterior_metrics["seawater_mass_median"] = numpy.median(seawater_masses,axis=0)
useful_posterior_metrics["seawater_mass_lower_quantile"] = numpy.percentile(seawater_masses,2.5,axis=0)
useful_posterior_metrics["seawater_mass_upper_quantile"] = numpy.percentile(seawater_masses,97.5,axis=0)

seawater_d114Cds = [model["seawater_δ114Cd"] for model in posterior_model_json]
useful_posterior_metrics["seawater_d114Cd_median"] = numpy.median(seawater_d114Cds,axis=0)
useful_posterior_metrics["seawater_d114Cd_lower_quantile"] = numpy.percentile(seawater_d114Cds,2.5,axis=0)
useful_posterior_metrics["seawater_d114Cd_upper_quantile"] = numpy.percentile(seawater_d114Cds,97.5,axis=0)

organic_flux = [model["organic_mass"] for model in posterior_model_json]
useful_posterior_metrics["organic_flux_median"] = numpy.median(organic_flux,axis=0)
useful_posterior_metrics["organic_flux_lower_quantile"] = numpy.percentile(organic_flux,2.5,axis=0)
useful_posterior_metrics["organic_flux_upper_quantile"] = numpy.percentile(organic_flux,97.5,axis=0)

relative_organic_flux = [list(numpy.array(model["organic_mass"])/model["organic_mass"][0]) for model in posterior_model_json]
useful_posterior_metrics["relative_organic_flux_median"] = numpy.median(relative_organic_flux,axis=0)
useful_posterior_metrics["relative_organic_flux_lower_quantile"] = numpy.percentile(relative_organic_flux,2.5,axis=0)
useful_posterior_metrics["relative_organic_flux_upper_quantile"] = numpy.percentile(relative_organic_flux,97.5,axis=0)


useful_posterior_metrics.to_excel("./data/output/invert_organic_posterior_metrics.xlsx")


