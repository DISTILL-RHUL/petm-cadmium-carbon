import pandas,numpy,json

model_json = json.load(open("./data/output/invert_organic_posterior_results_with_burial.json"))
time = model_json[0]["time"]
time_step_length = numpy.abs(numpy.diff(numpy.array(time)*model_json[0]["time_multiplier"]))

total_organic_carbon_burial = [model["total_organic_burial"] for model in model_json]
total_excess_organic_burial = [model["total_excess_organic_burial"] for model in model_json]

organic_carbon_burial = [model["organic_burial"] for model in model_json]
organic_carbon_burial_per_year = [organic_burial/time_step_length for organic_burial in organic_carbon_burial]

excess_organic_carbon_burial = [model["excess_organic_burial"] for model in model_json]
excess_organic_burial_per_year = [organic_burial/time_step_length for organic_burial in excess_organic_carbon_burial]

burial_per_year_upper = numpy.percentile(organic_carbon_burial_per_year,97.5,axis=0)
burial_per_year_median = numpy.median(organic_carbon_burial_per_year,axis=0)
burial_per_year_lower = numpy.percentile(organic_carbon_burial_per_year,2.5,axis=0)

burial_dataframe = pandas.DataFrame(data={"time":time[1:],
                                                 "2.5":burial_per_year_lower,
                                                 "50.0":burial_per_year_median,
                                                 "97.5":burial_per_year_upper})
burial_dataframe.sort_values(by="time",inplace=True,ascending=True)
burial_dataframe.to_excel("./data/output/excess_organic_burial_metrics.xlsx",index_label='index')

sorted_burial_indices = numpy.argsort(total_organic_carbon_burial)
lower_burial_index = sorted_burial_indices[int(len(sorted_burial_indices)*0.025)]
middler_burial_index = sorted_burial_indices[int(len(sorted_burial_indices)*0.5)]
upper_burial_index = sorted_burial_indices[int(len(sorted_burial_indices)*0.975)]

lowest_burial_per_year = organic_carbon_burial_per_year[lower_burial_index]
middlest_burial_per_year = organic_carbon_burial_per_year[middler_burial_index]
highest_burial_per_year = organic_carbon_burial_per_year[upper_burial_index]

burial_sample_dataframe = pandas.DataFrame(data={"time":time[1:],
                                                        "lowest":lowest_burial_per_year,
                                                        "middlest":middlest_burial_per_year,
                                                        "highest":highest_burial_per_year})
burial_sample_dataframe.sort_values(by="time",inplace=True,ascending=True)
burial_sample_dataframe.to_excel("./data/output/excess_organic_burial_sample.xlsx",index_label="index")