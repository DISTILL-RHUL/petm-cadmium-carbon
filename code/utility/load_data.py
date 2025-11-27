import pandas

def load_cadmium(data_directory="./data/"):
    d114cd_data  = pandas.read_excel(data_directory+"/input/cadmium_data.xlsx",usecols="A:H").dropna(subset=["δ114Cddetrital-corrected"])
    d114cd_data["δ114Cd(‰)"] = d114cd_data["δ114Cddetrital-corrected"]

    return d114cd_data

def split_data_by_site(d114cd_data):
    svalbard_data = d114cd_data[d114cd_data["Core/section"]=="BH9-05, Svalbard"]
    arctic_data = d114cd_data[d114cd_data["Core/section"]=="IODP-ACEX 302-M0004a, Arctic"]
    northsea_data = d114cd_data[d114cd_data["Core/section"]=="E-8X, North Sea"]
    kheu_data = d114cd_data[d114cd_data["Core/section"]=="Kheu River, Georgia"]

    return svalbard_data, arctic_data, northsea_data, kheu_data

def load_carbon(data_directory="./data/"):
    d13c_data = pandas.read_excel(data_directory+"/input/cadmium_data_carbon.xlsx",usecols="A:D").dropna(subset=["δ13Corg(‰)"])

    return d13c_data
