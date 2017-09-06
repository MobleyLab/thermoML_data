#!/usr/bin/env python
import os
import thermopyl as th
from thermopyl import thermoml_lib
from thermopyl.utils import pandas_dataframe
import cirpy
import numpy as np
import pandas as pd
from sklearn.externals.joblib import Memory

mem = Memory(cachedir="/Users/guilhermematos/.thermoml/")

@mem.cache
def resolve_cached(x, rtype):
    return cirpy.resolve(x, rtype)

df = pandas_dataframe()

bad_filenames = ["/Users/guilhermematos/.thermoml/j.fluid.2013.12.014.xml"]
df = df[~df.filename.isin(bad_filenames)]

experiments = ["Activity coefficient","(Relative) activity"]

ind_list = [df[exp].dropna().index for exp in experiments]
ind = reduce(lambda x,y: x.union(y), ind_list)
df = df.loc[ind]

# Extract rows with two components
df["n_components"] = df.components.apply(lambda x: len(x.split("__")))
df = df[df.n_components == 2]
df.dropna(axis=1, how='all', inplace=True)

# Separate components nominally
df["component_0"] = df.components.apply(lambda x: x.split("__")[0])
df["component_1"] = df.components.apply(lambda x: x.split("__")[1])

# Find names
name_to_formula = pd.read_hdf("/Users/guilhermematos/.thermoml/compound_name_to_formula.h5", 'data')
name_to_formula = name_to_formula.dropna()

# Add formulas to the table
df["formula_0"] = df.component_0.apply(lambda chemical: name_to_formula[chemical])
df["formula_1"] = df.component_1.apply(lambda chemical: name_to_formula[chemical])

heavy_atoms = ["C","O","N","P","S","Cl","F"]
desired_atoms = ["H"] + heavy_atoms

# Add extra information
df["n_atoms_0"] = df.formula_0.apply(lambda formula_string : thermoml_lib.count_atoms(formula_string))
df["n_heavy_atoms_0"] = df.formula_0.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, heavy_atoms))
df["n_desired_atoms_0"] = df.formula_0.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, desired_atoms))
df["n_other_atoms_0"] = df.n_atoms_0 - df.n_desired_atoms_0

df["n_atoms_1"] = df.formula_1.apply(lambda formula_string : thermoml_lib.count_atoms(formula_string))
df["n_heavy_atoms_1"] = df.formula_1.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, heavy_atoms))
df["n_desired_atoms_1"] = df.formula_1.apply(lambda formula_string : thermoml_lib.count_atoms_in_set(formula_string, desired_atoms))
df["n_other_atoms_1"] = df.n_atoms_1 - df.n_desired_atoms_1

# Remove systems that have atoms outside `desired_atoms`
df = df[df.n_other_atoms_0 == 0]
df = df[df.n_other_atoms_1 == 0]
df.dropna(axis=1, how='all', inplace=True)

# Add SMILES string for each component
df["SMILES_0"] = df.component_0.apply(lambda x: resolve_cached(x, "smiles"))
df = df[df.SMILES_0 != None]
df.dropna(subset=["SMILES_0"], inplace=True)
df.loc[df.SMILES_0.dropna().index]

df["SMILES_1"] = df.component_1.apply(lambda x: resolve_cached(x, "smiles"))
df = df[df.SMILES_1 != None]
df.dropna(subset=["SMILES_1"], inplace=True)
df.loc[df.SMILES_1.dropna().index]

# Add cas and InChI for each component
df["cas_0"] = df.component_0.apply(lambda x: thermoml_lib.get_first_entry(resolve_cached(x, "cas")))
df["InChI_0"] = df.component_0.apply(lambda x: thermoml_lib.get_first_entry(resolve_cached(x, "stdinchikey")))
df = df[df.cas_0 != None]
df = df.loc[df.cas_0.dropna().index]

df["cas_1"] = df.component_1.apply(lambda x: thermoml_lib.get_first_entry(resolve_cached(x, "cas")))
df["InChI_1"] = df.component_1.apply(lambda x: thermoml_lib.get_first_entry(resolve_cached(x, "stdinchikey")))
df = df[df.cas_1 != None]
df = df.loc[df.cas_1.dropna().index]

#Extract rows with temperatures between 250 and 400 K
df = df[df['Temperature, K'] > 250.]
df = df[df['Temperature, K'] < 400.]

# Strip rows not in liquid phase
df = df[df['phase']=='Liquid']

# Rename
df["filename"] = df["filename"].map(lambda x: x.lstrip('/Users/guilhermematos/.thermoml/').rstrip('.xml'))

# More cleanup
df = df[df.n_heavy_atoms_0 > 0]
df = df[df.n_heavy_atoms_0 <= 30]
df = df[df.n_heavy_atoms_1 > 0]
df = df[df.n_heavy_atoms_1 <= 30]
df.dropna(axis=1, how='all', inplace=True)

# Organize data for keeping.
keys = ["filename","component_0","component_1","SMILES_0","SMILES_1","cas_0",
        "cas_1","InChI_0","InChI_1","Temperature, K","Pressure, kPa",
        "Activity coefficient","Activity coefficient_std",
        "(Relative) activity","(Relative) activity_std"]

dfnew = pd.concat([df['filename'],df['component_0'],df['component_1'],df['SMILES_0'],df['SMILES_1'],df["cas_0"],
                   df["cas_1"],df["InChI_0"],df["InChI_1"],df["Temperature, K"],df["Pressure, kPa"],
                   df["Activity coefficient"],df["Activity coefficient_std"],
                   df["(Relative) activity"],df["(Relative) activity_std"]], axis=1,
                  keys = keys)

a = dfnew["filename"].value_counts()
a = a.reset_index()
a.rename(columns={"index":"Filename", "filename":"Count"},inplace=True)

b0 = dfnew["InChI_0"].value_counts()
b0 = b0.reset_index()
b0.rename(columns={"index":"InChI","InChI":"Count"},inplace=True)
b0["Component"] = b0.InChI.apply(lambda x: resolve_cached(x, "iupac_name"))
b0["SMILES"] = b0.InChI.apply(lambda x: resolve_cached(x, "smiles"))

b1 = dfnew["InChI_1"].value_counts()
b1 = b1.reset_index()
b1.rename(columns={"index":"InChI","InChI":"Count"},inplace=True)
b1["Component"] = b1.InChI.apply(lambda x: resolve_cached(x, "iupac_name"))
b1["SMILES"] = b1.InChI.apply(lambda x: resolve_cached(x, "smiles"))

# Save data to cvs and pickle files
csvfile1 = "parseddata.csv"
picklefile1 = "parseddata.pickle"
csvfile2 = "alldata.csv"
picklefile2 = "alldata.pickle"

datapath = os.getcwd()

dfnew.to_csv(os.path.join(datapath,csvfile1),sep=';')
dfnew.to_pickle(os.path.join(datapath,picklefile1))
df.to_csv(os.path.join(datapath,csvfile2),sep=';')
df.to_pickle(os.path.join(datapath,picklefile2))




