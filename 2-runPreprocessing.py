import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import preprocessing.preprocessing as prep
import json
import numpy as np
import warnings

warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')    

DataPath = "./data/DatasetFromSDW.csv"
Dataset = pd.read_csv(DataPath)

# On définit les variables qualitatives et quantitatives de notre dataset
var_quali = ["COUNTRY"]
var_index = ["COUNTRY_IDX", "OBS_DATE"]
var_quanti = [x for x in Dataset.columns if x not in var_index if x not in var_quali]

# On définit également une liste de variables de référence si jamais on souhaite pré-encoder
# certaines variables qualitatives ayant beaucoup de modalités
var_ref = []

# On définit nos index pour notre panel
Dataset.set_index(var_index, inplace=True)
Dataset.head()

# Hyperinflation in Bulgaria in 1997
# Dataset = Dataset[Dataset['YEAR'] >= 1999]

# On applique le preprocessing selon diverses options spécifiées
train_scaled, test_scaled, vali_scaled, meta = prep.construct_AE_input_from_cleaned_df(
    Dataset,
    var_quali,
    var_quanti,
    var_ref,
    percentage=[70, 10, 20],
    scaler=MinMaxScaler(),
    FillingMethod="Mean",
)

# On sauvegarde nos fichier train/test/validation
train_scaled.to_csv("data/TrainingSet.csv")
test_scaled.to_csv("data/TestingSet.csv")
vali_scaled.to_csv("data/ValidationSet.csv")


# On enregistre un fichier json qui contient plusieurs informations sur le preprocessing
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


with open("data/Meta.json", "w") as fp:
    json.dump(meta, fp, sort_keys=True, indent=4, cls=NpEncoder)
