import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
import keras

mse = keras.losses.MeanSquaredError()

# On importe les fichiers d'entraînement et de validation
X_train = pd.read_csv("./data/TrainingSet.csv")
X_validation = pd.read_csv("./data/ValidationSet.csv")
X_test = pd.read_csv("./data/TestingSet.csv")

# On définit nos index pour notre panel
var_index =["COUNTRY_IDX","OBS_DATE"]
X_train.set_index(var_index, inplace=True)
X_validation.set_index(var_index, inplace=True)
X_test.set_index(var_index, inplace=True)

ACP_results = {'DimensionEncoder':[],
	            'Pourcentage': [],
                'SizeColumn':[],
                'Seuil' : [],
                'MSETrain':[],
                'MSEValidation': [],
                'MSETest': [],
                'MSETrainDummy':[],
                'MSEValidationDummy': [],
                'MSETestDummy': [],
                'MSETrainNotDummy':[],
                'MSEValidationNotDummy': [],
                'MSETestNotDummy': []}

    
for seuil in ['None'] :

    var_quanti = [x  for x in X_validation.columns if '__' not in x]
    not_dummy = var_quanti
    sample_weight_not_dummy = [x in not_dummy for x in X_validation.columns]
    sample_weight_dummy = [x not in not_dummy for x in X_validation.columns]
    
    SizeColumn = X_train.shape[1]
    for dim_encod in tqdm([0.005,0.01,0.02,0.03, 0.04, 0.045,0.05, 0.055, 0.06, 0.075, 0.08, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25]):

            
            DimensionEncoder = int(X_train.shape[1]*dim_encod)
            print('Fitting...')
            pca = PCA(DimensionEncoder)
            pca.fit_transform(X_train)
            # Encoding
            print('Encoding...')
            X_train_encoded = pca.fit_transform(X_train)
            X_validation_encoded = pca.transform(X_validation)
            X_test_encoded = pca.transform(X_test)
            
            # Predicting
            print('Predicting...')
            X_train_predicted = pd.DataFrame(pca.inverse_transform(X_train_encoded), index=X_train.index, columns = X_train.columns)
            X_validation_predicted = pd.DataFrame(pca.inverse_transform(X_validation_encoded), index=X_validation.index, columns = X_validation.columns)            
            X_test_predicted = pd.DataFrame(pca.inverse_transform(X_test_encoded), index=X_test.index, columns = X_test.columns)
            
            # Calculations
            print('Calculations...')
            mse_total_train= mse(X_train, X_train_predicted).numpy()
            mse_dummy_train= mse(X_train.iloc[:,sample_weight_dummy], X_train_predicted.iloc[:,sample_weight_dummy]).numpy()
            mse_not_dummy_train= mse(X_train.iloc[:,sample_weight_not_dummy], X_train_predicted.iloc[:,sample_weight_not_dummy]).numpy()
            mse_total_vali= mse(X_validation, X_validation_predicted).numpy()
            mse_dummy_vali=mse(X_validation.iloc[:,sample_weight_dummy], X_validation_predicted.iloc[:,sample_weight_dummy]).numpy()
            mse_not_dummy_vali=mse(X_validation.iloc[:,sample_weight_not_dummy], X_validation_predicted.iloc[:,sample_weight_not_dummy]).numpy()
            mse_total_test= mse(X_test, X_test_predicted).numpy()
            mse_dummy_test=mse(X_test.iloc[:,sample_weight_dummy], X_test_predicted.iloc[:,sample_weight_dummy]).numpy()
            mse_not_dummy_test=mse(X_test.iloc[:,sample_weight_not_dummy], X_test_predicted.iloc[:,sample_weight_not_dummy]).numpy()

            # Saving
            ACP_results['DimensionEncoder'].append(DimensionEncoder)
            ACP_results['Pourcentage'].append(dim_encod)
            ACP_results['SizeColumn'].append(SizeColumn) 
            ACP_results['Seuil'].append(seuil)
            ACP_results['MSETrain'].append(mse_total_train)
            ACP_results['MSEValidation'].append(mse_total_vali) 
            ACP_results['MSETest'].append(mse_total_test) 
            ACP_results['MSETrainDummy'].append(mse_dummy_train)
            ACP_results['MSEValidationDummy'].append(mse_dummy_vali)
            ACP_results['MSETestDummy'].append(mse_dummy_test)
            ACP_results['MSETrainNotDummy'].append(mse_not_dummy_train)
            ACP_results['MSEValidationNotDummy'].append(mse_not_dummy_vali)
            ACP_results['MSETestNotDummy'].append(mse_not_dummy_test)
      
# On sauvegarde les resultats
ACP_results = pd.DataFrame(ACP_results)
ACP_results.to_csv('data/ACP_results.csv')
