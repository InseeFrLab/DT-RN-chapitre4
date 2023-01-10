import keras
from tensorflow.keras.optimizers import RMSprop
import training.Autoencoder as TrainAE
import pandas as pd
from tqdm import tqdm

# On importe les fichiers d'entraînement et de validation
X_train = pd.read_csv("./data/TrainingSet.csv")
X_validation = pd.read_csv("./data/ValidationSet.csv")
X_test = pd.read_csv("./data/TestingSet.csv")

# On définit nos index pour notre panel
var_index = ["COUNTRY_IDX", "OBS_DATE"]
X_train.set_index(var_index, inplace=True)
X_validation.set_index(var_index, inplace=True)
X_test.set_index(var_index, inplace=True)

# On définit les options que l'on souhaite appliquer lors de l'entraînement
epochs = 30
batch_size = 25
optimizer = RMSprop()
loss = "mse"
dimensions = [X_train.shape[1], 44, X_train.shape[1]]
activations = ["relu", "sigmoid"]
encoded_layer = 2
# alphas = [0.5, 0.3,0.5]
# dropouts = [0.7, 0.3,0.7,0.3, 0.7]

# On entraîne l'Autoencoder
losses, encoder, autoencoder, model = TrainAE.AE(
    X_train,
    X_validation,
    epochs,
    batch_size,
    optimizer,
    loss,
    dimensions,
    activations,
    encoded_layer,
)


# On peut également entraîner plusieurs autocodeur afin d'analyser les effets des
# différents hyperparamètres

epochs = 50
batch_size = 25
loss = "mse"
activations = ["relu", "relu", "relu", "sigmoid"]
encoded_layer = 5
mse = keras.losses.MeanSquaredError()

AE_results = {
    "DimensionEncoder": [],
    "Pourcentage": [],
    "SizeColumn": [],
    "Seuil": [],
    "MSETrain": [],
    "MSEValidation": [],
    "MSETest": [],
    "MSETrainDummy": [],
    "MSEValidationDummy": [],
    "MSETestDummy": [],
    "MSETrainNotDummy": [],
    "MSEValidationNotDummy": [],
    "MSETestNotDummy": [],
}


for seuil in ["None"]:

    var_quanti = [x for x in X_validation.columns if "__" not in x]
    not_dummy = var_quanti
    sample_weight_not_dummy = [x in not_dummy for x in X_validation.columns]
    sample_weight_dummy = [x not in not_dummy for x in X_validation.columns]

    SizeColumn = X_train.shape[1]
    for dim_encod in tqdm(
        [
            0.005,
            0.01,
            0.02,
            0.03,
            0.04,
            0.045,
            0.05,
            0.055,
            0.06,
            0.075,
            0.08,
            0.1,
            0.125,
            0.15,
            0.175,
            0.2,
            0.25,
        ]
    ):

        DimensionEncoder = int(X_train.shape[1] * dim_encod)
        print("\nTraining AE...")
        dimensions = [X_train.shape[1], 250, DimensionEncoder, 250, X_train.shape[1]]
        losses, encoder, history, Autoencoder = TrainAE.AE(
            X_train,
            X_validation,
            epochs,
            batch_size,
            RMSprop(),
            loss,
            dimensions,
            activations,
            encoded_layer,
            verbose=0,
        )

        # Predicting
        print("\nPredicting...")
        X_train_predicted = pd.DataFrame(
            Autoencoder.predict(X_train), index=X_train.index, columns=X_train.columns
        )
        X_validation_predicted = pd.DataFrame(
            Autoencoder.predict(X_validation),
            index=X_validation.index,
            columns=X_validation.columns,
        )
        X_test_predicted = pd.DataFrame(
            Autoencoder.predict(X_test), index=X_test.index, columns=X_test.columns
        )

        # Calculations
        print("\nCalculations...")
        mse_total_train = mse(X_train, X_train_predicted).numpy()
        mse_dummy_train = mse(
            X_train.iloc[:, sample_weight_dummy],
            X_train_predicted.iloc[:, sample_weight_dummy],
        ).numpy()
        mse_not_dummy_train = mse(
            X_train.iloc[:, sample_weight_not_dummy],
            X_train_predicted.iloc[:, sample_weight_not_dummy],
        ).numpy()
        mse_total_vali = mse(X_validation, X_validation_predicted).numpy()
        mse_dummy_vali = mse(
            X_validation.iloc[:, sample_weight_dummy],
            X_validation_predicted.iloc[:, sample_weight_dummy],
        ).numpy()
        mse_not_dummy_vali = mse(
            X_validation.iloc[:, sample_weight_not_dummy],
            X_validation_predicted.iloc[:, sample_weight_not_dummy],
        ).numpy()
        mse_total_test = mse(X_test, X_test_predicted).numpy()
        mse_dummy_test = mse(
            X_test.iloc[:, sample_weight_dummy],
            X_test_predicted.iloc[:, sample_weight_dummy],
        ).numpy()
        mse_not_dummy_test = mse(
            X_test.iloc[:, sample_weight_not_dummy],
            X_test_predicted.iloc[:, sample_weight_not_dummy],
        ).numpy()

        # Saving
        AE_results["DimensionEncoder"].append(DimensionEncoder)
        AE_results["Pourcentage"].append(dim_encod)
        AE_results["SizeColumn"].append(SizeColumn)
        AE_results["Seuil"].append(seuil)
        AE_results["MSETrain"].append(mse_total_train)
        AE_results["MSEValidation"].append(mse_total_vali)
        AE_results["MSETest"].append(mse_total_test)
        AE_results["MSETrainDummy"].append(mse_dummy_train)
        AE_results["MSEValidationDummy"].append(mse_dummy_vali)
        AE_results["MSETestDummy"].append(mse_dummy_test)
        AE_results["MSETrainNotDummy"].append(mse_not_dummy_train)
        AE_results["MSEValidationNotDummy"].append(mse_not_dummy_vali)
        AE_results["MSETestNotDummy"].append(mse_not_dummy_test)

# On sauvegarde les resultats
AE_results = pd.DataFrame(AE_results)
AE_results.to_csv("data/AE_Results.csv")
