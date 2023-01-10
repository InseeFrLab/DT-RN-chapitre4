import pandas as pd
import numpy as np
import random
import pickle
from functools import reduce
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def charge_df(filepath):
    """
    This function aims to open a csv file as a pandas dataframe.

    args:
        filepath : (str) path to the specified csv

    returns:
        df : (pandas DataFrame) Dataframe of the file in the specified path
    """
    df = pd.read_csv(filepath)
    return df


def nan_in_df(df):
    """
    This function aims to check if there exists NaN values in a pandas dataframe.

    args:
        df : (pandas DataFrame)

    returns:
        (bool) : whether there are NaN in the specified dataframe
    """
    return df.isnull().values.any()


def all_nan_in_column(df, col):
    """
    This function aims to check if there exists NaN values in a pandas dataframe.

    args:
        df : (pandas DataFrame)

    returns:
        (bool) : whether there are NaN in the specified dataframe
    """
    return df[col].isnull().values.all()


# Enquete Emploi :


def load_lists():
    """
    This function creates every list of variables that we need fo our analysis of French Labour Survey
    You can replace this step with the list needed in you own dataset

    returns:
        var_quali : (list) List of categorical variables
        var_quanti : (list) List of quantitative variables
        var_index : (list) List of index variables, used to identify respondants
        var_dates : (list) List of dates variables
        var_to_drop : (list) List of variables that are not needed in our analysis.
                             In our case, it contains variable that allow to identify responders,
                             and are thus non necessary in the analysis.
        var_replace_minus1 : (list) List of  quantitative variables where modality -1 means something special
        var_replace_99999: (list) List of quantitative variables where modality 9999, 99999 or 999999 means 'do not know'
    """
    pickle_in = open("./Data/Variables_EE/var_quali.pickle", "rb")
    var_quali = pickle.load(pickle_in)
    pickle_in = open("./Data/Variables_EE/var_quanti.pickle", "rb")
    var_quanti = pickle.load(pickle_in)
    pickle_in = open("./Data/Variables_EE/var_index.pickle", "rb")
    var_index = pickle.load(pickle_in)
    pickle_in = open("./Data/Variables_EE/var_dates.pickle", "rb")
    var_dates = pickle.load(pickle_in)
    pickle_in = open("./Data/Variables_EE/var_to_drop.pickle", "rb")
    var_to_drop = pickle.load(pickle_in)
    pickle_in = open("./Data/Variables_EE/var_replace_minus1.pickle", "rb")
    var_replace_minus1 = pickle.load(pickle_in)
    pickle_in = open("./Data/Variables_EE/var_replace_99999.pickle", "rb")
    var_replace_99999 = pickle.load(pickle_in)
    return (
        var_quali,
        var_quanti,
        var_index,
        var_dates,
        var_to_drop,
        var_replace_minus1,
        var_replace_99999,
    )


def load_lists_final():
    """
    This function loads every list of variables that we need fo the creation of our dataset for NN.

    returns:
        var_quali : (list) List of categorical variables
        var_quanti : (list) List of quantitative variables
        var_index : (list) List of index variables, used to identify respondants
        var_ref : (list) List of variables used as reference
    """
    pickle_in = open("./Data/Output/var_quali_final.pickle", "rb")
    var_quali = pickle.load(pickle_in)
    pickle_in = open("./Data/Output/var_quanti_final.pickle", "rb")
    var_quanti = pickle.load(pickle_in)
    pickle_in = open("./Data/Output/var_index_final.pickle", "rb")
    var_index = pickle.load(pickle_in)
    pickle_in = open("./Data/Output/var_ref_final.pickle", "rb")
    var_ref = pickle.load(pickle_in)
    return var_quali, var_quanti, var_index, var_ref


def remove_variables_not_in_df(
    df,
    var_quali,
    var_quanti,
    var_index,
    var_dates,
    var_to_drop,
    var_replace_minus1,
    var_replace_99999,
):
    """
    This function aims to delete from the lists specified above every variable that does not appear in the dataframe studied.
    If lists were created using only the dataset, skip this step.

    args:
        df (pandas DataFrame) DataFrame under study
        var_quali : (list) List of categorical variables
        var_quanti : (list) List of quantitative variables
        var_index : (list) List of index variables, used to identify respondants
        var_dates : (list) List of dates variables
        var_to_drop : (list) List of variables that are not needed in our analysis.
                             In our case, it contains variable that allow to identify responders,
                             and are thus non necessary in the analysis.
        var_replace_minus1 : (list) List of  quantitative variables where modality -1 means something special
        var_replace_99999: (list) List of quantitative variables where modality 9999, 99999 or 999999 means 'do not know'

    returns: Same lists but without variables that are not in the studied dataset
        var_quali : (list) List of categorical variables
        var_quanti : (list) List of quantitative variables
        var_index : (list) List of index variables, used to identify respondants
        var_dates : (list) List of dates variables
        var_to_drop : (list) List of variables that are not needed in our analysis.
                             In our case, it contains variable that allow to identify responders,
                             and are thus non necessary in the analysis.
        var_replace_minus1 : (list) List of  quantitative variables where modality -1 means something special
        var_replace_99999: (list) List of quantitative variables where modality 9999, 99999 or 999999 means 'do not know'
    """
    columns_df = df.columns.unique().tolist()
    extra = [
        x
        for x in var_to_drop
        + var_replace_99999
        + var_replace_minus1
        + var_dates
        + var_index
        + var_quanti
        + var_quali
        if x not in columns_df
    ]
    for liste in (
        var_to_drop,
        var_replace_99999,
        var_replace_minus1,
        var_dates,
        var_index,
        var_quanti,
        var_quali,
    ):
        for x in extra:
            if x in liste:
                liste.remove(x)
    return (
        var_quali,
        var_quanti,
        var_index,
        var_dates,
        var_to_drop,
        var_replace_minus1,
        var_replace_99999,
    )


def solve_minus1_issue(var_quanti, var_replace_minus1):
    """
    This functions solve the issue of quantitative variables that had the modality "-1" encoded in a particular way.
    The way we chose to handle this issue is simply ignoring it.
    args:
        var_quanti : (list) List of quantitative variables
        var_replace_minus1 : (list) List of  quantitative variables where modality -1 means something special

    returns:
        var_quanti : (list) List of quantitative variables
    """
    var_quanti.extend(var_replace_minus1)
    return var_quanti


def solve_99999_issue(df, var_quanti, var_replace_99999):
    """
    This function handle the issue with 99999 values. We solve this issue by replacing all the identified values by a np.nan
    In our case, this method seemt to be the most efficient way of replacing values in a pandas DataFrame.
    It is twice as fast as pandas.replace function

    args:
        df : (pandas DataFrame) dataset studied
        var_quanti : (list) List of quantitative variables
        var_replace_99999: (list) List of quantitative variables where modality 9999, 99999 or 999999 means 'do not know'
    returns:
        df : (pandas DataFrame) Datafram cleaned of non wanted values
        var_quanti : (list) list of quantitative variable extended with the variables from the other list
    """
    dic = {"9999999": np.nan, "9999998": np.nan, "9999": np.nan}
    for var in var_replace_99999:
        df[var] = df[var].map(lambda x: dic.get(x, x))
    var_quanti.extend(var_replace_99999)
    return df, var_quanti


def create_quali_multimod_list(df, var_quali, threshold):
    """
    This function is used to create a list of variables that has more than a certain threshold modalities.
    We used it as some variables in FLS have more than 500 modalities, and thus makes it difficult to simply one-hot encode them

    args:
        df : (pandas DataFrame) Dataset studied
        var_quali : (list) List of categorical variables
        threshold : threshold above which variables are included in the other list, and excluded of var_quali.

    returns:
        var_quali : (list) List of categorical variables that have less than threshold modalities
        var_quali_multimod : (list) List of categorical variables that have more than threshold modalities
    """

    if threshold == None:
        var_quali_multimod = []
    else:

        var_quali_multimod = [
            var for var in var_quali if len(pd.unique(df[var])) > threshold
        ]
        var_quali = [x for x in var_quali if x not in var_quali_multimod]
    return var_quali_multimod, var_quali


def clean_df(df, index=None):
    """
    This function is used to clean the DataFrame.
    Indeed, the French Labour Survey (at least the first year) have many mistakes.
    This function solve most of this issues.

    args:
        df : (pandas DataFrame) DataFrame studied
        index : (index) list of variables that are used as index
                Default = None

    returns:
        df : (pandas DataFrame) cleaned dataframe with index set.
    """
    if index != None:
        df.set_index(index, inplace=True)
    df = df.apply(lambda x: x.apply(str) if x.dtype == "object" else x)
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df.replace([""], [np.nan], inplace=True)
    df.replace(["nan"], [np.nan], inplace=True)
    df.replace(["."], [np.nan], inplace=True)
    df.replace([None], [np.nan], inplace=True)
    return df


def create_quali_df(df, var_quali, prefix_sep="__", dummy_na=True, drop_first=False):
    """
    This functions create a one hot encoding of variables contained in a list.

    args:
        df : (pandas DataFrame) DataFrame studied
        var_quali : (list) List of categorical variables to be one hot encoded
    The other arguments are those from pandas get_dummies function:
        prefix_sep :    (str) separator between the variable name and the modality
                        Default = '__'
        dummy_na :      (bool) Whether to create a dummy for NaNs
                        Default = True
        drop_first :    (bool) Whether to drop one of the dummy for each variable
                        Default = True

    returns:
        df_quali : (pandas DataFrame) Dataframe of one hot-encoded variables
    """
    df_quali = pd.get_dummies(
        df[var_quali],
        prefix_sep=prefix_sep,
        columns=var_quali,
        dummy_na=dummy_na,
        drop_first=drop_first,
    )
    return df_quali


def create_quanti_df(df, var_quanti, suffix="__nan"):
    """
    This function creates the dataframe of quantitative variable and a dummy for the modality NaN

    args:
        df : (pandas DataFrame) DataFrame studied
        var_quanti : (list) List of quantitative variables
        suffix :    (str) suffix after the variable name
                        Default = '__nan'
    returns:
        df : (pandas DataFrame) Dataframe with quantitative variable and a nan dummy for every variable that contains NaNs
    """
    df = df[var_quanti].astype(np.float32)
    if nan_in_df(df):
        bool_quanti = df.isna().astype(np.uint8).add_suffix(suffix)
        return pd.concat([df, bool_quanti], axis=1)
    else:
        return df


def regroup_df(list_of_dataframes, how="inner", right_index=True, left_index=True):
    """
    This functions merges all dataframes in a list into a single one.

    args:
        list_of_dataframe : (list) list containing all pandas dataframes that want to be merged
    Other arguments are those of the pandas merge function. See docs for more detailed explainations:
        how : (str)  method for merging
                Default = 'inner'
        right_index:    merge on right index
                        Default = True
        left_index:     merge on left index
                        Default=True
    returns:
        (pandas DataFrame) merged DataFrame
    """
    return reduce(
        lambda left, right: pd.merge(
            left, right, how=how, left_index=left_index, right_index=right_index
        ),
        list_of_dataframes,
    )


def custom_train_test_vali_split(
    df, index=True, percentage=[70, 10, 20], random_seed=257
):
    """
    This function is used to split our dataset in training, test and validation set.
    Each person must be in only one of those sets, otherwise the network would learn information on the person.
    Thus, we need to identify each unique person and then split according to those unique persons.

    args:
        df : (pandas DataFrame) DataFrame that need to be splitted in 3 sets
        index : (bool) True if there is an index to split data
        percentage : (list of int) each int is the fraction of unique person in each dataset.
                        Default =[70, 10, 20] : first set is 70%, second is 10 and last is 20% of unique individuals
        random_seed : (int) random seed for random tool.
                        Default = 257

    returns :
        3 datasets, each one representing the specified number of unique individuals in the whole dataframe.
    """
    if index == False:
        raise KeyError("Cannot split without index")
    else:
        random.seed(random_seed)
        liste_id = df.index.values
        sizes = [round(pourc * len(liste_id) / 100) for pourc in percentage]

        if sum(sizes) != len(liste_id):
            sizes[2] = sizes[2] - (sum(sizes) - len(liste_id))
        print(
            "Les taille (nombre de lignes) des échantillons de training, de validation et de testing sont les suivantes: ",
            sizes,
        )

        it = iter(liste_id)
        liste_id_fin = [[next(it) for _ in range(size)] for size in sizes]
        # sample(frac=df, replace=False, random_state=257)

        return df.loc[liste_id_fin[0]], df.loc[liste_id_fin[1]], df.loc[liste_id_fin[2]]


def split_x_y(df, var_interest):
    """
    This function is used to split a dataset in x and y.

    args:
        df : (pandas DataFrame) df studied
        var_interest : (str or list of strings) variable of interest in the study

    returns:
        x : dataframe whitout the interest variable
        y : dataframe with only the interest variable
    """
    y = df[var_interest]
    x = df.drop(var_interest, axis=1)
    return x, y


def update_metadata(meta_dict, *tuples):
    """
    This function is used to update a dict of metadata with new dataframes.

    Args:
        meta_dict (dict) : Initial dict of metadata
        *tuples (tuples (string, pandas DataFrame)) : the string will be the key for metadata about this dataframe in the metadict

    Returns:
        meta_dict (dict) : dictionnary augmented with information on each column of each dataframe in *tuples

    """
    if type(meta_dict) != dict:
        raise ValueError("meta must be a dict item")
    for name, df in tuples:
        meta_dict[name] = {
            "shape": df.shape,
            "Min": dict(df.min()),
            "Max": dict(df.max()),
            "Mean": dict(df.mean()),
            "Variance": dict(df.var()),
            # "Unique":{col:pd.unique(df[col]).tolist() for col in df.columns},
            "NaN_count": dict(df.isna().sum()),
        }
    return meta_dict


def create_metadata(
    var_quanti, var_quali, var_quali_multimod, exceptions, total_columns
):
    """
    This function is used to create a meta_dictionnary that stores information.

    Args:
        var_quanti (list) : List of quantitative variables
        var_quali (list) : List of categorical variables that have less than threshold modalities
        var_quali_multimod (list) : List of categorical variables that have more than threshold modalities
        exceptions (list) : List of variables that sould not be normalised. We use this for dummies that do not need to be normalized
        total_columns (list) : Total number of columns under study

    Returns:
        meta_dict (dict) : dictionnary containing the different lists
    """
    meta_dict = {
        "quali": var_quali,
        "quanti": var_quanti,
        "multimodality": var_quali_multimod,
        "dummies": exceptions,
        "normalized": [col for col in total_columns if col not in exceptions],
    }
    return meta_dict


def normalisation_df(train, test, vali, scaler, exceptions, FillingMethod="Mean"):
    """
    This function aims to normalize the train test and validation set,
    and create a dictionnary containing detailed description of data.

    args:
        train : (pandas DataFrame)
        test : (pandas DataFrame)
        vali : (pandas DataFrame)
        scaler : MinMaxScaler() or StandardScaler()
        exceptions : (list) List of variables that sould not be normalised. We use this for dummies that do not need to be normalized
        FillingMethod : (str) ["Mean", "Median"]

    returns:
        train_scaled : scaled version of train set
        test_scaled : scaled version of test set
        vali_scaled : scaled version of validation set
        scaler : scaler informations
    """
    to_scale = [col for col in train.columns if col not in exceptions]
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train[to_scale]).astype(np.float32),
        index=train.index,
        columns=train[to_scale].columns,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test[to_scale]).astype(np.float32),
        index=test.index,
        columns=test[to_scale].columns,
    )
    vali_scaled = pd.DataFrame(
        scaler.transform(vali[to_scale]).astype(np.float32),
        index=vali.index,
        columns=vali[to_scale].columns,
    )

    Training = train_scaled.merge(train[exceptions], left_index=True, right_index=True)
    Testing = test_scaled.merge(test[exceptions], left_index=True, right_index=True)
    Validation = vali_scaled.merge(vali[exceptions], left_index=True, right_index=True)

    if FillingMethod == "Mean":
        Training.fillna(train_scaled.mean(), inplace=True)
        Testing.fillna(test_scaled.mean(), inplace=True)
        Validation.fillna(vali_scaled.mean(), inplace=True)
    elif FillingMethod == "Median":
        Training.fillna(train_scaled.median(), inplace=True)
        Testing.fillna(test_scaled.median(), inplace=True)
        Validation.fillna(vali_scaled.median(), inplace=True)
    else:
        raise ValueError(
            """ This method is not supported.\nImplemented methods are "Mean" and "Median" """
        )

    return Training, Testing, Validation, scaler


def getManualVectorisation(key, dic):
    """
    This function is used to create our vectorized df
    It takes a key and a dictionnary, and returns the value associated to this key.
    If a key error was to be raised, it returns the last value in the dictionnary. We made this exception because :
    np.nan != np.nan and np.nan is always the last value created by groupby().
    Be cautious using this, because if the specified key is not in the dictionnary, it would not raise an error.
    """
    try:
        return dic[key]
    except KeyError:
        return list(dic.values())[-1]


def check_all_nan(liste_df):
    """
    This function checks for columns full of nan in a list of dataframes
    And removes each column that is problematic for autoencoders

    args:
        liste_df (list of dataframes)

    returns
        liste_df (list of dataframes) cleaned of problematic columns.
    """
    to_remove = list(
        set(
            [
                col
                for df in liste_df
                for col in df.columns
                if df[col].isna().sum() == len(df)
            ]
        )
    )
    for df in liste_df:
        df.drop(to_remove, axis=1, inplace=True)
    print("Les variables suivantes ont été supprimés des dataframes : ", to_remove)
    return liste_df


def save_as_pickle(dic, folder_path="./Data/Output/"):
    """
    This function takes as input a dict where keys are file names and items are whatever python object you want
    And saves each object as the name provided in key.
    If the folder path provided does not exist, it creates one.
    args:
        dic_df : (dict) dict where keys are file names and items are whatever python object you want
        folder_path : (str) folder in which the file is saved
                Default = "./Data/Output/"
    returns:
        DataFrame saved under pickle format in path folder
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    for name, item in dic.items():
        save = open(folder_path + name + ".pickle", "wb")
        pickle.dump(item, save, protocol=4)
        save.close()
    return "SAVING DONE"


def construct_AE_input_from_cleaned_df(
    df,
    var_quali,
    var_quanti,
    var_ref,
    threshold=None,
    percentage=[70, 10, 20],
    scaler=MinMaxScaler(),
    ManualVectorisationMethod="Mean",
    FillingMethod="Mean",
    random_seed=257,
):
    """
    This function is used to preprocess our dataframe. It computes all steps and returns 3 set (training, validation and test) ready to be used in our auto-encoder
    args:
        df : (pandas Dataframe)
        var_quali : (list)
        var_quanti : (list)
        var_ref : (list)
        threshold : (int) threshold above which variables are included in the var_quali_multimod, and excluded of var_quali
        percentage : (list of int) each float is the fraction of unique person in each dataset.
                        Default =[70, 10, 20] : int set is 70%, second is 10% and last is 20% of unique individuals
        scaler : MinMaxScaler() or StandardScaler()
        ManualVectorisationMethod : (str) "Mean" or "Median" are supported for the moment
        FillingMethod : (str) argument for normalisation_df function "Mean" or "Median" are supported
        random_seed : (int) random seed for random python package

    returns:
        train_scaled : training set normalized using specified scaler
        test_scaled : test set normalized using specified scaler
        vali_scaled : validation set normalized using specified scaler
        meta : json dictionnary containing infroamtion on each variable of the original dataset
    """

    # Create df_quanti,
    if not var_quanti:
        pass
    else:
        print("Creating the quantitative dataset...")
        df_quanti = create_quanti_df(df, var_quanti)
        print("DONE!")

    # Create df_quali
    if not var_quali:
        var_quali_multimod = []
    else:
        var_quali_multimod, var_quali = create_quali_multimod_list(
            df, var_quali, threshold=threshold
        )
        print("Creating the categorical dataset...")
        df_quali = create_quali_df(df, var_quali)
        print("DONE!")

        # Create vectorised_df
        if threshold is None:
            pass
        else:
            print("Creating the manual vectorised dataset...")
            df_cropped = df[var_ref + var_quali_multimod]
            vectorised_df = pd.DataFrame()
            for variable in tqdm(var_quali_multimod):
                for ref in var_ref:
                    if ManualVectorisationMethod == "Mean":
                        means_dict = {
                            index: row
                            for index, row in df_cropped.groupby(
                                variable, dropna=False
                            )[ref]
                            .mean()
                            .iteritems()
                        }
                    elif ManualVectorisationMethod == "Median":
                        means_dict = {
                            index: row
                            for index, row in df_cropped.groupby(
                                variable, dropna=False
                            )[ref]
                            .median()
                            .iteritems()
                        }
                    else:
                        raise ValueError(
                            "This method is not supported.\nImplemented methods are 'Mean' and 'Median'"
                        )
                    vectorised_df[variable + "__" + ref] = df_cropped[variable].apply(
                        lambda x: getManualVectorisation(x, means_dict)
                    )
            print("DONE!")

    # Concatenation
    print("Merging datasets...")
    if "df_quanti" in locals():
        if "df_quali" in locals():
            if "vectorised_df" in locals():
                df_total = regroup_df([df_quanti, df_quali, vectorised_df])
            else:
                df_total = regroup_df([df_quanti, df_quali])

            not_to_scale = df_quali.columns.tolist() + [
                x for x in df_quanti.columns.tolist() if "__nan" in x
            ]

        else:
            df_total = df_quanti
            not_to_scale = [x for x in df_quanti.columns.tolist() if "__nan" in x]
    else:
        if "vectorised_df" in locals():
            df_total = regroup_df([df_quali, vectorised_df])
        else:
            df_total = df_quali
        not_to_scale = df_quali.columns.tolist()

    print("DONE!")

    # Create MetaData:
    print("Creating Meta Dict...")
    meta = create_metadata(
        var_quanti, var_quali, var_quali_multimod, not_to_scale, df_total.columns
    )
    meta = update_metadata(meta, ("all", df_total))
    print("DONE!")

    # Split test train vali
    print("Splitting in training, validation and testing set...")
    df_train, df_vali, df_test = custom_train_test_vali_split(
        df_total, percentage=percentage, random_seed=random_seed
    )
    print("DONE!")

    # Update Meta:
    print("Updating Meta Dict...")
    meta = update_metadata(
        meta, ("train", df_train), ("vali", df_vali), ("test", df_test)
    )
    print("DONE!")
    # Normalization
    print("Normalizing training, validation and testing set...")
    train_scaled, test_scaled, vali_scaled, scaler = normalisation_df(
        df_train,
        df_test,
        df_vali,
        scaler=scaler,
        exceptions=not_to_scale,
        FillingMethod=FillingMethod,
    )
    print("DONE!")

    # Check NaNs
    print("Checking for columns full of nan...")
    train_scaled, test_scaled, vali_scaled = tuple(
        check_all_nan([train_scaled, test_scaled, vali_scaled])
    )
    print("DONE")
    return train_scaled, test_scaled, vali_scaled, meta
