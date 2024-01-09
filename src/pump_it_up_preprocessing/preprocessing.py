from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.utils.constant import x_independent_data_path, y_dependent_data_path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def load_dataset(data_path):
    """
       Use Pandas read_csv function to read the dataset from the specified file path.
       Return the loaded dataset.
       """
    data_set = pd.read_csv(data_path)
    return data_set


def drop_unused_feature(dataset, lists):
    """
        :param dataset: The input dataset in a Pandas DataFrame.
        :param lists: A list of feature names to be dropped from the dataset.
        :return: Return the modified dataset.
        """
    datasets = dataset.drop(lists, axis=1)
    return datasets


def merge_training_label_dataset(dataset1, dataset2):
    """
    :param dataset1: The first dataset to be merged in a Pandas DataFrame.
    :param dataset2: The second dataset to be merged in a Pandas DataFrame.
    :return: Return the merged dataset.
    """
    dataset = pd.merge(dataset1, dataset2, on='id', how="inner", validate="many_to_many")
    return dataset


def remove_missing_value(dataset):
    """
    :param dataset: The input dataset in a Pandas DataFrame.
    :return: The dataset after handling missing values.
    """
    return dataset.bfill(limit=1).fillna(0)


def date_parse(dataset):
    """
    :param dataset:The input dataset in a Pandas DataFrame.
    :return: Return the modified dataset.
    """
    dataset.date_recorded = pd.to_datetime(dataset.date_recorded)
    # Extract year, month, and day information from the 'date_recorded' column.
    dataset['record_year'] = dataset['date_recorded'].dt.year
    dataset['record_month'] = dataset['date_recorded'].dt.month
    dataset['record_day_of_months'] = dataset['date_recorded'].dt.day
    # Drop the original 'date_recorded' column.
    dataset.drop(columns=['date_recorded'], axis=1, inplace=True)
    return dataset

# This function creates a heatmap to visualize the correlation between numerical attributes in the dataset.
def create_heat_map(dataset):
    """
    :param dataset: The input dataset in a Pandas DataFrame.
    """
    numerical_vars = [col for col in dataset.columns if
                      dataset[col].dtype in ['int64', 'float64']]
    df = dataset[numerical_vars]
    correlation = df.corr()
    _, _ = plt.subplots(figsize=(17, 17))
    plt.title('Correlation of numerical attributes', size=20)
    colormap = sns.color_palette("BrBG", 10)
    sns.heatmap(correlation, cmap=colormap, annot=True, fmt=".2f")
    plt.show()


def replace_zero_to_nan(mis_column, df):
    """
    :param mis_column: A list of column names where zero values should be replaced with NaN.
    :param df: The input dataset in a Pandas DataFrame.
    :return: The dataset after replacing zero values with NaN in the specified columns.
    """
    for col in mis_column:
        df[col].replace(0, np.nan, inplace=True)
    return df


def fill_mean(cols, dataset):
    """
    :param cols: A list of column names where missing values should be filled with the mean.
    :param dataset: The input dataset in a Pandas DataFrame.
    :return: The dataset after filling missing values with the mean in the specified columns.
    """
    for col in cols:
        dataset[col] = dataset[col].fillna(round(dataset[col].mean()))
    return dataset


def fill_mode(cols, dataset):
    """
    :param cols: A list of column names where missing values should be filled with the mode.
    :param dataset: The input dataset in a Pandas DataFrame.
    :return: The dataset after filling missing values with the mode in the specified columns.
    """
    for col in cols:
        dataset[col] = dataset[col].fillna(round(dataset[col].mode()))
    return dataset


# This function creates a count plot to visualize the number of pumps constructed over the years.
def create_plot_pump_constructed(df):
    """
    :param df: The input dataset in a Pandas DataFrame.
    """
    sns.countplot(x=df["construction_year"], hue=df["status_group"])
    plt.xticks(rotation=90,
               horizontalalignment='right')
    plt.title("Number of pumps constructed over the years", fontsize=14)
    plt.xlabel("Construction year", fontsize=12)
    plt.ylabel("Number of pumps constructed", fontsize=12)
    plt.show()



def scheme_top_data_change(df):
    """
    :param df: A row of the dataset in a Pandas DataFrame.
    :return: The mapped category based on the 'scheme_management' column value.
    """
    if df['scheme_management'] == 'VWC':
        return 'vwc'
    elif df['scheme_management'] == 'WUG':
        return 'wug'
    elif df['scheme_management'] == 'Water authority':
        return 'water_auth'
    elif df['scheme_management'] == 'WUA':
        return 'wua'
    elif df['scheme_management'] == 'Water Board':
        return 'water_brd'
    elif df['scheme_management'] == 'Parastatal':
        return 'parastatal'
    elif df['scheme_management'] == 'Private operator':
        return 'private_opetrator'
    elif df['scheme_management'] == 'SWC':
        return 'swc'
    elif df['scheme_management'] == 'Company':
        return 'company'
    else:
        return 'other'


def funder_top_data_change(df):
    """
    :param df: A row of the dataset in a Pandas DataFrame.
    :return: The mapped category based on the 'funder' column value.
    """
    if df['funder'] == 'Government Of Tanzania':
        return 'gov'
    elif df['funder'] == 'Danida':
        return 'danida'
    elif df['funder'] == 'Hesawa':
        return 'hesawa'
    elif df['funder'] == 'Rwssp':
        return 'rwssp'
    elif df['funder'] == 'World Bank':
        return 'world_bank'
    elif df['funder'] == 'Kkkt':
        return 'kkkt'
    elif df['funder'] == 'World Vision':
        return 'world_vision'
    elif df['funder'] == 'Unicef':
        return 'unicef'
    elif df['funder'] == 'Tasaf':
        return 'tasaf'
    else:
        return 'other'


def extraction_top_data_change(df):
    """
    :param df: A row of the dataset in a Pandas DataFrame.
    :return: The mapped category based on the 'extraction_type' column value.
    """
    if df['extraction_type'] == 'gravity':
        return 'gravity'
    elif df['extraction_type'] == 'nira/tanira':
        return 'nira_tanira'
    elif df['extraction_type'] == 'submersible':
        return 'submersible'
    elif df['extraction_type'] == 'swn 80':
        return 'swn_80'
    elif df['extraction_type'] == 'mono':
        return 'mono'
    elif df['extraction_type'] == 'india mark ii':
        return 'india_mark_ii'
    elif df['extraction_type'] == 'afridev':
        return 'afridev'
    elif df['extraction_type'] == 'ksb':
        return 'ksb'
    elif df['extraction_type'] == 'windmill':
        return 'windmill'
    else:
        return 'other'


def encoding_ordinal(values, col, df):
    """
    :param values:A dictionary containing the mapping of ordinal values to numerical values.
    :param col: The column to be encoded in the dataset.
    :param df:The input dataset in a Pandas DataFrame.
    :return:The dataset after performing ordinal encoding on the specified column.
    """
    df[col] = df[col].map(values)
    return df


def construction_change_to_year_wise(df):
    """
    :param df: A row of the dataset in a Pandas DataFrame.
    :return: The categorized construction year based on decade bins.
    """
    if 1960 <= df['construction_year'] < 1970:
        return '60s'
    elif 1970 <= df['construction_year'] < 1980:
        return '70s'
    elif 1980 <= df['construction_year'] < 1990:
        return '80s'
    elif 1990 <= df['construction_year'] < 2000:
        return '90s'
    elif 2000 <= df['construction_year'] < 2010:
        return '00s'
    elif df['construction_year'] >= 2010:
        return '10s'
    else:
        return 'unknown'


def encoding_label(col, df):
    """
    :param col: The column to be encoded in the dataset.
    :param df: The input dataset in a Pandas DataFrame.
    :return: The dataset after performing label encoding on the specified column.
    """
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    return df


# mean normalization
def normalize_mean(df, columns):
    """
    :param df: The input dataset in a Pandas DataFrame.
    :param columns: A list of column names to be normalized.
    :return: The dataset after mean normalization of specified columns.
    """
    result = df.copy()
    for feature_name in columns:
        mean_value = df[feature_name].mean()
        std_value = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / std_value
    return result


def one_hot_encoding(cat_vars, df):
    """
    :param cat_vars:  A list of column names representing categorical variables to be one-hot encoded.
    :param df: The input dataset in a Pandas DataFrame.
    :return: The dataset after performing one-hot encoding on specified categorical variables.
    """
    ohe = OneHotEncoder()
    feature_arr = ohe.fit_transform(df[cat_vars]).toarray()
    feature_label = ohe.categories_
    ohe_df = pd.DataFrame(feature_arr, columns=feature_label)
    df = pd.concat([df, ohe_df], axis=1)
    return df


def set_feature_target_variable(dataset):
    """
    :param dataset: The input dataset in a Pandas DataFrame.
    :return:
                df_feature: The DataFrame containing feature variables (all columns except 'status_group').
             df_targets: The DataFrame containing the target variable ('status_group').
    """
    df_feature = dataset.drop("status_group", axis='columns')
    df_targets = dataset['status_group']
    return df_feature, df_targets


def split_dataset(df_feature, df_targets):
    """
    :param df_feature: The DataFrame containing feature variables.
    :param df_targets: The DataFrame containing the target variable.
    :return:
            x_train: The training set of feature variables.
            x_test: The testing set of feature variables.
            y_train: The training set of target variable.
            y_test: The testing set of target variable.
    """
    x_train, x_test, y_train, y_test = \
        (train_test_split(df_feature, df_targets, random_state=91, test_size=0.2))
    return x_train, x_test, y_train, y_test


def standardize_feature(x_train, x_test):
    """
    :param x_train: The training set of feature variables.
    :param x_test: The testing set of feature variables.
    :return:
            x_train_scaled: The standardized training set of feature variables.
            x_test_scaled: The standardized testing set of feature variables.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled


def pca_implement(scaled_features):
    """
    :param scaled_features: The standardized feature variables.
    :return: The DataFrame containing principal components obtained from PCA.
    """
    n = 8
    pca = PCA(n_components=n)
    pc = pca.fit_transform(scaled_features)
    columns = [f'PC{i + 1}' for i in range(n)]
    df_pca: DataFrame = pd.DataFrame(data=pc, columns=columns)
    return df_pca


def complete_work_flow_preprocessing():
    # Load independent and dependent datasets
    x_data = load_dataset(x_independent_data_path)
    y_data = load_dataset(y_dependent_data_path)

    # List of features to remove from the dataset
    remove_list = ['id', 'amount_tsh', 'installer', 'wpt_name', 'num_private', 'basin', 'subvillage',
                   'region', 'lga', 'ward', 'scheme_name', 'extraction_type_group',
                   'payment', 'water_quality', 'quantity', 'source', 'source_type', 'waterpoint_type_group',
                   'recorded_by', 'management_group']

    # Merge independent and dependent datasets
    x_data = merge_training_label_dataset(x_data, y_data)

    # Drop unused features from the dataset
    x_data = drop_unused_feature(x_data, remove_list)

    # Remove missing values and parse date information
    x_data = remove_missing_value(x_data)

    # Remove missing values and parse date information
    x_data = date_parse(x_data)

    # Create a heatmap to visualize the correlation between numerical attributes
    create_heat_map(x_data)

    # Replace zero values with NaN in specified columns
    mis_columns = ['construction_year', 'population']
    x_data = replace_zero_to_nan(mis_columns, x_data)

    # Fill missing values with mean and mode
    x_data = fill_mean(["population"], x_data)
    x_data = fill_mode(["construction_year"], x_data)

    # Create a plot to visualize the number of pumps constructed over the years
    create_plot_pump_constructed(x_data)

    # Map values in  column to corresponding categories
    x_data['scheme_management'] = x_data.apply(lambda df: scheme_top_data_change(df), axis=1)
    x_data['funder'] = x_data.apply(lambda df: funder_top_data_change(df), axis=1)
    x_data['extraction_type'] = x_data.apply(lambda df: extraction_top_data_change(df), axis=1)
    x_data['construction_year'] = x_data.apply(lambda df: construction_change_to_year_wise(df), axis=1)

    # Perform ordinal encoding on 'quality_group', 'quantity_group', and 'payment_type' columns
    value = {
        'good': 0,
        'salty': 1,
        'unknown': 2,
        'milky': 3,
        'colored': 4,
        'fluoride': 5
    }
    x_data = encoding_ordinal(value, 'quality_group', x_data)
    value = {
        'enough': 0,
        'insufficient': 1,
        'dry': 2,
        'seasonal': 3,
        'unknown': 4
    }
    x_data = encoding_ordinal(value, 'quantity_group', x_data)
    value = {
        'never pay': 0,
        'per bucket': 1,
        'monthly': 2,
        'unknown': 3,
        'on failure': 4,
        'annually': 5,
        'other': 6
    }

    # Perform label encoding on 'public_meeting' and 'status_group' columns
    x_data = encoding_ordinal(value, 'payment_type', x_data)
    value = {'groundwater': 0, 'surface': 1, 'unknown': 2}
    x_data['source_class'] = x_data['source_class'].map(value)
    value = {
        False: 1,
        True: 0
    }
    x_data = encoding_ordinal(value, 'public_meeting', x_data)
    x_data = encoding_ordinal(value, 'permit', x_data)
    x_data = encoding_label('public_meeting', x_data)
    x_data = encoding_label('status_group', x_data)

    # Normalize 'population' using mean normalization
    x_data = normalize_mean(x_data, ['population'])
    cat_vars = ['funder', 'scheme_management', 'extraction_type']

    # Extract features and target variable
    df_features, df_target = set_feature_target_variable(x_data)

    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = split_dataset(df_features, df_target)

    # One-hot encode categorical variables in training and testing sets
    x_train = one_hot_encoding(cat_vars, x_train)
    x_test = one_hot_encoding(cat_vars, x_test)

    # Standardize feature variables using StandardScaler
    x_train_scaled, x_test_scaled = standardize_feature(x_train, x_test)

    # Implement PCA on standardized features
    x_train_scaled = pca_implement(x_train_scaled)
