import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # or RobustScaler
from torch.utils.data import Dataset


class MetabricDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.features[idx]
        label = self.labels[idx]
        return data, label


def load_and_preprocess_data(path: str):
    """Load and preprocess the METABRIC dataset.

    Args:
        path (str): Path to the dataset csv file.

    Returns:
        dict: A dictionary containing the preprocessed dataframe, numeric and categorical columns,
              and train, test, val datasets.
        - df: Preprocessed dataframe.
        - numeric_cols: List of numeric column names.
        - categorical_cols: List of categorical column names.
        - sets: A dictionary with keys 'train', 'test', 'val' containing respective datasets.
    """

    df = pd.read_csv(path)

    patient_meta_cols = ["Patient ID", "Oncotree Code", "Cohort"]
    demographic_cols = ["Age at Diagnosis", "Sex", "Inferred Menopausal State"]
    cancer_characteristics_cols = [
        "Cancer Type",
        "Cancer Type Detailed",
        "Cellularity",
        "Neoplasm Histologic Grade",
        "Tumor Other Histologic Subtype",
        "Tumor Size",
        "Tumor Stage",
        "Primary Tumor Laterality",
    ]
    biomarker_cols = [
        "ER Status",
        "ER status measured by IHC",
        "PR Status",
        "HER2 Status",
        "HER2 status measured by SNP6",
        "Pam50 + Claudin-low subtype",
        "3-Gene classifier subtype",
        "Integrative Cluster",
    ]
    treatment_cols = [
        "Type of Breast Surgery",
        "Hormone Therapy",
        "Chemotherapy",
        "Radio Therapy",
    ]
    clinical_outcome_cols = [
        "Overall Survival (Months)",
        "Overall Survival Status",
        "Patient's Vital Status",
        "Relapse Free Status (Months)",
        "Relapse Free Status",
    ]
    clinical_metrics_cols = [
        "Lymph nodes examined positive",
        "Mutation Count",
        "Nottingham prognostic index",
    ]

    df["Type of Breast Surgery"] = (
        df.groupby(["Cancer Type Detailed"])["Type of Breast Surgery"]
        .apply(lambda x: x.fillna(x.mode()[0]))
        .sort_index(level=1)
        .values
    )

    # Fill event and duration null values with the mean or the most frequent value
    grouped_df = df.groupby(["Cancer Type Detailed"])["Relapse Free Status"].apply(
        lambda x: x.fillna(x.mode()[0])
    )
    df["Relapse Free Status"] = grouped_df.sort_index(level=1).values
    grouped_df = df.groupby(["Cancer Type Detailed"])[
        "Relapse Free Status (Months)"
    ].apply(lambda x: x.fillna(x.mean()))
    df["Relapse Free Status (Months)"] = grouped_df.sort_index(level=1).values

    # Since the relapse free status has two classes, we have to have two level of grouping
    grouped_df = df.groupby(["Cancer Type Detailed", "Relapse Free Status"])[
        "Overall Survival Status"
    ].apply(lambda x: x.fillna(x.mode()[0]))
    df["Overall Survival Status"] = grouped_df.sort_index(level=2).values
    grouped_df = df.groupby(["Cancer Type Detailed", "Overall Survival Status"])[
        "Overall Survival (Months)"
    ].apply(lambda x: x.fillna(x.mean()))
    df["Overall Survival (Months)"] = grouped_df.sort_index(level=2).values

    # Fill null values in treatment columns with mode of each cancer type detailed
    # For each cancer type detailed, find the mode of each treatment column and fill null values with that mode
    grouped_df = df.groupby(["Cancer Type Detailed"])["Chemotherapy"].apply(
        lambda x: x.fillna(x.mode()[0])
    )
    df["Chemotherapy"] = grouped_df.sort_index(level=1).values
    grouped_df = df.groupby(["Cancer Type Detailed"])["Hormone Therapy"].apply(
        lambda x: x.fillna(x.mode()[0])
    )
    df["Hormone Therapy"] = grouped_df.sort_index(level=1).values
    grouped_df = df.groupby(["Cancer Type Detailed"])["Radio Therapy"].apply(
        lambda x: x.fillna(x.mode()[0])
    )
    df["Radio Therapy"] = grouped_df.sort_index(level=1).values

    df["ER status measured by IHC"] = df["ER status measured by IHC"].fillna(
        df["ER status measured by IHC"].mode()[0]
    )
    grouped_df = df.groupby(["ER status measured by IHC"])["ER Status"].apply(
        lambda x: x.fillna(x.mode()[0])
    )
    df["ER Status"] = grouped_df.sort_index(level=1).values
    df["HER2 status measured by SNP6"] = df["HER2 status measured by SNP6"].fillna(
        df["HER2 status measured by SNP6"].mode()[0]
    )
    grouped_df = df.groupby(["HER2 status measured by SNP6"])["HER2 Status"].apply(
        lambda x: x.fillna(x.mode()[0])
    )
    df["HER2 Status"] = grouped_df.sort_index(level=1).values
    grouped_df = df.groupby(["Cancer Type Detailed"])["PR Status"].apply(
        lambda x: x.fillna(x.mode()[0])
    )
    df["PR Status"] = grouped_df.sort_index(level=1).values

    df["Pam50 + Claudin-low subtype"] = df["Pam50 + Claudin-low subtype"].fillna(
        df["Pam50 + Claudin-low subtype"].mode()[0]
    )
    df["3-Gene classifier subtype"] = (
        df.groupby(["Cancer Type Detailed"])["3-Gene classifier subtype"]
        .apply(lambda x: x.fillna(x.mode()[0]))
        .sort_index(level=1)
        .values
    )
    df["Integrative Cluster"] = (
        df.groupby(["Cancer Type Detailed"])["Integrative Cluster"]
        .apply(lambda x: x.fillna(x.mode()[0]))
        .sort_index(level=1)
        .values
    )

    df["Cohort"] = (
        df.groupby(["Cancer Type Detailed"])["Cohort"]
        .apply(lambda x: x.fillna(x.mode()[0]))
        .sort_index(level=1)
        .values
    )

    # Fill null age at diagnosis with the mean age of each cancer type detailed
    df["Age at Diagnosis"] = (
        df.groupby(["Cancer Type Detailed"])["Age at Diagnosis"]
        .apply(lambda x: x.fillna(x.mean()))
        .sort_index(level=1)
        .values
    )

    # Fill the inferred menopausal state with the common state
    df["Inferred Menopausal State"] = df["Inferred Menopausal State"].fillna(
        df["Inferred Menopausal State"].mode()[0]
    )

    df["Cellularity"] = (
        df.groupby(["Cancer Type Detailed"])["Cellularity"]
        .apply(lambda x: x.fillna(x.mode()[0]))
        .sort_index(level=1)
        .values
    )
    # Need to have fall back mechanism as their exists cases where all cellularity values are null for a given cancer type detailed
    df["Tumor Stage"] = (
        df.groupby(["Cancer Type Detailed", "Cellularity"])["Tumor Stage"]
        .apply(lambda x: x.fillna(x.median()))
        .sort_index(level=2)
        .values
    )
    df["Tumor Stage"] = (
        df.groupby(["Cancer Type Detailed"])["Tumor Stage"]
        .apply(lambda x: x.fillna(x.median()))
        .sort_index(level=1)
        .values
    )
    # 3 fallback levels to fill tumor size as there exists cases where all tumor size values are null for a given cancer type detailed and tumor stage
    df["Tumor Size"] = (
        df.groupby(["Cancer Type Detailed", "Tumor Stage"])["Tumor Size"]
        .apply(lambda x: x.fillna(x.median()))
        .sort_index(level=2)
        .values
    )
    df["Tumor Size"] = (
        df.groupby(["Cancer Type Detailed"])["Tumor Size"]
        .apply(lambda x: x.fillna(x.median()))
        .sort_index(level=1)
        .values
    )
    df["Tumor Size"] = df["Tumor Size"].fillna(df["Tumor Size"].median())
    # 2 fallback levels to fill neoplasm histologic grade as there exists cases where all neoplasm histologic grade values are null for a given cancer type detailed
    df["Neoplasm Histologic Grade"] = (
        df.groupby(["Cancer Type Detailed"])["Neoplasm Histologic Grade"]
        .apply(lambda x: x.fillna(x.mode()))
        .sort_index(level=1)
        .values
    )
    df["Neoplasm Histologic Grade"] = df["Neoplasm Histologic Grade"].fillna(
        df["Neoplasm Histologic Grade"].mode()[0]
    )
    df["Tumor Other Histologic Subtype"] = (
        df.groupby(["Cancer Type Detailed"])["Tumor Other Histologic Subtype"]
        .apply(lambda x: x.fillna(x.mode()))
        .sort_index(level=1)
        .values
    )
    df["Tumor Other Histologic Subtype"] = df["Tumor Other Histologic Subtype"].fillna(
        "Ductal/NST"
    )
    df["Primary Tumor Laterality"] = (
        df.groupby(["Cancer Type Detailed"])["Primary Tumor Laterality"]
        .apply(lambda x: x.fillna(x.mode()[0]))
        .sort_index(level=1)
        .values
    )

    df["Lymph nodes examined positive"] = (
        df.groupby(["Cancer Type Detailed"])["Lymph nodes examined positive"]
        .apply(lambda x: x.fillna(x.mode()))
        .sort_index(level=1)
        .values
    )
    df["Lymph nodes examined positive"] = df["Lymph nodes examined positive"].fillna(
        df["Lymph nodes examined positive"].mode()[0]
    )
    df["Mutation Count"] = (
        df.groupby(["Cancer Type Detailed"])["Mutation Count"]
        .apply(lambda x: x.fillna(x.mode()))
        .sort_index(level=1)
        .values
    )
    df["Mutation Count"] = df["Mutation Count"].fillna(df["Mutation Count"].mode()[0])
    df["Nottingham prognostic index"] = (
        df.groupby(["Tumor Size"])["Nottingham prognostic index"]
        .apply(lambda x: x.fillna(x.median()))
        .sort_index(level=1)
        .values
    )
    df["Nottingham prognostic index"] = df["Nottingham prognostic index"].fillna(
        df["Nottingham prognostic index"].median()
    )

    # drop patient vital status as it is redundant with overall survival status
    df.drop(columns=["Patient's Vital Status", "Patient ID"], inplace=True)

    numeric_cols = df._get_numeric_data().columns
    categorical_cols = df.select_dtypes(include=["object", "bool"]).columns

    object_cols = [
        "Type of Breast Surgery",
        "Cancer Type",
        "Cancer Type Detailed",
        "Cellularity",
        "Chemotherapy",
        "Chemotherapy",
        "Pam50 + Claudin-low subtype",
        "ER status measured by IHC",
        "ER Status",
        "HER2 status measured by SNP6",
        "HER2 Status",
        "Tumor Other Histologic Subtype",
        "Hormone Therapy",
        "Inferred Menopausal State",
        "Integrative Cluster",
        "Primary Tumor Laterality",
        "Oncotree Code",
        "PR Status",
        "Radio Therapy",
        "Sex",
        "3-Gene classifier subtype",
    ]

    for col in object_cols:
        df[col] = np.uint8(LabelEncoder().fit_transform(df[col]))

    df["Overall Survival Status"] = np.uint8(
        df["Overall Survival Status"].map({"Living": 0, "Deceased": 1})
    )
    df["Relapse Free Status"] = np.uint8(
        df["Relapse Free Status"].map({"Not Recurred": 0, "Recurred": 1})
    )

    features = (
        demographic_cols
        + cancer_characteristics_cols
        + biomarker_cols
        + treatment_cols
        + clinical_metrics_cols
    )
    overall_survival_cols = ["Overall Survival (Months)", "Overall Survival Status"]
    relapse_free_cols = ["Relapse Free Status (Months)", "Relapse Free Status"]

    # Assuming 'df' is your Pandas DataFrame
    # 'features' are your independent variables, 'target' is your dependent variable
    X = df[features]
    current_label = df[overall_survival_cols]

    # Ratio: 80 - 10 - 10
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        current_label,
        test_size=0.1,
        stratify=current_label["Overall Survival Status"],
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=1 / 9,
        stratify=y_train["Overall Survival Status"],
    )

    numeric_cols = [col for col in X_train.columns if col in numeric_cols]
    categorical_cols = [col for col in X_train.columns if col in categorical_cols]

    num_features = [col for col in numeric_cols if col in features]
    num_pipe = Pipeline(
        steps=[("scale", StandardScaler())]
    )  # Set both to False for not scaling

    # Only transform numeric columns; passthrough others if you have them
    pre = ColumnTransformer(
        transformers=[("num", num_pipe, num_features)],
        remainder="passthrough",  # or "passthrough" if you also have pre-encoded categoricals
    )

    # Fit on TRAIN ONLY
    X_train = pre.fit_transform(X_train)

    # Transform
    X_test = pre.transform(X_test)
    X_val = pre.transform(X_val)

    return {
        "df": df,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "sets": {
            "train": MetabricDataset(X_train, y_train.to_numpy()),
            "test": MetabricDataset(X_test, y_test.to_numpy()),
            "val": MetabricDataset(X_val, y_val.to_numpy()),
        },
    }
