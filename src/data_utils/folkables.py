from folktables import ACSDataSource, BasicProblem
from folktables.acs import adult_filter
import pandas as pd
import numpy as np

CSIncomeRegression = BasicProblem(
    features=[
        "AGEP",  # age
        "SEX",   # sex
        "MAR",   # marital status
        "SCHL",  # education
        "INDP",  # industry
        "WKHP",  # hours worked per week
        "ESR",   # employment status
        "DIS",   # disability
    ],
    target="PINCP",
    target_transform=lambda x: x,
    group="RAC1P",
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

INDUSTRY_GROUPS = [
    ((170, 290), "Agriculture"),
    ((370, 490), "Mining"),
    ((570, 690), "Utilities"),
    ((770, 770), "Construction"),
    ((1070, 3990), "Manufacturing"),
    ((4070, 4590), "Wholesale"),
    ((4670, 5790), "Retail"),
    ((6070, 6390), "Transportation"),
    ((6470, 6780), "Information"),
    ((6870, 6992), "Finance"),
    ((7071, 7190), "Real estate"),
    ((7270, 7490), "Professional services"),
    ((7860, 7890), "Education"),
    ((7970, 8470), "Healthcare"),
    ((8561, 8590), "Arts & recreation"),
    ((8660, 8690), "Accommodation & food"),
    ((8770, 9290), "Other services"),
    ((9370, 9590), "Public administration"),
    ((9670, 9870), "Military"),
]

def map_industry(code):
    for (low, high), label in INDUSTRY_GROUPS:
        if low <= code <= high:
            return label
    return "Other / N.A."


def map_education(schl):
    if pd.isna(schl) or schl < 1: return np.nan
    if schl <= 11: return "Less than high school"
    if schl <= 15: return "Some high school"
    if schl <= 17: return "High school / GED"
    if schl <= 19: return "Some college"
    if schl == 20: return "Associate degree"
    if schl == 21: return "Bachelor's degree"
    if schl == 22: return "Master's degree"
    if schl == 23: return "Professional degree"
    if schl == 24: return "Doctorate"
    return np.nan

ESR_MAP = {
    1: "Employed - at work",
    2: "Employed - temporarily absent",
    3: "Unemployed",
    4: "Armed forces - at work",
    5: "Armed forces - temporarily absent",
    6: "Not in labor force",
}

DIS_MAP = {
    1: "With disability",
    2: "Without disability"
    }

SEX_MAP = {
    1: "Male",
    2: "Female"
    }

MAR_MAP = {
    1: "Married",
    2: "Widowed",
    3: "Divorced",
    4: "Separated",
    5: "Never married"
    }


def prepare_data(state, n=10_000):
    source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    data = source.get_data(states=[state], download=True)
    X, y, _ = ACSIncomeRegression.df_to_pandas(data)
    df = pd.concat([X, y], axis=1)

    df = df[df["PINCP"] < 300_000].sample(n=n, random_state=42)
    df["INDP"] = df["INDP"].apply(map_industry)
    df["SCHL"] = df["SCHL"].apply(map_education)
    df["ESR"] = df["ESR"].map(ESR_MAP)
    df["DIS"] = df["DIS"].map(DIS_MAP)
    df["SEX"] = df["SEX"].map(SEX_MAP)
    df["MAR"] = df["MAR"].map(MAR_MAP)
    return df