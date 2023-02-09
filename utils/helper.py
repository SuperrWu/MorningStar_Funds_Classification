from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.utils import shuffle as reset
from tqdm import tqdm


def train_valid_split(data, test_size=0.3, shuffle=True, random_state=None):
    if shuffle:
        data = reset(data, random_state=random_state)
	
    train = data[int(len(data)*test_size):].reset_index(drop = True)
    val  = data[:int(len(data)*test_size)].reset_index(drop = True)
    
    return train, val

def missing_values(df):
    return ((df.isnull().sum())/df.shape[0]).sort_values(ascending=False).map(lambda x:"{:.2%}".format(x))

def AnalystRatingScaleValue(x):
  if x == "1":
    return "Negative"
  elif x == "2":
    return "Neutral"
  elif x == "3":
    return "Bronze"
  elif x == "4":
    return "Sliver"
  elif x == "5":
    return "Gold"
  elif x == "6":
    return "Under Review"
  else:
    return x

def QuantitativeRatingValue(x):
  if x == float(1):
    return "Negative"
  elif x == float(2):
    return "Neutral"
  elif x == float(3):
    return "Bronze"
  elif x == float(4):
    return "Sliver"
  elif x == float(5):
    return "Gold"
  else:
    return x

def clean_df(df):
  df.AnalystRatingScale = df.AnalystRatingScale.apply(AnalystRatingScaleValue)
  df.QuantitativeRating = df.QuantitativeRating.apply(QuantitativeRatingValue)
  return df

def unique_name(x):
    feq_name_split = ["CNH", "RMB", "AUS", "EUR", "CAD", "HKD", "NZD", "SGD", "GBP"]
    split_name = set(x.split(" ")) - set(feq_name_split)
    return split_name

def get_similar_funds_dict(df):
    similar_funds_dict = {}
    feq_name_split = ["CNH", "RMB", "AUS", "EUR", "CAD", "HKD", "NZD", "SGD", "GBP"]
    search_bin = 10
    for indexi, item in tqdm(enumerate(df.LegalName.to_list())):
        similar_funds_dict[item] = []
        split_name = unique_name(item)
        search_lower_bound = max(0, indexi - search_bin)
        search_higher_dound = min(indexi + search_bin, len(df))
        for j in df.LegalName.to_list()[search_lower_bound : search_higher_dound]:
            if item != j:
                compare_name = unique_name(j)
                if len(split_name - (split_name - unique_name(j))) / len(split_name) >= 0.7:
                    similar_funds_dict[item].append(j)
    return similar_funds_dict

def predict_missing_return_rfr(df, feature):
    features = ['GBRReturnD1', 'GBRReturnW1', 'GBRReturnM1', 'GBRReturnM0'] + [feature]
    GBRReturn_df = df[features]
    known_Return = GBRReturn_df[GBRReturn_df[feature].notnull()].values
    unknown_Return = GBRReturn_df[GBRReturn_df[feature].isnull()].values
    y = known_Return[:,2]
    x = known_Return[:,:2]
    rfr = RandomForestRegressor(random_state=0,n_estimators=200,n_jobs=-1)
    rfr.fit(x,y)
    predict_Return = rfr.predict(unknown_Return[:,:2])
    df.loc[(df[feature].isnull()), feature] = predict_Return
    return df

def replace_missing_mean(df, reference_feature, feature):
    index = df[df[feature].isnull()].index
    dict = df.groupby([reference_feature]).mean().to_dict()[feature]
    for i in index:
        df.loc[i, feature] = dict[(df.loc[i][reference_feature])]
    return df
    
def replace_missing_similar_funds(df, continuous_feature, similar_dict, type):
    assert type in ["mean", "maxtimes"], "Your type should in [mean, maxtimes]"
    bool_df = df.isnull()
    for index, row in df.iterrows():
        for j in row.keys():
            if bool_df.loc[index, j] and j in continuous_feature:
                temp_similar_copmany = similar_dict[row.LegalName]
                temp_similar_df = df[df.LegalName.isin(temp_similar_copmany)]
                new_df = temp_similar_df[temp_similar_df[j].notnull()]
                if len(new_df) == 0:
                    pass
                else:
                    if type == "mean":
                        df.loc[index, j] = temp_similar_df[j].mean()
                    if type == "maxtimes":
                        df.loc[index, j] = temp_similar_df[j].value_counts(ascending=False).keys()[0]
    return df

def get_dummy(df, feature):
    dummies_feature = pd.get_dummies(df[feature], prefix= feature)
    df = pd.concat([df, dummies_feature], axis=1)
    df.drop([feature], axis=1, inplace=True)
    return df
  
def val_model(predict_y, val_Y):
    correct = 0
    for i in zip(predict_y, val_Y):
      if i[0] == i[1]:
        correct += 1
    return correct / len(predict_y)