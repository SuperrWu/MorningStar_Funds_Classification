# Goal
> This is a supervised machine learning project where in the goal is to predict the Mutual Funds rating as well as risk rating given by Morningstar based on the features captured from website. The dataset is from  [morningstar] website (https://www.morningstar.com/) with a total number of 10,652 mutual funds. Random Forest models are selected to use in this project due to the strong explanation ability.
# Data scraping
Human search involves manually browsing websites, reading articles, and gathering information through manual research. This method is time-consuming and often relies on an individual's knowledge and ability to find relevant information.

Data scraping, on the other hand, involves using automated tools to extract specific data from websites and APIs. This method is much faster and can gather large amounts of data in a short amount of time. Data scraping algorithms can also be programmed to extract specific types of information, making it a more efficient way to gather specific data.
## Scrap current funds' basic information
```python
URL = ""
import requests
res = requests.get(URL)
data = json.loads(res.content)
df = pd.DataFrame(data["rows"])
```
## Scrap extra and detailed information such as holdings, previous performances, etc.
### Get token
```python
import bs4
import re
def get_token(SecId):
    url = f"https://www.morningstar.hk/hk/report/fund/portfolio.aspx?t={SecId}&fundservcode=&lang=zh-HK"
    res = requests.get(url)
    # match token
    token = re.findall("maas_token=.*content", str(bs4.BeautifulSoup(res.content.decode("utf-8")).find(name="sal-components")))[0][12:-13]
    return token
```
### Run
```python
from tqdm import tqdm
def get_extra_info(df):
  sum_df = pd.DataFrame()
  # Iterate the SecId in the basic info dataframe to scrap different funds' extra info
  for SecId in tqdm(df.SecId.to_list()):
    # get token first
    token = get_token(SecId)
    token = f"Bearer {token}"
    # define header
    headers = {"authorization": token}
    # implement your own function here
    AssetAlloc_df = get_df(SecId, "AssetAllocationInfo", headers = headers)
    CreditQuality_df = get_df(SecId, "CreditQualityBonds", headers = headers)
    FixIncome_df = get_df(SecId, "FixIncomeAllocationInfo", headers = headers)
    EquityAlloc_df = get_df(SecId, "EquityAllocationInfo", headers = headers)
    Measure_df = get_df(SecId, "StockMeasure", headers = headers)
    Holding_values_df = values_split(SecId, headers)
    Holding_name_df = name_split(SecId, headers)
    Holding_weight_df = weight_split(SecId, headers)
    all_df = pd.concat([AssetAlloc_df, CreditQuality_df, FixIncome_df, EquityAlloc_df, Holding_values_df, Holding_name_df, Holding_weight_df, Measure_df], axis=1)
    all_df["SecId"] = SecId
    sum_df = pd.concat([sum_df, all_df], axis=0)
  return sum_df
```
## Dataset for model training
Our experiments are conducted based on the raw data in the acquisition of information on 10,652 mutual funds, which possess 21 distinct attributes, on Morningstar website.The raw data from the website, along with its name, description, and missing proportion, has been illustrated in Table below for reference.
| Attributes        | Type          | Missing rate |
| ----------------- | ------------ | -------------------- |
| AnalystRatingScale| category     | 75.94%               |
| BondStyleBox      | category     | 74.33%               |
| EffectiveDuration | float        | 70.78%               |
| AverageCreditQualityCode | category | 69.79%               |
| QuantitativeRating | category     | 60.52%               |
| EmailStarRiskM255 | category     | 39.56%               |
| StarRatingM255    | category     | 39.56%               |
| AverageMarketCapital | float      | 35.78%               |
| InitialPurchase   | float        | 33.88%               |
| EquityStyleBox    | category     | 32.14%               |
| GBRReturnM60      | float        | 27.68%               |
| R2M36             | float        | 23.56%               |
| AlphaM36          | float        | 23.54%               |
| BetaM36           | float        | 23.54%               |
| SustainabilityRank| category     | 17.85%               |
| SharpeM36         | float        | 17.51%               |
| StandardDeviationM36 | float    | 17.18%               |
| OngoingCharge     | float        | 16.74%               |
| FundTNAV          | float        | 3.33%                |
| Yield_M12         | float        | 3.13%                |
| GBRReturnM12      | float        | 2.86%                |
| GBRReturnM6       | float        | 1.28%                |
| GBRReturnM3       | float        | 0.50%                |
| GBRReturnM0       | float        | 0.33%                |
| GBRReturnM1       | float        | 0.16%                |
| GBRReturnW1       | float        | 0.06%                |
| GBRReturnD1       | float        | 0.06%                |
| ClosePrice        | float        | 0.01%                |
| PriceCurrency     | category     | 0.01%                |
| CategoryName      | category     | 0.00%                |
| LegalName         | str          | 0.00%                |
| Name              | str          | 0.00%                |
| SecId             | str          | 0.00%                |
| Total Number      | 10652 rows (funds) | |



# Data Preprocessing Pipeline
 However, the obtained data was first needed to be preprocessed, as it was marred by numerous missing values. To ensure the data's accuracy, we must undertake a thorough data preprocessing pipeline including cleaning, fixing and etc. 
## Filling in Missing Values
> Some funds share very similar names, resulting from their sale in various locations or currencies but with similar properties. This presents possible tricks for data preprocessing.
* Filling the missing values of PriceCurrency and ClosePrice, we will utilize data from similar sources/funds name. There is only 1 record having missing values. Firstly, we will fill in the currency according to the name of the fund. Then, we will take the average value of PriceCurrency from funds with the similar fund names to use as the filling value.
```python
df.loc[df[df.PriceCurrency.isnull()].index[0], "PriceCurrency"] = "USD"
df.loc[(df.ClosePrice.isnull()), 'ClosePrice'] = df[df.Name.str.contains("JPMorgan SAR European") & df.PriceCurrency.notnull()].ClosePrice.mean()
```
* Filling in missing values in all features by utilizing the mean and the item that appears most frequently among funds with similar LegalNames. Float type cleaning as example:
```python
import math
from tqdm import tqdm

def unique_name(x):
    # subtract frequent words in names for more accurate matching
    feq_name_split = ["CNH", "RMB", "AUS", "EUR", "CAD", "HKD", "NZD", "SGD", "GBP"]
    # get unique name list
    split_name = set(x.split(" ")) - set(feq_name_split)
    return split_name

def get_similar_funds_dict(df):
    similar_funds_dict = {}
    feq_name_split = ["CNH", "RMB", "AUS", "EUR", "CAD", "HKD", "NZD", "SGD", "GBP"]
    # search for 20 nearby rows for matching imilar legalnames
    search_bin = 10
    for indexi, item in tqdm(enumerate(df.LegalName.to_list())):
        similar_funds_dict[item] = []
        split_name = unique_name(item)
        # boundary detection
        search_lower_bound = max(0, indexi - search_bin)
        search_higher_dound = min(indexi + search_bin, len(df))
        # iterize all funds legal name
        for j in df.LegalName.to_list()[search_lower_bound : search_higher_dound]:
            if item != j:
                compare_name = unique_name(j)
                # Define as similar names if the similarity proportion >= 7
                if len(split_name - (split_name - unique_name(j))) / len(split_name) >= 0.7:
                    similar_funds_dict[item].append(j)
    return similar_funds_dict

# define missing values position
bool_df = df.isnull()
similiar_dict = get_similar_funds_dict(df)

for index, row in df.iterrows():
    for j in row.keys():
        if bool_df.loc[index, j] and j in continue_feature:
            temp_similar_copmany = similiar_dict[row.LegalName]
            temp_similar_df = df[df.LegalName.isin(temp_similar_copmany)]
            new_df = temp_similar_df[temp_similar_df[j].notnull()]
            # if no similar funds, pass it and leave it missing
            if len(new_df) == 0:
                pass
            # else filling the missing with mean of other similar funds
            else:
                df.loc[index, j] = temp_similar_df[j].mean()

```
* Drop unfixalbe records and use GBRReturnD1, W1, M1, M0 to predict those with missing values in ['GBRReturnM12', 'GBRReturnM6', 'GBRReturnM3', 'GBRReturnM60'] by using random forest.
```python
from sklearn.ensemble import RandomForestRegressor
def set_missing_return(feature):
    features = ['GBRReturnD1', 'GBRReturnW1', 'GBRReturnM1', 'GBRReturnM0'] + [feature]
    GBRReturn_df = df[features]
    known_Return = GBRReturn_df[GBRReturn_df[feature].notnull()].values
    unknown_Return = GBRReturn_df[GBRReturn_df[feature].isnull()].values
    y = known_Return[:,2]
    x = known_Return[:,:2]
    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(x,y)
    predict_Return = rfr.predict(unknown_Return[:,:2])
    df.loc[(df[feature].isnull()), feature] = predict_Return
    return df

missing_features = ['GBRReturnM12', 'GBRReturnM6', 'GBRReturnM3', 'GBRReturnM60']                
for feature in missing_features:
    df = set_missing_return(feature)
```
## Turning category variable to dummy variable
By converting categorical data into numerical data in the form of dummy variables, you can use the categorical data in your analysis. This is also known as one-hot encoding, where each category is represented by a binary column in the data. This allows you to use categorical variables as inputs for your model without having to worry about their non-numeric nature.
```python
def get_dummy(df, feature):
    dummies_feature = pd.get_dummies(df[feature], prefix= feature)
    df = pd.concat([df, dummies_feature], axis=1)
    df.drop([feature], axis=1, inplace=True)
    return df

cate_features = ['PriceCurrency', 'main_category', 'CategoryName', 'SustainabilityRank', 'EquityStyleBox', "QuantitativeRating", "BondStyleBox", "AnalystRatingScale"]

for feature in cate_features:
    cls_df = get_dummy(cls_df, feature)
```
## Train & Valid split (Predicting Risk as examples)
```python
import numpy as np
import pandas as pd
from sklearn.utils import shuffle as reset


def train_valid_split(data, test_size=0.3, shuffle=True, random_state=None):
    if shuffle:
        data = reset(data, random_state=random_state)
	
    train = data[int(len(data)*test_size):].reset_index(drop = True)
    val  = data[:int(len(data)*test_size)].reset_index(drop = True)
    
    return train, val

cls_df_known = cls_df[cls_df.MorningstarRiskM255.notnull()]
cls_df_unknown = cls_df[cls_df.MorningstarRiskM255.isnull()]
train_df, val_df = train_valid_split(cls_df_known, test_size=0.2, shuffle=True, random_state=42)
```
# Train a random forest classifier
Tree-based machine learning models are selected for the project because Tree-based models divide the feature space into a set of rectangular regions, with each internal node of the tree representing a test on a single feature, and each leaf node representing a prediction. This structure allows us to easily see how each feature contributes to the final prediction, and how the interactions between features affect the outcome. 
## Train
```python
from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(random_state=0,n_estimators=2000,n_jobs=-1)
rfr.fit(train_X, train_Y)
```
## Test
```python
predict_y = rfr.predict(val_X)
correct = 0
for i in zip(predict_y, val_Y):
    if i[0] == i[1]:
        correct += 1
```
# Results
