import pandas as pd
from utils.helper import *
import argparse
from utils.parameters import BASIC_INFO_URL, CLS_FEATREUS, OUTPUT_FEATURES
import json
import requests
import math
from sklearn.ensemble import RandomForestClassifier

def scrap_basic_info():
    res = requests.get(BASIC_INFO_URL)
    data = json.loads(res.content)
    df = pd.DataFrame(data["rows"])
    df = clean_df(df)
    df.to_csv("YOUR_FILENAME_HERE.csv", header = True, index = False, encoding="utf-8-sig")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='federated_learning')
    parser.add_argument('-reload', type=bool, default=False)
    args = parser.parse_args()
    
    if args.reload:
        scrap_basic_info()
        
    # load basic info data
    df = pd.read_csv("YOUR_FILENAME_HERE.csv")   
    iof = ['SecId', 'Name', 'globalAssetClassName', 'GlobalCategoryName','LegalName','CategoryName', 'PriceCurrency', 'ClosePrice', 'Yield_M12', 'SustainabilityRank', 'GBRReturnD1', 'GBRReturnW1', 'GBRReturnM1', 'GBRReturnM3', 'GBRReturnM6', 'GBRReturnM0', 'GBRReturnM12', 'InitialPurchase', 'FundTNAV', 'EquityStyleBox', 'AverageMarketCapital', 'EffectiveDuration', 'QuantitativeRating', 'GBRReturnM60', 'BondStyleBox', 'StandardDeviationM36', 'SharpeM36', 'StarRatingM255', 'MorningstarRiskM255', 'AlphaM36', 'BetaM36', 'R2M36', 'OngoingCharge', 'AverageCreditQualityCode', 'AnalystRatingScale']
    df = df[iof]
    
    df.loc[df[df.PriceCurrency.isnull()].index[0], "PriceCurrency"] = "USD"
    df.loc[(df.ClosePrice.isnull()), 'ClosePrice'] = df[df.Name.str.contains("JPMorgan SAR European") & df.PriceCurrency.notnull()].ClosePrice.mean()
    
    # fill in missing values with similar funds' name
    similar_dict = get_similar_funds_dict(df)
    continuous_feature = ['SharpeM36', 'StandardDeviationM36', 'OngoingCharge', 'FundTNAV', 'Yield_M12', 'GBRReturnM12', 'GBRReturnM6', 'GBRReturnM3', 'GBRReturnM0', 'GBRReturnM1', 'GBRReturnW1', 'GBRReturnD1', 'GBRReturnM60', 'R2M36', 'AlphaM36', 'BetaM36', 'AverageMarketCapital', 'InitialPurchase']             
    df = replace_missing_similar_funds(df, continuous_feature, similar_dict, "mean")
    category_features = ['SustainabilityRank', 'EquityStyleBox', 'QuantitativeRating', 'AnalystRatingScale', 'BondStyleBox']
    df = replace_missing_similar_funds(df, category_features, similar_dict, "maxtimes")
    
    missing_features = ["GBRReturnD1", "GBRReturnW1", "GBRReturnM1", "GBRReturnM0"]
    for missing_feature in missing_features:
        df = replace_missing_mean(df, "globalAssetClassName", missing_feature)
        df = replace_missing_mean(df, "GlobalCategoryName", missing_feature)
        
    missing_features = ['GBRReturnM12', 'GBRReturnM6', 'GBRReturnM3', 'GBRReturnM60']                
    for feature in missing_features:
        df = predict_missing_return_rfr(df, feature)
        
    missing_features = ["Yield_M12", "FundTNAV", 'InitialPurchase', 'StandardDeviationM36', 'SharpeM36', 'AlphaM36', 'BetaM36', 'R2M36', 'AverageMarketCapital', 'OngoingCharge']
    for missing_feature in missing_features:
        df = replace_missing_mean(df, "globalAssetClassName", missing_feature)
        df = replace_missing_mean(df, "GlobalCategoryName", missing_feature)
        
    df.EquityStyleBox = df.EquityStyleBox.fillna(0)
    df.SustainabilityRank = df.SustainabilityRank.fillna(0)
    df.QuantitativeRating = df.QuantitativeRating.fillna("Unknown")
    df.AnalystRatingScale = df.AnalystRatingScale.fillna("unknown")
    df.BondStyleBox = df.BondStyleBox.fillna("unknown")
    
    df.drop(["EffectiveDuration", "AverageCreditQualityCode"], axis=1, inplace=True)
    df.loc[df.OngoingCharge.isnull().index, "OngoingCharge"] = df[df.OngoingCharge.notnull()].OngoingCharge.mean()
    
    # make a random forest classficiation 
    cls_df = df[CLS_FEATREUS]
    category_features = ['PriceCurrency', 'globalAssetClassName', 'GlobalCategoryName', 'CategoryName', 'SustainabilityRank', 'EquityStyleBox', "QuantitativeRating", "BondStyleBox", "AnalystRatingScale"]
    for feature in category_features:
        cls_df = get_dummy(cls_df, feature)
        
    # train, val, test split 
    cls_df_known = cls_df[cls_df.MorningstarRiskM255.notnull()]
    cls_df_unknown = cls_df[cls_df.MorningstarRiskM255.isnull()]
    train_df, val_df = train_valid_split(cls_df_known, test_size=0.2, shuffle=True, random_state=42)
    test_df = cls_df_unknown
    
    # all features for training
    train_features = train_df.columns.to_list()
    train_features.remove("MorningstarRiskM255")
    train_features.remove("StarRatingM255")
    train_X = train_df[train_features].values
    train_Y = train_df[["MorningstarRiskM255", "StarRatingM255"]].values
    rfr = RandomForestClassifier(random_state=0,n_estimators=200,n_jobs=-1)
    rfr.fit(train_X, train_Y)
    
    # validation
    val_X = val_df[train_features].values
    val_Y = val_df[["MorningstarRiskM255", "StarRatingM255"]].values
    predict_y = rfr.predict(val_X)
    print(f"Model predicting MorningstarRiskM255 accuracy is about {val_model(predict_y[:, 0], val_Y[:, 0])}.")
    print(f"Model predicting StarRatingM255 accuracy is about {val_model(predict_y[:,1], val_Y[:, 1])}")
    
    test_X = test_df[train_features].values
    predict_y = rfr.predict(test_X)
    
    df.loc[(df["MorningstarRiskM255"].isnull()), "MorningstarRiskM255"] = predict_y[:, 0]
    df.loc[(df["StarRatingM255"].isnull()), "StarRatingM255"] = predict_y[:, 1]
    df[['SecId', 'LegalName', 'CategoryName', 'globalAssetClassName', 'GlobalCategoryName', 'PriceCurrency', 'StarRatingM255', 'MorningstarRiskM255']].to_csv("Results.csv", header = True, index = False, encoding="utf-8-sig")
    df[OUTPUT_FEATURES].to_csv("clean_funds.csv", header = True, index = False, encoding="utf-8-sig")