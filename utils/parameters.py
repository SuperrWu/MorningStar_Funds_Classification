# BASIC_INFO_URL = "https://tools.morningstar.co.uk/api/rest.svc/roq2rh8blz/security/screener?page=1&pageSize=106610&sortOrder=LegalName asc&outputType=json&version=1&languageId=en-HK&currencyId=HKD&universeIds=FOHKG$$ALL&securityDataPoints=SecId|Name|PriceCurrency|TenforeId|LegalName|ClosePrice|Yield_M12|CategoryName|AnalystRatingScale|StarRatingM255|QuantitativeRating|SustainabilityRank|GBRReturnD1|GBRReturnW1|GBRReturnM1|GBRReturnM3|GBRReturnM6|GBRReturnM0|GBRReturnM12|GBRReturnM36|GBRReturnM60|GBRReturnM120|MaxFrontEndLoad|OngoingCharge|ManagerTenure|MaxDeferredLoad|InitialPurchase|FundTNAV|EquityStyleBox|BondStyleBox|AverageMarketCapital|AverageCreditQualityCode|EffectiveDuration|MorningstarRiskM255|AlphaM36|BetaM36|R2M36|StandardDeviationM36|SharpeM36|TrackRecordExtension&filters=&term=&subUniverseId="
BASIC_INFO_URL = "https://tools.morningstar.co.uk/api/rest.svc/roq2rh8blz/security/screener?page=1&pageSize=106610&sortOrder=LegalName%20asc&outputType=json&version=1&languageId=en-HK&currencyId=HKD&universeIds=FOHKG$$ALL&securityDataPoints=SecId|Name|geoRegion|globalAssetClassName|GlobalCategoryName|PriceCurrency|Region|Sector|TenforeId|LegalName|ClosePrice|Yield_M12|CategoryName|GeoRegionName|AnalystRatingScale|StarRatingM255|QuantitativeRating|SustainabilityRank|GBRReturnD1|GBRReturnW1|GBRReturnM1|GBRReturnM3|GBRReturnM6|GBRReturnM0|GBRReturnM12|GBRReturnM36|GBRReturnM60|GBRReturnM120|MaxFrontEndLoad|OngoingCharge|ManagerTenure|MaxDeferredLoad|InitialPurchase|FundTNAV|EquityStyleBox|BondStyleBox|AverageMarketCapital|AverageCreditQualityCode|EffectiveDuration|MorningstarRiskM255|AlphaM36|BetaM36|R2M36|StandardDeviationM36|SharpeM36|TrackRecordExtension&filters=&term=&subUniverseId="
CLS_FEATREUS = ['CategoryName', 'globalAssetClassName', 'GlobalCategoryName', 'PriceCurrency',
       'ClosePrice', 'Yield_M12', 'SustainabilityRank', 'GBRReturnD1',
       'GBRReturnW1', 'GBRReturnM1', 'GBRReturnM3', 'GBRReturnM6',
       'GBRReturnM0', 'GBRReturnM12', 'InitialPurchase', 'FundTNAV',
       'EquityStyleBox', 'AverageMarketCapital', 'QuantitativeRating',
       'GBRReturnM60', 'BondStyleBox', 'StandardDeviationM36', 'SharpeM36',
       'StarRatingM255', 'MorningstarRiskM255', 'AlphaM36', 'BetaM36', 'R2M36',
       'OngoingCharge', 'AnalystRatingScale']

OUTPUT_FEATURES = ['SecId', 'LegalName', 'CategoryName', 'globalAssetClassName', 'GlobalCategoryName', 'PriceCurrency',
       'ClosePrice', 'Yield_M12', 'SustainabilityRank', 'GBRReturnD1',
       'GBRReturnW1', 'GBRReturnM1', 'GBRReturnM3', 'GBRReturnM6',
       'GBRReturnM0', 'GBRReturnM12', 'InitialPurchase', 'FundTNAV',
       'EquityStyleBox', 'AverageMarketCapital', 'QuantitativeRating',
       'GBRReturnM60', 'BondStyleBox', 'StandardDeviationM36', 'SharpeM36',
       'StarRatingM255', 'MorningstarRiskM255', 'AlphaM36', 'BetaM36', 'R2M36',
       'OngoingCharge', 'AnalystRatingScale']