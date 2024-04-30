from datetime import datetime as dt
from numpy import nan
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.feature_extraction.date_time import DateTimeVectorizer
from sagemaker_sklearn_extension.impute import RobustImputer
from sagemaker_sklearn_extension.preprocessing import NALabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sagemaker_sklearn_extension.preprocessing import ThresholdOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(
    column_names=[
        'Mid_Price_Future', 'Date', 'Mid_Price_Raw', 'Mid_Price_EMA_Raw',
        'Mid_Price_z-score_normalised', 'Total_Order_Volume_z-score_normalised',
        'OBV_z-score_normalised', 'Total_Volume_Imbalance_z-score_normalised',
        'Order_Imbalance_z-score_normalised',
        'Weighted_Mid_Price_z-score_normalised',
        'Bid_Ask_Spread_z-score_normalised',
        'Level_1_Bid_Price_z-score_normalised',
        'Level_1_Ask_Price_z-score_normalised',
        'Level_1_Bid_Quantity_z-score_normalised',
        'Level_1_Ask_Quantity_z-score_normalised',
        'Level_1_Order_Imbalance_z-score_normalised', 'RSI_z-score_normalised',
        'Stochastic_RSI_z-score_normalised',
        'Awesome_Oscillator_z-score_normalised',
        'Accelerator_Oscillator_z-score_normalised', 'MACD_z-score_normalised',
        'MACD_Signal_z-score_normalised', 'Hull_MA_z-score_normalised',
        'Keltner_Channel_Middle_z-score_normalised',
        'Keltner_Channel_Upper_z-score_normalised',
        'Keltner_Channel_Lower_z-score_normalised', 'SMA_20_z-score_normalised',
        'DPO_z-score_normalised', 'Upper_BB_z-score_normalised',
        'Lower_BB_z-score_normalised', 'Log_Returns_z-score_normalised',
        'Realised_Semi_Variance_z-score_normalised',
        'Squared_Log_Returns_z-score_normalised',
        'Realised_Volatility_z-score_normalised',
        'Abs_Log_Returns_z-score_normalised',
        'Realised_Bipower_Variation_z-score_normalised',
        'Total_Quadratic_Variation_z-score_normalised',
        'Jump_Variation_z-score_normalised',
        'Smoothed_Mid_Price_z-score_normalised',
        'Mid_Price_EMA_Short_z-score_normalised', 'Mid_Price_Past_1',
        'Mid_Price_Past_2', 'Mid_Price_Past_3', 'Mid_Price_Past_4',
        'Mid_Price_Past_5'
    ],
    target_column_name='Mid_Price_Future'
)


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features can be parsed as numeric.

    numeric = HEADER.as_feature_indices(
        [
            'Mid_Price_Raw', 'Mid_Price_EMA_Raw',
            'Mid_Price_z-score_normalised',
            'Total_Order_Volume_z-score_normalised', 'OBV_z-score_normalised',
            'Total_Volume_Imbalance_z-score_normalised',
            'Order_Imbalance_z-score_normalised',
            'Weighted_Mid_Price_z-score_normalised',
            'Bid_Ask_Spread_z-score_normalised',
            'Level_1_Bid_Price_z-score_normalised',
            'Level_1_Ask_Price_z-score_normalised',
            'Level_1_Bid_Quantity_z-score_normalised',
            'Level_1_Ask_Quantity_z-score_normalised',
            'Level_1_Order_Imbalance_z-score_normalised',
            'RSI_z-score_normalised', 'Stochastic_RSI_z-score_normalised',
            'Awesome_Oscillator_z-score_normalised',
            'Accelerator_Oscillator_z-score_normalised',
            'MACD_z-score_normalised', 'MACD_Signal_z-score_normalised',
            'Hull_MA_z-score_normalised',
            'Keltner_Channel_Middle_z-score_normalised',
            'Keltner_Channel_Upper_z-score_normalised',
            'Keltner_Channel_Lower_z-score_normalised',
            'SMA_20_z-score_normalised', 'DPO_z-score_normalised',
            'Upper_BB_z-score_normalised', 'Lower_BB_z-score_normalised',
            'Log_Returns_z-score_normalised',
            'Realised_Semi_Variance_z-score_normalised',
            'Squared_Log_Returns_z-score_normalised',
            'Realised_Volatility_z-score_normalised',
            'Abs_Log_Returns_z-score_normalised',
            'Realised_Bipower_Variation_z-score_normalised',
            'Total_Quadratic_Variation_z-score_normalised',
            'Jump_Variation_z-score_normalised',
            'Smoothed_Mid_Price_z-score_normalised',
            'Mid_Price_EMA_Short_z-score_normalised', 'Mid_Price_Past_1',
            'Mid_Price_Past_2', 'Mid_Price_Past_3', 'Mid_Price_Past_4',
            'Mid_Price_Past_5'
        ]
    )

    # These features contain a relatively small number of unique items.

    categorical = HEADER.as_feature_indices(
        [
            'Mid_Price_Raw', 'Mid_Price_z-score_normalised',
            'Total_Order_Volume_z-score_normalised', 'OBV_z-score_normalised',
            'Total_Volume_Imbalance_z-score_normalised',
            'Order_Imbalance_z-score_normalised',
            'Bid_Ask_Spread_z-score_normalised',
            'Level_1_Bid_Price_z-score_normalised',
            'Level_1_Ask_Price_z-score_normalised',
            'Level_1_Bid_Quantity_z-score_normalised',
            'Level_1_Ask_Quantity_z-score_normalised',
            'Level_1_Order_Imbalance_z-score_normalised',
            'RSI_z-score_normalised', 'Stochastic_RSI_z-score_normalised',
            'Awesome_Oscillator_z-score_normalised',
            'SMA_20_z-score_normalised', 'DPO_z-score_normalised',
            'Log_Returns_z-score_normalised',
            'Realised_Semi_Variance_z-score_normalised',
            'Squared_Log_Returns_z-score_normalised',
            'Abs_Log_Returns_z-score_normalised',
            'Realised_Bipower_Variation_z-score_normalised',
            'Smoothed_Mid_Price_z-score_normalised'
        ]
    )

    # These features can be parsed as date or time.

    datetime = HEADER.as_feature_indices(['Date'])

    numeric_processors = Pipeline(
        steps=[
            (
                'robustimputer',
                RobustImputer(strategy='constant', fill_values=nan)
            )
        ]
    )

    categorical_processors = Pipeline(
        steps=[
            ('thresholdonehotencoder', ThresholdOneHotEncoder(threshold=2000))
        ]
    )

    datetime_processors = Pipeline(
        steps=[
            (
                'datetimevectorizer',
                DateTimeVectorizer(
                    mode='ordinal',
                    default_datetime=dt(year=1970, month=1, day=1)
                )
            )
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric_processing', numeric_processors, numeric
            ), ('categorical_processing', categorical_processors, categorical
               ), ('datetime_processing', datetime_processors, datetime)
        ]
    )

    return Pipeline(
        steps=[
            ('column_transformer', column_transformer
            ), ('robuststandardscaler', RobustStandardScaler())
        ]
    )


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return NALabelEncoder()
