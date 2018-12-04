import numpy as np
import pandas as pd


def preprocess_individual(data):
    """
    
    This function applied the various preprocessing techniques and feature transformations
    discussed in part 1.1 to the data.
    
    :param data: Original Data
    :return: Preprocessed dataset
    
    """
    def get_date(name):
        a = name.strip().split('T')[0]
        return a
    
    #Dealing with Missing Values/Incorrect Values 
    
    #Work Execution year after or before the artists birth or death
    mean_work_year = np.round(np.mean(data[data.work_execution_year != -1].work_execution_year))
    data['work_execution_year'].replace(-1,mean_work_year)
    data.loc[data.work_execution_year<data.artist_birth_year, 
             'work_execution_year' ] = mean_work_year
    data.loc[data.work_execution_year>data.artist_death_year, 
             'work_execution_year' ] = mean_work_year
    

    #replacing missing estimate high with mean(only 14 values are missing)
    mean_estimate_high = np.round(np.mean(data[data.estimate_high != -1].estimate_high))
    data['estimate_high'] = data['estimate_high'].replace(-1,mean_estimate_high)
    data['estimate_high'] = data['exchange_rate_to_usd']*data['estimate_high']
    
    mean_estimate_low = np.round(np.mean(data[data.estimate_low != -1].estimate_low))
    data['estimate_low'] = data['estimate_low'].replace(-1,mean_estimate_low)
    data['estimate_low'] = data['exchange_rate_to_usd']*data['estimate_low']
    
    
    #Get adjusted hammer price
    data['adjusted_hammer_price'] = data['exchange_rate_to_usd']*data['hammer_price']
    
    
    #Get the auction datetime in terms of datetime values
    data['auction_timestamp'] = data.auction_date.apply(get_date).astype('datetime64[D]')
    
    #Get the Difference between the high and low estimate
    #We also include 'high_estimate'
    data['high_low_diff'] = data['estimate_high'] - data['estimate_low']
    
    #is_alive_on_auction
    data['alive_at_auction'] = pd.DatetimeIndex(
        data['auction_timestamp']).year <= data['artist_death_year']
    
    #auction_month
    
    data['auction_month'] = pd.DatetimeIndex(data['auction_timestamp']).month
    data['auction_year'] = pd.DatetimeIndex(data['auction_timestamp']).year
    
    #death_auction_diff
    data['death_auction_diff'] = 0
    data.loc[data['alive_at_auction'] == False, 
             'death_auction_diff'] = pd.DatetimeIndex(data['auction_timestamp']).year - data['artist_death_year']
    
    #artist name should be included when doing other parts, this part only does picasso
    
    #categorical variables
    data =pd.get_dummies(data, columns = ['auction_department', 'auction_location', 'work_medium', 'artist_name','auction_house'], 
                         prefix = ['auction_department', 'auction_location', 
                                   'work_medium', 'artist_name', 'auction_house'], 
                         dummy_na = False)
        
    #Painting v/s non Painting
    data['is_2d'] = 0
    data.loc[data['work_depth'] == -1, 'is_2d'] = 1
    
    #Work Execution diff
    
    
    data['execution_auction_diff'] = data['auction_year'] - data['work_execution_year']  
    data['execution_birth_diff'] = data['work_execution_year']  - data['artist_birth_year']
    
    

    #Removing useless columns
    data = data.drop(columns = ['artist_birth_year', 'artist_death_year', 'auction_date',
                                'auction_timestamp','estimate_low', 'auction_currency', 
                                'artist_nationality', 'auction_timestamp', 'auction_sale_id',
                                'work_title', 'auction_currency', 'exchange_rate_to_usd', 
                                'auction_lot_count','lot_id','lot_place_in_auction',
                                'lot_description','lot_link','work_title','work_dimensions', 
                                'work_height', 'work_width', 'work_depth', 'work_measurement_unit', 
                                'buyers_premium','estimate_low','hammer_price'])
    
    
    return data
    

def percentage_diff(y_actual, y_predicted):
    
    """
    :param y_actual: Actual Labels
    :param y_predicted: Predicted Labels
    :return: Dict with percentage differences
    
    """
        
    f = np.array(abs(y_predicted - y_actual))/y_actual
    data_dict = {'10% or lesser Difference':0 , '25% - 10% Difference': 0, 
                 '50% - 25% Difference': 0, '75% - 50% Difference': 0, 
                 'Greater than or Equal to 100%': 0}
    for item in f:
        if item <= 10:
            data_dict['10% or lesser Difference'] += 1
        elif item <= 25:
            data_dict['25% - 10% Difference'] += 1
        elif item <= 50:
            data_dict['50% - 25% Difference'] += 1
        elif item <= 75:
            data_dict['75% - 50% Difference'] += 1
        else:
            data_dict['Greater than or Equal to 100%'] += 1
            
    return data_dict




def preprocess_all(data):
    """
    This function applies the various preprocessing techniques and feature transformations
    discussed in part 1.1 to the data.
    
    :param data: Pooled or Individual Data
    :return: Preprocessed dataset
    
    """
    def get_date(name):
        a = name.strip().split('T')[0]
        return a
    
    #Dealing with Missing Values/Incorrect Values 
    
    #Work Execution year after or before the artists birth or death
    
    work_median = data.loc[data['work_execution_year'] != -1].groupby('artist_name').median()['work_execution_year']
    names = list(set(data.artist_name))
    
    for i in names:
        mask = ((data.artist_name == i) & (data.work_execution_year == -1))
        data.loc[mask, 'work_execution_year'] = work_median[i]
        data.loc[(data.artist_name == i) & (data.work_execution_year<=10 + data.artist_birth_year),
               'work_execution_year'] = work_median[i]
        data.loc[(data.artist_name == i) & (data.work_execution_year>data.artist_death_year),
               'work_execution_year'] = work_median[i]
    

    #replacing missing estimate high with mean(only 26 values are missing)
    mean_estimate_high = np.round(np.mean(data[data.estimate_high != -1].estimate_high))
    data['estimate_high'] = data['estimate_high'].replace(-1,mean_estimate_high)
    data['estimate_high'] = data['exchange_rate_to_usd']*data['estimate_high']
    
    mean_estimate_low = np.round(np.mean(data[data.estimate_low != -1].estimate_low))
    data['estimate_low'] = data['estimate_low'].replace(-1,mean_estimate_low)
    data['estimate_low'] = data['exchange_rate_to_usd']*data['estimate_low']
    
    
    #Get adjusted hammer price
    data['adjusted_hammer_price'] = data['exchange_rate_to_usd']*data['hammer_price']
    
    
    #Get the auction datetime in terms of datetime values
    data['auction_timestamp'] = data.auction_date.apply(get_date).astype('datetime64[D]')
    
    #Get the Difference between the high and low estimate
    #We also include 'high_estimate'
    data['high_low_diff'] = data['estimate_high'] - data['estimate_low']
    
    #is_alive_on_auction
    data['alive_at_auction'] = pd.DatetimeIndex(
        data['auction_timestamp']).year <= data['artist_death_year']
    
    #auction_month
    
    data['auction_month'] = pd.DatetimeIndex(data['auction_timestamp']).month
    data['auction_year'] = pd.DatetimeIndex(data['auction_timestamp']).year
    
    #death_auction_diff
    data['death_auction_diff'] = 0
    mask = data['alive_at_auction'] == False
    data.loc[mask, 'death_auction_diff'] = pd.DatetimeIndex(
    data['auction_timestamp']).year[mask]- data['artist_death_year'][mask]
    
    #artist name should be included when doing other parts, this part only does picasso
    
    #categorical variables
    data =pd.get_dummies(data, columns = ['auction_department', 'auction_location', 'work_medium', 'artist_name','auction_house'], 
                         prefix = ['auction_department', 'auction_location', 
                                   'work_medium', 'artist_name', 'auction_house'], 
                         dummy_na = False)
        
    #Painting v/s non Painting
    data['is_2d'] = 0
    data.loc[data['work_depth'] == -1, 'is_2d'] = 1
    
    #Work Execution diff
    
    
    data['execution_auction_diff'] = data['auction_year'] - data['work_execution_year']  
    data['execution_birth_diff'] = data['work_execution_year']  - data['artist_birth_year']
    
    

    #Removing useless columns
    data = data.drop(columns = ['artist_birth_year', 'artist_death_year', 'auction_date',
                                'auction_timestamp','estimate_low', 'auction_currency', 
                                'artist_nationality', 'auction_timestamp', 'auction_sale_id',
                                'work_title', 'auction_currency', 'exchange_rate_to_usd', 
                                'auction_lot_count','lot_id','lot_place_in_auction',
                                'lot_description','lot_link','work_title','work_dimensions', 
                                'work_height', 'work_width', 'work_depth', 'work_measurement_unit', 
                                'buyers_premium','estimate_low','hammer_price'])
    
    
    return data














