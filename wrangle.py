import os
import env
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split







def get_zillow_data():
    filename = "zillow.csv"

    if os.path.isfile(filename):

        return pd.read_csv(filename, index_col=0)
    else:
        # Create the url
        url = env.get_db_url('zillow')
        
        sql_query = '''
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
            taxvaluedollarcnt, yearbuilt, taxamount, fips
            FROM properties_2017
            WHERE propertylandusetypeid = 261'''

        # Read the SQL query into a dataframe
        df = pd.read_sql(sql_query, url)

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df
    
    
    
    
    
def prep_zillow(df):
    '''
    This function takes in a dataframe
    renames the columns and drops nulls values
    Additionally it changes datatypes for appropriate columns
    and renames fips to actual county names.
    Then returns a cleaned dataframe
    '''
    df = df.rename(columns = {'bedroomcnt':'bedrooms',
                     'bathroomcnt':'bathrooms',
                     'calculatedfinishedsquarefeet':'area',
                     'taxvaluedollarcnt':'property_value',
                     'fips':'county'})
    
    df = df.dropna()
    
    make_ints = ['bedrooms','area','property_value','yearbuilt']

    for col in make_ints:
        df[col] = df[col].astype(int)
        
    df.county = df.county.map({6037:'LA',6059:'Orange',6111:'Ventura'})
    
    return df




def wrangle_zillow():
    
    
    
    df=prep_zillow(get_zillow_data())
    
    
    
    return df




def split_data(df):
    
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    
    # Create train_validate and test datasets
    train, validate_test = train_test_split(df, train_size=0.60, random_state=123)
    
    # Create train and validate datsets
    validate, test = train_test_split(validate_test, test_size=0.5, random_state=123)

    # Take a look at your split datasets

    print(f"""
       train  ----> {train.shape}
    validate  ----> {validate.shape}
        test  ----> {test.shape}""")
    
    return train, validate, test
    
    
    

def wrangle_zillow_split():
    
    train,validate,test=split_data(prep_zillow(get_zillow_data()))
    
    
    return train,validate,test




#create a function to isolate the target variable AND split a DF
def X_y_split(df, target):
    '''
    This function takes in a dataframe and a target variable
    Then it returns the X_train, y_train, X_validate, y_validate, X_test, y_test
    and a print statement with the shape of the new dataframes
    '''  
    train, validate, test = split_data(df)

    X_train = train.drop(columns= target)
    y_train = train[target]

    X_validate = validate.drop(columns= target)
    y_validate = validate[target]

    X_test = test.drop(columns= target)
    y_test = test[target]
        
    # Have function print datasets shape
    print()
    print(f'X_train -> {X_train.shape}')
    print(f'y_train -> {y_train.shape}')
    print()
    print(f'X_validate -> {X_validate.shape}')
    print(f'y_validate -> {y_validate.shape}')
    print()      
    print(f'X_test -> {X_test.shape}')  
    print(f'y_validate -> {y_test.shape}')
          
    return X_train, y_train, X_validate, y_validate, X_test, y_test
    
    
    
    

    
    
    











    

    
    
    


    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
