import pandas as pd
import numpy as np

# Load datasets
user_health = pd.read_csv("user_health_data.csv")
supplement_usage = pd.read_csv("supplement_usage.csv")
experiments = pd.read_csv("experiments.csv")
user_profiles = pd.read_csv("user_profiles.csv")

# Display first few rows of each dataset to understand the data
print("User Health Data:")
print(user_health.head(), "\n")
print(user_health.info(), "\n")

print("Supplement Usage Data:")
print(supplement_usage.head(), "\n")
print(supplement_usage.info(), "\n")

print("Experiments Data:")
print(experiments.head(), "\n")
print(experiments.info(), "\n")

print("User Profiles Data:")
print(user_profiles.head(), "\n")
print(user_profiles.info(), "\n")


def merge_all_data(user_health_file, supplement_file, experiments_file, user_profiles_file):
    # Load datasets
    #user_health = pd.read_csv(user_health_file)
   # supplement_usage = pd.read_csv(supplement_file)
   # experiments = pd.read_csv(experiments_file)
   # user_profiles = pd.read_csv(user_profiles_file)
    
    # Standardize date format
    user_health['date'] = pd.to_datetime(user_health['date'])
    supplement_usage['date'] = pd.to_datetime(supplement_usage['date'])
    
    # Convert sleep_hours to float (removing 'h' or 'H')
    user_health['sleep_hours'] = user_health['sleep_hours'].str.replace(r'[^0-9.]', '', regex=True).astype(float)
    
    # Convert dosage to grams if in mg
    supplement_usage['dosage_grams'] = supplement_usage.apply(
        lambda row: row['dosage'] / 1000 if row['dosage_unit'].lower() == 'mg' else row['dosage'], axis=1
    )
    
    # Merge supplement data with experiment details
    supplement_usage = supplement_usage.merge(experiments, on='experiment_id', how='left')
    supplement_usage.rename(columns={'name': 'experiment_name'}, inplace=True)
    
    # Merge all datasets
    merged_df = pd.merge(user_health, supplement_usage, on=['user_id', 'date'], how='outer')
    merged_df = pd.merge(merged_df, user_profiles, on='user_id', how='left')
    
    # Fill missing values
    merged_df['supplement_name'].fillna('No intake', inplace=True)
    merged_df['experiment_name'].fillna(np.nan, inplace=True)
    merged_df['dosage_grams'].fillna(np.nan, inplace=True)
    merged_df['is_placebo'].fillna(np.nan, inplace=True)
    
    # Assign age group
    def get_age_group(age):
        if pd.isna(age): return 'Unknown'
        age = int(age)
        if age < 18: return 'Under 18'
        elif age <= 25: return '18-25'
        elif age <= 35: return '26-35'
        elif age <= 45: return '36-45'
        elif age <= 55: return '46-55'
        elif age <= 65: return '56-65'
        else: return 'Over 65'
    
    merged_df['user_age_group'] = merged_df['age'].apply(get_age_group)
    
    # Select and reorder columns
    final_columns = [
        'user_id', 'date', 'email', 'user_age_group', 'experiment_name', 'supplement_name', 'dosage_grams', 
        'is_placebo', 'average_heart_rate', 'average_glucose', 'sleep_hours', 'activity_level'
    ]
    
    return merged_df[final_columns]

