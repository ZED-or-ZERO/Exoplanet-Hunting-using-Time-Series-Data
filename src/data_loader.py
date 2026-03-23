import pandas as pd

def load_and_preprocess_raw_data(flux_path, time_path, features_path):
    """
    Downloads raw data, cleans up technical debris, 
    synchronizes arrays and fills in gaps.
    """
    print("Loading datasets...")
    flux_df = pd.read_csv(flux_path)
    time_df = pd.read_csv(time_path)
    features_df = pd.read_csv(features_path)

    print("Dropping garbage columns...")
    flux_df = flux_df.drop(columns=[col for col in flux_df.columns if 'Unnamed' in col], errors='ignore')
    time_df = time_df.drop(columns=[col for col in time_df.columns if 'Unnamed' in col], errors='ignore')
    features_df = features_df.drop(columns=[col for col in features_df.columns if 'Unnamed' in col], errors='ignore')

    print("Aligning rows and columns...")
    valid_ids = features_df['id']
    flux_df = flux_df[flux_df['id'].isin(valid_ids)].copy()
    time_df = time_df[time_df['id'].isin(valid_ids)].copy()

    flux_df.reset_index(drop=True, inplace=True)
    time_df.reset_index(drop=True, inplace=True)
    features_df.reset_index(drop=True, inplace=True)

    # Trim off the extra columns of the stream to match the time
    cols_to_drop_flux = ['flux_4609', 'flux_4610']
    flux_df.drop(columns=cols_to_drop_flux, inplace=True, errors='ignore')

    print("Interpolating missing values (NaNs)...")
    flux_df = flux_df.interpolate(method='linear', axis=1, limit_direction='both')
    time_df = time_df.interpolate(method='linear', axis=1, limit_direction='both')
    
    print(f"Final shapes -> Flux: {flux_df.shape}, Time: {time_df.shape}, Features: {features_df.shape}")
    
    return flux_df, time_df, features_df

if __name__ == "__main__":
    # Local script test
    f_df, t_df, feat_df = load_and_preprocess_raw_data(
        "../data/raw/Astro_Flux_data.csv",
        "../data/raw/Astro_Time_data.csv",
        "../data/raw/features_dataset.csv"
    )