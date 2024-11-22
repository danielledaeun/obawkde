import numpy as np
import pandas as pd
from typing import Tuple, Dict
from pathlib import Path

def get_one(df: pd.DataFrame, random_state: np.random.RandomState) -> Tuple[pd.Series, pd.DataFrame]:
    """Efficient random sampling without replacement."""
    idx = random_state.randint(df.shape[0])  # Faster than choice for single value
    return df.iloc[idx], df.drop(index=df.index[idx])  # No need for reset_index

def noise(data: pd.DataFrame, prob_noise: float, seed: int = 0) -> pd.DataFrame:
    """Generate noise with probability more efficiently."""
    # Work with numpy arrays for better performance
    df = data.copy()
    n_samples = int(df.shape[0] * prob_noise)
    
    # Use numpy for noise generation
    noise_idx = np.random.RandomState(seed).choice(
        df.index, 
        size=n_samples, 
        replace=False
    )
    
    # Vectorized operation instead of apply
    df.loc[noise_idx, 'class'] = 1 - df.loc[noise_idx, 'class']
    return df

def load_data_cache() -> Dict[str, pd.DataFrame]:
    """Cache data loading for reuse."""
    cache = {}
    data_path = Path('./data')
    
    for boundary in ['sine', 'circle', 'sea']:
        for suffix in ['', '2', '_b', '_b2']:
            file_path = data_path / f'{boundary}{suffix}.csv'
            if file_path.exists():
                cache[f'{boundary}{suffix}'] = pd.read_csv(file_path)
    
    return cache

def simulation_data(
    random_state: np.random.RandomState,
    time_steps: int,
    time_drift: int,
    boundary: str,
    ir: float,
    imbtype: str,
    data_cache: Dict[str, pd.DataFrame] = None
) -> pd.DataFrame:
    """Generate simulation data more efficiently."""
    # Use cached data if available
    if data_cache is None:
        data_cache = load_data_cache()
    
    # Get appropriate dataframes
    suffix = '_b' if imbtype == 'borderline' else ''
    df1 = data_cache[f'{boundary}{suffix}']
    df2 = data_cache[f'{boundary}{suffix}2']
    
    # Pre-split data
    df_neg = df1[df1['class'] == 0]
    df_pos = df1[df1['class'] == 1]
    df_neg2 = df2[df2['class'] == 0]
    df_pos2 = df2[df2['class'] == 1]
    
    # Pre-allocate result array
    res = []
    res_capacity = time_steps
    
    # Generate random choices for the entire sequence
    choices = random_state.rand(time_steps) > ir
    
    for t in range(time_steps):
        # Handle concept drift
        if t == time_drift:
            df_neg, df_pos = df_neg2, df_pos2
            
        # Efficient sampling
        if choices[t]:
            one, df_neg = get_one(df_neg, random_state)
        else:
            one, df_pos = get_one(df_pos, random_state)
            
        res.append(one)
    
    # Convert to DataFrame once
    result_df = pd.DataFrame(res)
    
    # Apply noise if needed
    if imbtype == 'noise':
        result_df = noise(result_df, 0.1, 0)
        
    return result_df

