import pandas as pd
import numpy as np
import time

# Sample DataFrame
data = pd.DataFrame({
    'A': np.random.rand(1000000),
    'B': np.random.rand(1000000)
})

# Non-vectorized approach
def non_vectorized(df):
    start_time = time.time()
    df['C'] = df.apply(lambda row: row['A'] * row['B'], axis=1)
    print(f"Non-vectorized: {time.time() - start_time:.4f} sec")

# Vectorized approach
def vectorized(df):
    start_time = time.time()
    df['C'] = df['A'] * df['B']
    print(f"Vectorized: {time.time() - start_time:.4f} secs")

# Usage example
non_vectorized(data.copy())
vectorized(data.copy())