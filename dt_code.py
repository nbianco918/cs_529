import pandas as pd
import numpy as np

df = pd.read_csv("test_dataset.csv")

def entropy(S:np.array) -> float:
    size_S = len(S)
    unique_vals, unique_count = np.unique(S, return_counts= True)
    #print(unique_vals, unique_count)
    probs_logs = [(val/size_S)*np.log2(val/size_S) for val in unique_count]
    E = -sum(probs_logs)
    return E

def gain(S:np.array, A:np.array) -> float:
    """
    Inputs:
        - S : Array of target values
        - A : Array of attribute values
    Returns:
        - G : Gain 
    """
    G = entropy(S)
    size_S = len(S)
    unique_vals, unique_count = np.unique(A, return_counts= True)
    for val in unique_vals:
        Sv = S[np.where(A == val)[0]]
        size_Sv = len(Sv)
        G = G - (size_Sv/size_S)*entropy(Sv)
        #print(Sv, size_Sv)
    return G


# Target
S = df['PlayTennis'].values

# Attribute
#A = df['Temperature'].values

E = entropy(S)
for col in df.columns.values.tolist()[1:-1]:
    # Attribute
    A = df[col].values
    G = gain(S,A)
    print(f"{col}: ", G)


