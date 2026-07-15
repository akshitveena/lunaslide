import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def prepare_dataset(df, target_col='damage_grade'):
    """Preprocesses a raw hazard/damage dataframe."""
    df = df.dropna()
    if df[target_col].min() > 0:
        df[target_col] = df[target_col] - df[target_col].min()
        
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    categorical_cols = X.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
    return X, y

def split_and_balance(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Only use SMOTE if there is more than 1 class in the training set and enough samples
    if len(y_train.unique()) > 1 and len(X_train) > 10:
        sm = SMOTE(random_state=random_state, k_neighbors=min(5, len(X_train)-1))
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        return X_train_res, X_test, y_train_res, y_test
    return X_train, X_test, y_train, y_test
