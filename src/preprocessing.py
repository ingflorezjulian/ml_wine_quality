import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(url):
    """Carga datos desde URL"""
    df = pd.read_csv(url, sep=';')
    print(f"Dataset cargado: {df.shape}")
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """Preprocesamiento completo de datos"""
    df = df.dropna()

    # Separar features y target
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Convertir a clasificación binaria (bueno/malo)
    y = (y >= 6).astype(int)

    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Escalamiento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
