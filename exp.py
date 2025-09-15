import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    print("Error importing TensorFlow/Keras. Make sure tensorflow is installed.\n", e)
    raise


CSV_PATH = 'weatherHistory.csv'      
IMAGE_DIR = 'image'        
IMAGE_COLUMN = None                   
NROWS = 500                           
DATE_COL = 'Formatted Date'
TARGET = 'Temperature (C)'
BASE_FEATURES = ['Month', 'Day', 'DayOfYear', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']
LAG_DAYS = [1, 2, 3]                  
SEQ_FEATURES = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Pressure (millibars)']
SEQ_LEN = 7                          
TEST_SIZE = 0.2
RANDOM_STATE = 42



def load_data(csv_path, nrows=None):
    print(f"Loading CSV: {csv_path} (nrows={nrows})")
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"CSV loaded. Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def prepare_tabular_features(df, date_col=DATE_COL, target=TARGET, base_features=BASE_FEATURES, lag_days=LAG_DAYS):
    df = df.copy()
    print("Preparing date/time features...")
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    df['DayOfYear'] = df[date_col].dt.dayofyear

   
    print(f"Creating lag features: {lag_days}")
    for d in lag_days:
        df[f'Temp_t-{d}'] = df[target].shift(d)
       
        if 'Humidity' in df.columns:
            df[f'Humidity_t-{d}'] = df['Humidity'].shift(d)

    
    df.fillna(df.mean(numeric_only=True), inplace=True)

    
    df.dropna(inplace=True)
    print(f"After lag creation & dropna: Rows={len(df)}")
    return df


def extract_image_features(image_dir, df, image_column=None, target_size=(224, 224)):
    """Extract image features using ResNet50 (include_top=False, pooling='avg').
    Returns a numpy array of shape (n_rows_used, feature_dim).

    Alignment logic:
      - If image_column is not None and exists in df, use filenames from that column (absolute or relative).
      - Else: take sorted image files from image_dir and align them in order to the dataframe's first rows.
      - If an image is missing, we append a zero-vector but continue (warning printed).
    """
    print("Loading ResNet50 (this may download weights if not cached)...")
    cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_dim = cnn_model.output_shape[-1]
    print(f"ResNet50 loaded. Feature dim: {feature_dim}")

   
    if image_column and image_column in df.columns:
        filenames = df[image_column].tolist()
        print(f"Using filenames from column: {image_column}")
    else:
        all_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(all_files) == 0:
            raise ValueError(f"No image files found in {image_dir}")
        filenames = all_files[:len(df)]
        print(f"Mapping first {len(filenames)} images (sorted) from folder {image_dir} to dataframe rows.")

    features_list = []
    used_filenames = []

    for i, fname in enumerate(tqdm(filenames, desc='Extracting image features')):
       
        if os.path.isabs(fname) and os.path.exists(fname):
            fpath = fname
        else:
            fpath = os.path.join(image_dir, fname)

        if not os.path.exists(fpath):
            
            print(f"Warning: image not found for row {i}: {fpath} -> appending zeros vector")
            features_list.append(np.zeros(feature_dim, dtype=np.float32))
            used_filenames.append(None)
            continue

        try:
            img = keras_image.load_img(fpath, target_size=target_size)
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = cnn_model.predict(x, verbose=0)
            features_list.append(feat.flatten())
            used_filenames.append(fpath)
        except Exception as e:
            print(f"Failed to process image {fpath} (row {i}). Error: {e}. Appending zeros.")
            features_list.append(np.zeros(feature_dim, dtype=np.float32))
            used_filenames.append(None)

    features_arr = np.vstack(features_list)
    print(f"Extracted image features shape: {features_arr.shape}")
    return features_arr, used_filenames


def combine_tabular_and_image(df, image_features, base_features=BASE_FEATURES, lag_days=LAG_DAYS):
    df = df.copy()
  
    lag_cols = []
    for d in lag_days:
        lag_cols.append(f'Temp_t-{d}')
        if 'Humidity' in df.columns:
            lag_cols.append(f'Humidity_t-{d}')

    use_cols = [c for c in base_features if c in df.columns] + lag_cols
    print(f"Using tabular columns: {use_cols} + image features")

  
    n_rows = min(len(df), image_features.shape[0])
    df = df.iloc[:n_rows].copy()
    img_df = pd.DataFrame(image_features[:n_rows, :], index=df.index)
    img_df.columns = [f'img_feat_{i}' for i in range(img_df.shape[1])]

    X = pd.concat([df[use_cols].reset_index(drop=True), img_df.reset_index(drop=True)], axis=1)
    y = df[TARGET].reset_index(drop=True)
    print(f"Combined X shape: {X.shape}, y length: {len(y)}")
    return X, y


def train_classical_models(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    print("Training classical ML models (LinearRegression, SVR)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

   
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

   
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svr = SVR(kernel='rbf')
    svr.fit(X_train_s, y_train)
    y_pred_svr = svr.predict(X_test_s)


    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        return mae, rmse

    mae_lr, rmse_lr = metrics(y_test, y_pred_lr)
    mae_svr, rmse_svr = metrics(y_test, y_pred_svr)

    print(f'Linear Regression -> MAE: {mae_lr:.3f}, RMSE: {rmse_lr:.3f}')
    print(f'SVR -> MAE: {mae_svr:.3f}, RMSE: {rmse_svr:.3f}')

    
    nplot = min(100, len(y_test))
    plt.figure(figsize=(12,5))
    plt.plot(range(nplot), y_test.values[:nplot], label='Actual')
    plt.plot(range(nplot), y_pred_lr[:nplot], label='Pred - Linear')
    plt.plot(range(nplot), y_pred_svr[:nplot], label='Pred - SVR')
    plt.title('Classical models: Actual vs Predicted (first N samples of test set)')
    plt.xlabel('Sample index')
    plt.ylabel('Temperature (C)')
    plt.legend()
    plt.show()

    return {'lr': lr, 'svr': svr, 'scaler_for_svr': scaler}


def build_sequences_for_lstm(df, seq_features=SEQ_FEATURES, seq_len=SEQ_LEN):
    print(f"Building sequences for LSTM with seq_len={seq_len} and features={seq_features}")
    df_seq = df[seq_features].copy()
   
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(df_seq)

    temp_scaler = MinMaxScaler()
    temp_scaled = temp_scaler.fit_transform(df[[TARGET]])

    Xs = []
    ys = []
    for i in range(seq_len, len(df)):
        Xs.append(X_scaled[i-seq_len:i])  
        ys.append(temp_scaled[i, 0])      

    Xs = np.array(Xs)
    ys = np.array(ys)
    print(f"Sequences shapes -> X: {Xs.shape}, y: {ys.shape}")
    return Xs, ys, scaler_X, temp_scaler


def train_lstm(Xs, ys, epochs=40, batch_size=16):
   
    n_samples = len(Xs)
    train_size = int(n_samples * 0.8)
    X_train, X_test = Xs[:train_size], Xs[train_size:]
    y_train, y_test = ys[:train_size], ys[train_size:]

    print(f"Training LSTM on {len(X_train)} samples, validating on {len(X_test)} samples")

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    es = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=[es])

   
    y_pred = model.predict(X_test)

    return model, history, (X_train, X_test, y_train, y_test), y_pred


def inverse_scale_and_plot_lstm(y_test_scaled, y_pred_scaled, temp_scaler):
   
    y_test_inv = temp_scaler.inverse_transform(y_test_scaled.reshape(-1,1)).flatten()
    y_pred_inv = temp_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()

  
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f"LSTM -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    nplot = min(200, len(y_test_inv))
    plt.figure(figsize=(12,5))
    plt.plot(range(nplot), y_test_inv[:nplot], label='Actual')
    plt.plot(range(nplot), y_pred_inv[:nplot], label='Predicted - LSTM')
    plt.title('LSTM Forecast: Actual vs Predicted (first N samples of test set)')
    plt.xlabel('Sample index')
    plt.ylabel('Temperature (C)')
    plt.legend()
    plt.show()

    return y_test_inv, y_pred_inv


def main():
   
    if not os.path.exists(CSV_PATH):
        print(f"CSV path not found: {CSV_PATH}. Please put your weather CSV in the script folder or change CSV_PATH.")
        sys.exit(1)

    df = load_data(CSV_PATH, nrows=NROWS)

   
    df_prepped = prepare_tabular_features(df, date_col=DATE_COL, target=TARGET, base_features=BASE_FEATURES, lag_days=LAG_DAYS)

  
    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory not found: {IMAGE_DIR}. Skipping image extraction and proceeding with tabular-only models.")
        image_features = np.zeros((len(df_prepped), 1)) 
        used_filenames = [None] * len(df_prepped)
    else:
        image_features, used_filenames = extract_image_features(IMAGE_DIR, df_prepped, image_column=IMAGE_COLUMN)

   
    X_comb, y_comb = combine_tabular_and_image(df_prepped, image_features, base_features=BASE_FEATURES, lag_days=LAG_DAYS)

   
    models = train_classical_models(X_comb, y_comb)

   
    Xs, ys, scaler_X, temp_scaler = build_sequences_for_lstm(df_prepped, seq_features=SEQ_FEATURES, seq_len=SEQ_LEN)
    lstm_model, history, seq_split, y_pred_scaled = train_lstm(Xs, ys, epochs=40, batch_size=16)

   
    X_train, X_test, y_train, y_test = seq_split
    y_test_scaled = y_test
    y_pred_scaled = y_pred_scaled.flatten()
    y_test_inv, y_pred_inv = inverse_scale_and_plot_lstm(y_test_scaled, y_pred_scaled, temp_scaler)

    print("All done.\nSummary:\n - Classical models trained on tabular+image features\n - LSTM trained for sequence forecasting using past days\n")


if __name__ == '__main__':
    main()
