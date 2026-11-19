import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from preprocessing import prepare_datasets

def build_baseline_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(6, activation='softmax')
    ], name="Baseline_MLP")
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_bilstm(filters=128, dropout=0.4, lr=0.001):
    model = models.Sequential([
        layers.Input(shape=(128, 9)),
        layers.Conv1D(filters, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(dropout),
        layers.Bidirectional(layers.LSTM(64, return_sequences=False)),
        layers.Dropout(dropout),
        layers.Dense(64, activation='relu'),
        layers.Dense(6, activation='softmax')
    ], name="CNN_BiLSTM_Best")
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    tf.random.set_seed(42)
    (X_tr_raw, X_val_raw, _, X_tr_feat, X_val_feat, _, y_tr, y_val, _, raw_scaler) = prepare_datasets()

    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5)

    best_model = build_cnn_bilstm(filters=128, dropout=0.4, lr=0.001)
    best_model.fit(X_tr_raw, y_tr, validation_data=(X_val_raw, y_val),
                   epochs=40, batch_size=64, callbacks=[es, rlr], verbose=1)

    best_model.save("har_cnn_bilstm_best.keras")
    joblib.dump(raw_scaler, "har_raw_scaler.pkl")
    print("Mentés kész: har_cnn_bilstm_best.keras és har_raw_scaler.pkl")
