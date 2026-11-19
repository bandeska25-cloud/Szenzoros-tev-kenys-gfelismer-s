import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score
from preprocessing import prepare_datasets, ACTIVITY_LABELS

if __name__ == "__main__":
    (_, _, X_te_raw, _, _, _, _, _, y_te, _) = prepare_datasets()
    model = tf.keras.models.load_model("har_cnn_bilstm_best.keras")

    preds = np.argmax(model.predict(X_te_raw, verbose=0), axis=1)
    print(f"Test Accuracy: {accuracy_score(y_te, preds):.4f}")
    print(f"Test Macro F1: {f1_score(y_te, preds, average='macro'):.4f}")
    print(classification_report(y_te, preds, target_names=list(ACTIVITY_LABELS.values()), digits=4))
