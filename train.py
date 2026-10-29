"""Train the LSTM gesture classifier on keypoint sequences saved in MP_Data/.

Usage: python train.py [--epochs 81]
Expects MP_Data/<gesture>/<sequence>/<frame>.npy, produced by gestures.ipynb.
Saves the model to model.keras and the architecture to model.json.
"""
import argparse
import os

import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from function import DATA_PATH, actions, no_sequences, sequence_length

SEED = 42
NUM_FEATURES = 21 * 3


def load_sequences():
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    missing = 0
    for gesture in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                path = os.path.join(DATA_PATH, gesture, str(sequence), f'{frame_num}.npy')
                if not os.path.exists(path):
                    missing += 1
                    break
                window.append(np.load(path))
            # Skip incomplete sequences instead of creating ragged arrays
            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[gesture])
    if missing:
        print(f'Skipped {missing} incomplete sequences')
    if not sequences:
        raise SystemExit(f'No training data found in {DATA_PATH}/. Run gestures.ipynb to collect it first.')
    return np.array(sequences), to_categorical(labels, num_classes=len(actions)).astype(int)


def build_model():
    model = Sequential([
        Input(shape=(sequence_length, NUM_FEATURES)),
        LSTM(64, return_sequences=True, activation='relu'),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(len(actions), activation='softmax'),
    ])
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--epochs', type=int, default=81)
    args = parser.parse_args()

    X, y = load_sequences()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y.argmax(axis=1))
    print(f'{len(X_train)} training and {len(X_test)} test sequences, {len(actions)} gestures')

    model = build_model()
    model.fit(X_train, y_train, epochs=args.epochs, validation_split=0.1,
              callbacks=[TensorBoard(log_dir='Logs'),
                         EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)])
    model.summary()

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print(classification_report(y_test.argmax(axis=1), y_pred,
                                labels=range(len(actions)), target_names=list(actions), zero_division=0))

    with open('model.json', 'w') as json_file:
        json_file.write(model.to_json())
    model.save('model.keras')
    print('Saved model.keras and model.json')


if __name__ == '__main__':
    main()
