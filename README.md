# Real-Time Sign Language Detection

Recognizes sign language gestures from a live webcam feed. MediaPipe tracks 21 hand landmarks per frame, and a custom TensorFlow/Keras LSTM trained on my own gesture dataset turns them into words shown on screen in real time.

## Demo

![Live gesture prediction](https://github.com/user-attachments/assets/effc8acb-c01e-417a-a6b8-2d8ac6157ff2)
![Live gesture prediction](https://github.com/user-attachments/assets/0aff47be-c233-4270-be17-544e1f2b9602)
![Live gesture prediction](https://github.com/user-attachments/assets/4a730737-329b-4db7-8b06-5cfe0e7689b6)
![Live gesture prediction](https://github.com/user-attachments/assets/0a53be3a-1489-4379-8004-60046269c3d0)
![Live gesture prediction](https://github.com/user-attachments/assets/c28202ba-6542-4f4c-8b69-86efd75337f6)

## How it works

1. **Collect data** (`gestures.ipynb`): capture webcam images with keyboard controls and save them into one folder per gesture.
2. **Extract features** (`function.py`): MediaPipe Hands finds the hand in each frame and returns 21 landmarks × (x, y, z) = 63 features.
3. **Train** a stacked LSTM network (3 LSTM layers of 64/128/64 units, then dense layers) on 30-frame landmark sequences to classify 22 gestures. The architecture is saved in `model.json` and the weights in `model.keras`.
4. **Predict live** (`app.ipynb`): stream the webcam with OpenCV, run the model on the latest 30 frames, and overlay the predicted gesture and class probabilities.

Gestures in the current model: Hello, Yes, No, Thank you, Sorry, Eat, Like, Want, I Love You, Which, Peace, Bathroom, Good, Bad, Stop, Money, Happy, Phone, Hear, See, Drink, Walk.

## Tech stack

Python · TensorFlow / Keras · MediaPipe · OpenCV · NumPy

## Run it

```bash
pip install -r requirements.txt
python detect.py              # webcam 0, confidence threshold 0.8
python detect.py --camera 1 --threshold 0.9
```

Hold your hand inside the white box. A gesture is shown only after it wins 10 predictions in a row above the threshold, which stops the label from flickering. Press `q` to quit. `app.ipynb` runs the same loop inside Jupyter.

To train on your own gestures, collect data with `gestures.ipynb`, extract 30-frame keypoint sequences into `MP_Data/<gesture>/<sequence>/<frame>.npy`, then run:

```bash
python train.py
```

It holds out 15% of sequences (stratified), trains with early stopping, prints a per-gesture classification report and saves `model.json` and `model.keras`. Note that `gestures.ipynb` currently has key bindings for 12 of the 22 gestures, so add bindings for any new ones.

## Files

| File | Purpose |
| --- | --- |
| `detect.py` | Real-time recognition from the webcam (command line) |
| `train.py` | Train and evaluate the LSTM on collected keypoint sequences |
| `app.ipynb` | Real-time recognition inside Jupyter |
| `gestures.ipynb` | Gesture image collection |
| `function.py` | MediaPipe detection, keypoint extraction and gesture labels |
| `function.py.txt` | Earlier version of `function.py` for A–Z letters |
| `model.json`, `model.keras` | Trained model architecture and weights |
| `requirements.txt` | Python dependencies |
