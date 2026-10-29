"""Real-time sign language recognition from the webcam.

Usage: python detect.py [--camera 0] [--threshold 0.8]
Show your hand inside the white box. Press q to quit.
"""
import argparse

import cv2
import keras
import mediapipe as mp
import numpy as np

from function import actions, draw_styled_landmarks, extract_keypoints, mediapipe_detection, sequence_length

STABLE_FRAMES = 10  # the same gesture must win this many predictions in a row before it is shown


def draw_hand_bbox(image, results, margin=20):
    bbox_coords = None
    if results.multi_hand_landmarks:
        h, w, _ = image.shape
        for hand_landmarks in results.multi_hand_landmarks:
            xs = [int(landmark.x * w) for landmark in hand_landmarks.landmark]
            ys = [int(landmark.y * h) for landmark in hand_landmarks.landmark]
            x_min, y_min = max(0, min(xs) - margin), max(0, min(ys) - margin)
            x_max, y_max = min(w, max(xs) + margin), min(h, max(ys) + margin)
            bbox_coords = (x_min, y_min, x_max, y_max)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image, bbox_coords


def stable_prediction(predictions, probs, threshold, stable_frames=STABLE_FRAMES):
    """Return the gesture index once it has won the last `stable_frames` predictions with enough confidence."""
    best = int(np.argmax(probs))
    recent = predictions[-stable_frames:]
    if len(recent) == stable_frames and all(p == best for p in recent) and probs[best] > threshold:
        return best
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', type=int, default=0, help='webcam index')
    parser.add_argument('--threshold', type=float, default=0.8, help='minimum confidence to show a gesture')
    args = parser.parse_args()

    model = keras.models.load_model('model.keras')
    sequence, predictions = [], []
    current = None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f'Could not open camera {args.camera}')

    try:
        with mp.solutions.hands.Hands(model_complexity=0,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # The model was trained on the 300x360 region at the top left of the frame
                cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
                cropframe = frame[40:400, 0:300]
                image, results = mediapipe_detection(cropframe, hands)
                draw_styled_landmarks(image, results)
                image, bbox_coords = draw_hand_bbox(image, results, margin=40)

                if results.multi_hand_landmarks:
                    sequence.append(extract_keypoints(results))
                    sequence = sequence[-sequence_length:]

                    if len(sequence) == sequence_length:
                        probs = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        predictions.append(int(np.argmax(probs)))
                        predictions = predictions[-STABLE_FRAMES:]
                        best = stable_prediction(predictions, probs, args.threshold)
                        if best is not None:
                            current = (actions[best], float(probs[best]))
                else:
                    # Start over when the hand leaves the frame
                    sequence, predictions, current = [], [], None

                frame[40:400, 0:300] = image

                if bbox_coords and current:
                    # bbox is relative to the crop, which starts 40px down the frame
                    x_min, y_min, x_max, _ = bbox_coords
                    top = y_min + 40
                    label = f'{current[0]} ({current[1]:.0%})'
                    cv2.rectangle(frame, (x_min, max(0, top - 35)), (x_max, top), (245, 117, 16), -1)
                    cv2.putText(frame, label, (x_min + 3, max(25, top - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Sign Language Detection', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
