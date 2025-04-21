import numpy as np
import tensorflow as tf

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def detect_person(frame, interpreter, input_details, output_details, threshold=0.5):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (300, 300))
    input_tensor = np.expand_dims(img, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > threshold and int(classes[i]) == 0:
            ymin, xmin, ymax, xmax = boxes[i]
            return int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
    return None
