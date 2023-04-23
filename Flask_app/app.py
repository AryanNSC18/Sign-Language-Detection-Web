import numpy as np
from flask import Flask, request, jsonify, render_template,Response
import pickle
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
pred = 0
word_list = []
def model_predict():
    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        if not ret:
            break
        else:
            model_dict = pickle.load(open('path to the saved model', 'rb'))
            model = model_dict['model']
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles

            hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

            labels_dict = {1: 'Z', 2: 'Y', 3: 'X', 4: 'W', 5: "V", 6: "U", 7: "T", 8: "S", 9: "R", 10: "Q", 11: "P",
                           12: "O", 13: "N", 14: "M", 15: "L",
                           16: "K", 17: "J", 18: "I", 19: "H", 20: "G", 21: "F", 22: "E", 23: "D", 24: "C", 25: "B",
                           26: "A"}

            data_aux = []
            x_ = []
            y_ = []

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                global pred
                predicted_character = labels_dict[int(prediction[0])]
                pred = predicted_character
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
           b''b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def adv_model_predict(): # working on -> will be completed soon
    # import the created h5 model and start detection
    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        if not ret:
            break
        else:
            #model_dict = pickle.load(open('model path', 'rb'))


           ret, buffer = cv2.imencode('.jpg', frame)
           frame = buffer.tobytes()

        yield (b'--frame\r\n'
           b''b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form.get('action1') == 'Detect-Sign':
            word_list.append(pred)
            print(word_list)
            return render_template('index.html', recog_text='{}'.format(word_list))
        elif request.form.get('action2') == 'Remove-word':
            if len(word_list)==0:
                print("empty_list")
            else:
                word_list.pop()
                print("pooping a element")
            return render_template('index.html', recog_text='{}'.format(str(word_list)))
        elif request.form.get('action3') == 'Advance-Model':
            if request.form.get('action1') == 'Back':
                return render_template('index.html')
            else:
                return render_template('page_2.html')
        else:
            for i in range(0,len(word_list)):
                return render_template('copy_page.html', Final_text='{}'.format(str(i)))
    elif request.method == 'GET':
        return render_template('index.html')

    return render_template("index.html")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(model_predict(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/advance_video')
def adv_video():
    return Response(adv_model_predict(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
