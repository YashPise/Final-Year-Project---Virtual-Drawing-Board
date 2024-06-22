import base64
import logging
from flask import Flask, render_template, Response, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
import HandTrackingModule as htm
import os

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(filename='flask.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


brushThickness = 20
eraserThickness = 50
drawColor = (255, 0, 255)
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
whiteCanvas = np.zeros((720, 1280, 3), np.uint8) + 255

folderPath = "header"  # Adjusted path
myList = os.listdir(folderPath)
overlayList = [cv2.imread(os.path.join(folderPath, imPath)) for imPath in myList]  # Modified line
header = overlayList[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.50, maxHands=1)

def generate_frames():
    global header  # Declare header as global
    global drawColor

    drawColor = (255, 0, 255)  # Set default drawColor

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1], lmList[8][2]
            x2, y2 = lmList[12][1], lmList[12][2]
            fingers = detector.fingersUp()

            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = (255, 0, 0)
                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(whiteCanvas, (xp, yp), (x1, y1), (255, 255, 255), eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor)
                    cv2.line(whiteCanvas, (xp, yp), (x1, y1), drawColor)
                xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        img[0:125, 0:1280] = header

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def white_frames():
    global header  # Declare header as global
    global drawColor

    drawColor = (255, 0, 255)  # Set default drawColor

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1], lmList[8][2]
            x2, y2 = lmList[12][1], lmList[12][2]
            fingers = detector.fingersUp()

            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                if y1 < 125:
                    if 250 < x1 < 450:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 550 < x1 < 750:
                        header = overlayList[1]
                        drawColor = (66, 5, 5)
                    elif 800 < x1 < 950:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            if fingers[1] and not fingers[2]:
                cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(whiteCanvas, (xp, yp), (x1, y1), (255, 255, 255), eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor)
                    cv2.line(whiteCanvas, (xp, yp), (x1, y1), drawColor)
                xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        whiteImg = cv2.bitwise_or(img, whiteCanvas)

        img[0:125, 0:1280] = header

        ret, buffer = cv2.imencode('.jpg', whiteImg)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    logging.info('Received request to /video_feed')
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error('Error generating frames:', e)
        return str(e), 500

@app.route('/whiteimg')
def white_feed():
    logging.info('Received request to /whiteimg')
    try:
        return Response(white_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logging.error('Error generating frames:', e)
        return str(e), 500


@app.route('/save_screenshot', methods=['POST'])
def save_screenshot():
    try:
        # Capture the current frame from the white canvas stream
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=False)

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)
        whiteImg = cv2.bitwise_or(img, whiteCanvas)

        # Save the image
        img_path = 'images/screenshot.jpg'
        cv2.imwrite(img_path, whiteImg)
        return send_file(img_path, as_attachment=True)
    except Exception as e:
        logging.error('Error saving screenshot:', e)
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)





