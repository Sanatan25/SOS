import cv2
import numpy as np
import requests
import geocoder
from ultralytics import YOLO
from twilio.rest import Client

TWILIO_ACCOUNT_SID = 'ACf64a27f49c4025feecbfc0d8c0bfb271'
TWILIO_AUTH_TOKEN = '59e75d7f25a50c4507ae463dd928472c'
TWILIO_NUMBER = '+19782001831'
VERIFIED_USER = '+919348128142'  

def get_live_location():
    location = geocoder.ip('me') 
    if location.latlng:
        lat, lng = location.latlng
        return f"https://www.google.com/maps?q={lat},{lng}"
    return "Location not available"

def send_twilio_sms():
    location_link = get_live_location()
    message_body = f"🚨 SOS ALERT! Accident detected. Immediate help required.\nLive Location: {location_link}"

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=message_body,
        from_=TWILIO_NUMBER,
        to=VERIFIED_USER
    )
    print("Twilio SMS Sent! SID:", message.sid)

model = YOLO("yolov8n.pt")

FOCAL_LENGTH = 700 
KNOWN_WIDTH = 50 
SAFE_DISTANCE = 100  

def estimate_distance(bbox_width):
    return float("inf") if bbox_width == 0 else (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width

cam = "http://192.0.0.4:8080/video"
cap = cv2.VideoCapture(cam)

if not cap.isOpened():
    print("Error: Unable to open camera stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    results = model(frame)
    ts = len(results[0].boxes)  
    if ts == 0:
        send_twilio_sms()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            distance = estimate_distance(width)

           
            color = (0, 255, 0) if distance > SAFE_DISTANCE else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"Dist: {int(distance)} cm"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


            if distance < SAFE_DISTANCE:
                cv2.putText(frame, "WARNING: TOO CLOSE!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Collision Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
