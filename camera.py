import cv2
import time
import requests
from discord_webhook import DiscordWebhook

# Load the pre-trained MobileNet SSD model and its configuration
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "mobilenet_iter_73000.caffemodel"
)

# Discord webhook URL
webhook_url = 'https://discord.com/api/webhooks/1177142622683414618/Uw5ezUS39aw0uNGlVx_uSWFXQL3OHpycOD38nBVKP3lAICTqLHS3cz2fp-ju9EkgzvhE'

# Open a video capture object (0 for default camera)
cap = cv2.VideoCapture(0)

# Set the desired frame rate
desired_fps = 30
interval = 1 / desired_fps

while True:
    # Record the start time to calculate processing time
    start_time = time.time()

    # Read a frame from the camera
    ret, frame = cap.read()

    # Resize the frame to 300x300 pixels (the size expected by the model)
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (50, 50), 127.5)

    # Set the input to the neural network
    net.setInput(blob)

    # Perform object detection
    detections = net.forward()

    person_found = False

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If confidence is above a certain threshold (e.g., 0.2)
        if confidence > 0.80:
            person_found = True

            # Get the coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and confidence
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            text = f"{confidence:.2f}%"
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Full-Body Detection', frame)

    # If a person is found, save the frame as a screenshot
    if person_found:
        cv2.imwrite('screenshot.png', frame)

        # Send the screenshot to Discord
        webhook = DiscordWebhook(url=webhook_url)
        with open('screenshot.png', 'rb') as f:
            webhook.add_file(file=f.read(), filename='screenshot.png')
        webhook.execute()

    # Calculate the processing time
    processing_time = time.time() - start_time

    # Calculate the time to sleep to achieve the desired frame rate
    sleep_time = max(0, interval - processing_time)
    time.sleep(sleep_time)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
