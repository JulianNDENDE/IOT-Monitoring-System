import cv2
import time
from discord_webhook import DiscordWebhook

def load_model():
    # Load the pre-trained MobileNet SSD model and its configuration
    return cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")

def initialize_webhook():
    # Discord webhook URL
    return 'https://ptb.discord.com/api/webhooks/1179307722634690691/M6QFFoudj-08JaL3Pj2QT766jysEFut6I2IzPLSTwb3ZCDTvZadPb1g-IrAoWgBo2nY4'

def open_camera(capture_index=0):
    # Open a video capture object (0 for default camera)
    return cv2.VideoCapture(capture_index)

def process_frame(frame, net):
    # Resize the frame to 300x300 pixels (the size expected by the model)
    frame = cv2.resize(frame, (300, 300))

    # Set the input to the neural network
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Perform object detection
    net.setInput(blob)
    detections = net.forward()

    return frame, detections

def draw_and_display(frame, detections):
    person_found = False

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # If confidence is above a certain threshold (e.g., 0.8)
        if confidence > 0.8:
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

    return person_found

def send_screenshot_to_discord(frame, webhook_url):
    # Save the frame as a screenshot
    cv2.imwrite('screenshot.png', frame)

    # Send the screenshot to Discord
    webhook = DiscordWebhook(url=webhook_url)
    with open('screenshot.png', 'rb') as f:
        webhook.add_file(file=f.read(), filename='screenshot.png')
    webhook.execute()
    print("Screenshot sent to Discord!")

def main():
    net = load_model()
    webhook_url = initialize_webhook()
    cap = open_camera()

    desired_fps = 30
    interval = 1 / desired_fps
    timer = 0

    while True:
        start_time = time.time()

        frame = cap.read()

        frame, detections = process_frame(frame, net)

        person_found = draw_and_display(frame, detections)

        # Send a screenshot to Discord every 30 seconds if a person is found
        # (900 frames at 30 fps = 30 seconds)
        if person_found and timer % 900 == 0:
            send_screenshot_to_discord(frame, webhook_url)
            timer = 1

        # Print the timer every 10 seconds
        if timer % 300 == 0: 
            print(f"Timer: {timer / 30} seconds")

        timer += 1

        # Sleep for the remaining time to achieve the desired FPS
        processing_time = time.time() - start_time
        sleep_time = max(0, interval - processing_time)
        time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
