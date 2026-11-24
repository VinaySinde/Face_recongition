import cv2
import os

# ==========================================
# PROJECT PATH
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

# ==========================================
# CHECK HAAR CASCADE
# ==========================================

print("Cascade file:")
print(CASCADE_PATH)

if not os.path.exists(CASCADE_PATH):
    print("ERROR: Haar Cascade XML file not found!")
    exit()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("ERROR: Could not load Haar Cascade!")
    exit()

print("Haar Cascade loaded successfully!")

# ==========================================
# GET PERSON NAME
# ==========================================

name = input("\nEnter person's name: ").strip()

if name == "":
    print("Name cannot be empty.")
    exit()

# Create dataset folder
if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)

# Person folder
person_path = os.path.join(
    DATASET_PATH,
    name
)

if not os.path.exists(person_path):
    os.makedirs(person_path)

# ==========================================
# OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():

    print("ERROR: Could not open camera.")

    print("\nTry:")
    print("1. Check Windows Camera permissions")
    print("2. Close Camera/Zoom/Teams")
    print("3. Check camera driver")

    exit()

print("\nCamera started!")
print("Look at the camera.")
print("Press Q to stop.")
print("Collecting 50 face images...\n")

count = 0
MAX_IMAGES = 50

# ==========================================
# FACE COLLECTION
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Crop face
        face = gray[
            y:y + h,
            x:x + w
        ]

        count += 1

        # Save face
        filename = os.path.join(
            person_path,
            f"{count}.jpg"
        )

        cv2.imwrite(
            filename,
            face
        )

        # Display counter
        cv2.putText(
            frame,
            f"Images: {count}/{MAX_IMAGES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Show camera
    cv2.imshow(
        "Face Dataset Collection",
        frame
    )

    # Stop after 50 images
    if count >= MAX_IMAGES:
        break

    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================================
# CLEANUP
# ==========================================

camera.release()
cv2.destroyAllWindows()

print("\n===================================")
print("Face collection completed!")
print("===================================")

print(f"Person: {name}")
print(f"Images collected: {count}")
print(f"Saved to: {person_path}")