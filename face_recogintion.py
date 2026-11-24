import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
MODEL_PATH = os.path.join(BASE_DIR, "trainer", "face_model.yml")
NAMES_PATH = os.path.join(BASE_DIR, "trainer", "names.txt")

RECOGNITION_THRESHOLD = 55

print("===================================")
print("FACE RECOGNITION SYSTEM")
print("===================================")

# Check files
if not os.path.exists(CASCADE_PATH):
    print("ERROR: Haar Cascade not found!")
    exit()

if not os.path.exists(MODEL_PATH):
    print("ERROR: Trained model not found!")
    print("Run train.py first.")
    exit()

if not os.path.exists(NAMES_PATH):
    print("ERROR: names.txt not found!")
    print("Run train.py first.")
    exit()

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if face_cascade.empty():
    print("ERROR: Could not load Haar Cascade!")
    exit()

# Check cv2.face
if not hasattr(cv2, "face"):
    print("ERROR: cv2.face is not available.")
    print("Install opencv-contrib-python")
    exit()

# Load recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

print("Trained model loaded successfully!")

# Load registered names
names = {}

with open(NAMES_PATH, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        parts = line.split(",", 1)

        if len(parts) == 2:
            person_id = int(parts[0])
            person_name = parts[1]

            names[person_id] = person_name

print("\nRegistered People:")

for person_id, person_name in names.items():
    print(person_id, "->", person_name)


# Open camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("\n===================================")
print("CAMERA STARTED")
print("===================================")
print("Press Q to exit.")


while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(100, 100)
    )

    # Counters
    total_faces = len(faces)
    registered_faces = 0
    unknown_faces = 0

    # Process each face
    for (x, y, w, h) in faces:

        face = gray[y:y + h, x:x + w]

        person_id, distance = recognizer.predict(face)

        # Check whether person is registered
        if person_id in names and distance <= RECOGNITION_THRESHOLD:

            person_name = names[person_id]

            label = person_name
            status = "REGISTERED"

            box_color = (0, 255, 0)

            registered_faces += 1

        else:

            label = "UNKNOWN PERSON"
            status = "NOT REGISTERED"

            box_color = (0, 0, 255)

            unknown_faces += 1

        # Face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            3
        )

        # Person name
        cv2.putText(
            frame,
            label,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2
        )

        # Status
        cv2.putText(
            frame,
            status,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )

    # Calculate registered percentage
    if total_faces > 0:
        registered_percentage = (
            registered_faces / total_faces
        ) * 100
    else:
        registered_percentage = 0

    # Calculate unknown percentage
    if total_faces > 0:
        unknown_percentage = (
            unknown_faces / total_faces
        ) * 100
    else:
        unknown_percentage = 0

    # Dashboard background
    cv2.rectangle(
        frame,
        (10, 10),
        (400, 135),
        (30, 30, 30),
        -1
    )

    # Total faces
    cv2.putText(
        frame,
        f"Total Faces: {total_faces}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Registered count
    cv2.putText(
        frame,
        f"Registered: {registered_faces}",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # Unknown count
    cv2.putText(
        frame,
        f"Unknown: {unknown_faces}",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2
    )

    # Registered percentage
    cv2.putText(
        frame,
        f"Registered: {registered_percentage:.1f}%",
        (25, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # Show camera
    cv2.imshow(
        "Face Recognition System",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()

print("\nFace recognition stopped.")