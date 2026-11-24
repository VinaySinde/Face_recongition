import cv2
import os
import numpy as np

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

TRAINER_PATH = os.path.join(
    BASE_DIR,
    "trainer"
)

CASCADE_PATH = os.path.join(
    BASE_DIR,
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# CHECK HAAR CASCADE
# ==========================================

print("Checking Haar Cascade...")

print("Cascade path:")
print(CASCADE_PATH)

if not os.path.exists(CASCADE_PATH):

    print("\nERROR: Haar Cascade XML file not found!")

    print("\nMake sure this file exists:")
    print(CASCADE_PATH)

    exit()

face_detector = cv2.CascadeClassifier(
    CASCADE_PATH
)

if face_detector.empty():

    print("\nERROR: Could not load Haar Cascade.")

    exit()

print("Haar Cascade loaded successfully!")

# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(DATASET_PATH):

    print("\nERROR: Dataset folder not found.")

    print(DATASET_PATH)

    exit()

# ==========================================
# CREATE TRAINER FOLDER
# ==========================================

if not os.path.exists(TRAINER_PATH):

    os.makedirs(TRAINER_PATH)

# ==========================================
# CHECK LBPH
# ==========================================

if not hasattr(cv2, "face"):

    print("\nERROR: cv2.face is not available.")

    print("\nInstall:")
    print("python -m pip install opencv-contrib-python")

    exit()

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# ==========================================
# READ DATASET
# ==========================================

faces = []
labels = []

names = {}

current_id = 0

print("\nReading dataset...\n")

for person_name in os.listdir(DATASET_PATH):

    person_path = os.path.join(
        DATASET_PATH,
        person_name
    )

    # Skip files
    if not os.path.isdir(person_path):
        continue

    current_id += 1

    names[current_id] = person_name

    print(
        f"Loading images for: {person_name}"
    )

    for image_name in os.listdir(person_path):

        image_path = os.path.join(
            person_path,
            image_name
        )

        # Read image
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(
                f"Could not read: {image_path}"
            )
            continue

        # Detect face in image
        detected_faces = face_detector.detectMultiScale(
            image,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50)
        )

        # If face detected
        if len(detected_faces) > 0:

            for (x, y, w, h) in detected_faces:

                face = image[
                    y:y + h,
                    x:x + w
                ]

                faces.append(face)
                labels.append(current_id)

        else:

            # If image is already cropped to face,
            # use the entire image.
            faces.append(image)
            labels.append(current_id)

# ==========================================
# CHECK TRAINING DATA
# ==========================================

if len(faces) == 0:

    print("\nERROR: No face images found.")

    print("\nMake sure you ran:")
    print("python collect_face.py")

    exit()

labels = np.array(labels)

print("\n===================================")
print("Training information")
print("===================================")

print(
    f"People found: {len(names)}"
)

print(
    f"Training images: {len(faces)}"
)

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining model...")
print("Please wait...")

recognizer.train(
    faces,
    labels
)

# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    TRAINER_PATH,
    "face_model.yml"
)

recognizer.write(
    model_path
)

# ==========================================
# SAVE NAMES
# ==========================================

names_path = os.path.join(
    TRAINER_PATH,
    "names.txt"
)

with open(
    names_path,
    "w",
    encoding="utf-8"
) as file:

    for person_id, person_name in names.items():

        file.write(
            f"{person_id},{person_name}\n"
        )

# ==========================================
# FINISHED
# ==========================================

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY!")
print("===================================")

print("\nModel saved at:")

print(model_path)

print("\nNames saved at:")

print(names_path)

print("\nRegistered people:")

for person_id, person_name in names.items():

    print(
        f"{person_id} -> {person_name}"
    )

print("\nNow run:")

print("python face_recogintion.py")