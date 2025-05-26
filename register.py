import cv2
import os
import pickle
import numpy as np
from ultralytics import YOLO

# Constants
KNOWN_OBJECTS_DIR = "known_objects"
EMBEDDINGS_FILE = "object_embeddings.pkl"
MIN_OBJECT_SIZE = 100  # Minimum pixel width/height for registration

def extract_object_embedding(frame, box, model):
    """
    Extract embedding for an object using YOLO's features
    box: [x1, y1, x2, y2] coordinates
    """
    x1, y1, x2, y2 = map(int, box)
    object_roi = frame[y1:y2, x1:x2]
    
    # Skip if object region is too small
    if object_roi.size == 0 or min(object_roi.shape[:2]) < MIN_OBJECT_SIZE:
        return None
    
    # Get YOLO features for the object
    results = model(object_roi, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None
    
    # Use the first detected object's features as embedding
    return results[0].boxes[0].cls.cpu().numpy()

def save_object_data(name, embedding, image):
    """Save object data to disk"""
    os.makedirs(KNOWN_OBJECTS_DIR, exist_ok=True)
    
    # Save object image
    img_path = os.path.join(KNOWN_OBJECTS_DIR, f"{name}.jpg")
    cv2.imwrite(img_path, image)
    
    # Load or create embeddings dictionary
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}
    
    # Add new embedding
    embeddings[name] = embedding
    
    # Save updated embeddings
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)

def register_new_object():
    # Take user input
    name = input("Enter name for the object: ").strip()
    if not name:
        print("[-] Invalid name. Registration aborted.")
        return None

    # Check for existing registration
    existing_files = [f for f in os.listdir(KNOWN_OBJECTS_DIR) if f.startswith(name)]
    if existing_files:
        print(f"[-] Object '{name}' already exists in the database.")
        overwrite = input("Do you want to overwrite? (y/n): ").lower()
        if overwrite != 'y':
            print("[INFO] Registration cancelled.")
            return None

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[-] Could not open camera.")
        return None

    cv2.namedWindow("Register Object", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Register Object", 800, 600)
    
    print("[INFO] Press 's' to capture object, 'q' to quit.")
    print("[INFO] Make sure the object is clearly visible and well-lit.")

    registered = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[-] Failed to capture frame.")
            continue

        frame = cv2.flip(frame, 1)  # Mirror
        
        # Detect objects in real-time
        results = model(frame, verbose=False)
        annotated_frame = frame.copy()
        
        # Draw all detected objects
        if results:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls)
                label = model.names[cls_id]
                
                # Draw bounding box
                color = (0, 255, 0) if label.lower() == name.lower() else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Display instructions
        cv2.putText(annotated_frame, "Press 's' to capture", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Register Object", annotated_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if results and len(results[0].boxes) > 0:
                # Get the most confident detection
                best_box = max(results[0].boxes, key=lambda x: float(x.conf))
                x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                
                # Extract embedding
                embedding = extract_object_embedding(frame, [x1, y1, x2, y2], model)
                if embedding is not None:
                    # Save object image and embedding
                    object_image = frame[y1:y2, x1:x2]
                    save_object_data(name, EMBEDDINGS_FILE, object_image)
                    print(f"[✓] Object successfully registered: {name}")
                    registered = True
                else:
                    print("[-] Failed to extract object features. Try again.")
            else:
                print("[-] No objects detected. Try again.")
            break

        elif key == ord('q'):
            print("[INFO] Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return name if registered else None

if __name__ == "__main__":
    registered_name = register_new_object()
    if registered_name:
        print(f"Registration complete for object: {registered_name}")