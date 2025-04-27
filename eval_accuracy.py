import os
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, top_k_accuracy_score

# 1. Load known encodings
with open('encodings.pkl', 'rb') as f:
    data = pickle.load(f)
known_embs = np.array(data['encodings'])
known_names = data['names']

# 2. Prepare models
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False)      # detect one face :contentReference[oaicite:2]{index=2}
resnet = InceptionResnetV1(pretrained='vggface2').eval()   # pretrained FaceNet :contentReference[oaicite:3]{index=3}

# 3. Validation data
val_dir = 'validation'
y_true, y_pred = [], []

for name in os.listdir(val_dir):
    person_dir = os.path.join(val_dir, name)
    if not os.path.isdir(person_dir):
        continue
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        try:
            # 4. Detect & embed
            from PIL import Image
            img = Image.open(img_path)
            face = mtcnn(img)
            if face is None:
                continue
            emb = resnet(face.unsqueeze(0)).detach().numpy()[0]
            # 5. Predict via nearest neighbor
            dists = np.linalg.norm(known_embs - emb, axis=1)
            idx = np.argmin(dists)
            pred = known_names[idx] if dists[idx] < 0.8 else "Unknown"  # threshold :contentReference[oaicite:4]{index=4}
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            continue
        y_true.append(name)
        y_pred.append(pred)

# 6. Compute metrics
acc      = accuracy_score(y_true, y_pred)                                 # overall accuracy :contentReference[oaicite:5]{index=5}
bal_acc  = balanced_accuracy_score(y_true, y_pred)                        # per-class balanced acc :contentReference[oaicite:6]{index=6}
conf_mat = confusion_matrix(y_true, y_pred, labels=sorted(set(known_names)))  # confusion matrix :contentReference[oaicite:7]{index=7}

# Optional: top-2 accuracy
# scores = ...  # you’d need classifier scores; skip if unavailable
# top2 = top_k_accuracy_score(y_true, scores, k=2, labels=sorted(set(known_names))) :contentReference[oaicite:8]{index=8}

print(f"Total samples      : {len(y_true)}")
print(f"Overall Accuracy   : {acc * 100:.2f}%")
print(f"Balanced Accuracy  : {bal_acc * 100:.2f}%")
print("Confusion Matrix:")
print(conf_mat)