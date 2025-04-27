import os, pickle
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# 1. Point to your Windows dataset root
dataset_path = r"F:\Projects\Face-Detection-Mock\dataset"

# 2. Initialize models
mtcnn    = MTCNN(image_size=160, margin=0)
resnet   = InceptionResnetV1(pretrained='vggface2').eval()

encodings, names = [], []

# 3. Traverse each subfolder (folder1, folder2, …)
for celeb in os.listdir(dataset_path):
    celeb_dir = os.path.join(dataset_path, celeb)
    # Skip non-folders
    if not os.path.isdir(celeb_dir):
        continue
    for img_name in os.listdir(celeb_dir):
        img_path = os.path.join(celeb_dir, img_name)
        try:
            img = Image.open(img_path)
        except Exception:
            continue
        face = mtcnn(img)
        if face is None:
            continue
        emb = resnet(face.unsqueeze(0)).detach().numpy()[0]
        encodings.append(emb)
        names.append(celeb)

# 4. Persist to disk
with open('encodings.pkl', 'wb') as f:
    pickle.dump({'encodings': encodings, 'names': names}, f)

print("✅ Encodings saved from all folders!")
print("Total encodings:", len(encodings))