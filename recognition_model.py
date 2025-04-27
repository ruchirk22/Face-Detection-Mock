import cv2, pickle, numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from tkinter import Tk, Label
from PIL import Image, ImageTk

# Load embeddings
with open('encodings.pkl', 'rb') as f:
    data = pickle.load(f)
known_encs = np.array(data['encodings'])
known_names = data['names']

# Init models & camera
mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
cam = cv2.VideoCapture(0)

# Build Tkinter UI
root = Tk(); root.title("Celebrity Recognizer")
video_label = Label(root); video_label.pack()
greet_label = Label(root, font=('Arial', 20)); greet_label.pack()

def update():
    ret, frame = cam.read()
    if not ret: return
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    faces = mtcnn(img)
    name = "Unknown"
    if faces is not None:
        emb = resnet(faces.unsqueeze(0)).detach().numpy()[0]
        dists = np.linalg.norm(known_encs - emb, axis=1)
        idx = np.argmin(dists)
        if dists[idx] < 0.8:  # threshold
            name = known_names[idx]
    # Draw and display
    cv2.putText(frame, name, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)
    imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video_label.imgtk = imgtk; video_label.configure(image=imgtk)
    greet_label.config(text=f"Hello, {name}!" if name!="Unknown" else "")
    root.after(10, update)

update(); root.mainloop()
cam.release(); cv2.destroyAllWindows()