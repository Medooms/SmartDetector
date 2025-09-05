import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

# Liste des classes COCO
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# Charger le modèle avec les bons poids
poids = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
modele = fasterrcnn_resnet50_fpn(weights=poids)
modele.eval()

# Charger une image locale
image_path = "bus.jpg"
image_cv = cv2.imread(image_path)
if image_cv is None:
    raise FileNotFoundError(f"L'image '{image_path}' est introuvable.")
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# Conversion en tensor
image_tensor = F.to_tensor(image_rgb)

# Détection
with torch.no_grad():
    sorties = modele([image_tensor])

# Seuil de confiance
seuil = 0.5

# Boucle sur les résultats
for box, score, label in zip(sorties[0]['boxes'], sorties[0]['scores'], sorties[0]['labels']):
    if score > seuil:
        x1, y1, x2, y2 = box.int().tolist()  # Conversion en liste d'entiers
        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        texte = f"{COCO_CLASSES[label.item()]}: {score:.2f}"  # Nom + score
        cv2.putText(image_rgb, texte, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

# Afficher l'image avec matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()
