# SmartDetector - Détection d’objets avec IA

## Description
SmartDetector est un projet de vision par ordinateur qui utilise un modèle de deep learning pré-entraîné (Faster R-CNN basé sur ResNet50) pour détecter et identifier automatiquement des objets dans des images.  
Le modèle est entraîné sur le dataset COCO, capable de reconnaître 90 classes d’objets (personnes, voitures, animaux, ballons, etc.).

Chaque objet détecté est entouré d’un rectangle rouge et annoté avec son nom ainsi que son score de confiance.

---

## Fonctionnalités
- Détection automatique d’objets dans une image  
- Affichage du nom de l’objet et de la probabilité de détection  
- Supporte 90 classes COCO (ex : person, dog, car, sports ball…)  
- Affichage des résultats avec OpenCV et Matplotlib  