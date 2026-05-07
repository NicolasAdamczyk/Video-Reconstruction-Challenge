# Restauration Vidéo : Élimination d'Intrus et Réordonnancement

Ce test technique propose un pipeline algorithmique permettant de restaurer une vidéo corrompue (sans apprentissage profond). Il est capable de retrouver l'ordre chronologique des images et d'éliminer les "parasites" insérés volontairement, tout en minimisant l'empreinte mémoire (Memory-Safe).

## Installation & Utilisation

1. Cloner le dépôt et installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Placer la vidéo corrompue (ex: corrupted_video.mp4) à la racine du projet.

3. Lancer le pipeline de restauration :

```bash
python main.py
```

*Note : Un Notebook.ipynb est également fourni. Il détaille pas-à-pas les choix d'architecture, la justification mathématique, et affiche les preuves visuelles du nettoyage (Heatmap, Profil de continuité).*

## Architecture du Pipeline (4 Phases)

L'algorithme s'exécute en 4 étapes majeures :

1. **Extraction & Prétraitement** : Streaming des images HD sur le disque (pour éviter l'Out-Of-Memory) et chargement en RAM de "Proxys" légers redimensionnés proportionnellement en niveaux de gris.

2. **Réordonnancement Global (Pathfinding)** : Calcul d'une matrice de similarité globale via la métrique SSIM. Utilisation d'un algorithme glouton multi-départs (Greedy NN) pour reconstruire la continuité temporelle de la vidéo.

3. **Shot Boundary Detection (Trimming)** : Calcul d'un seuil dynamique robuste ($Médiane + 3*MAD$) basé sur les coûts de transition SSIM afin d'isoler et de supprimer mathématiquement les anomalies/parasites.

4. **Reconstruction** : Génération du fichier .mp4 final en streamant les frames HD validées depuis le disque, tout en respectant le framerate et l'aspect ratio d'origine.

## Limites assumées du pipeline

**Symétrie Temporelle** : La matrice SSIM étant symétrique, l'algorithme garantit une continuité visuelle parfaite mais la probabilité du sens de lecture initial est de 50/50. Une couche d'Optical Flow serait nécessaire pour deviner le sens de la gravité sans intervention humaine.

**Séparation Sémantique** : Le modèle se base sur la similarité structurelle pure. Si la vidéo corrompue est un mélange de deux vidéos de durées équivalentes, l'algorithme ne peut pas deviner "sémantiquement" laquelle doit être conservée sans modèle d'apprentissage profond (Deep Learning).

**Restauration Audio** : Le découpage de la vidéo fractionne l'audio en échantillons trop courts (< 0.04s) pour un réalignement classique. La piste son n'est donc pas reconstruite dans ce pipeline.
