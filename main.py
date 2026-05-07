import cv2
import numpy as np
import tempfile
import logging
from pathlib import Path
from typing import List, Dict
from skimage.metrics import structural_similarity as ssim

# Logger pour le suivi du processus dans la console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class VideoRestorer:
    def __init__(self, input_path: str, output_path: str):
        """
        Initialise le pipeline de restauration vidéo
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Métadonnées de la vidéo
        self.fps = 0.0
        self.width = 0
        self.height = 0
        
        # Stockage en RAM (Léger)
        self.proxies: Dict[int, np.ndarray] = {}
        # Stockage des chemins sur disque (Lourd)
        self.frame_paths: Dict[int, str] = {}
        
        # Dossier temporaire géré automatiquement par Python
        self.temp_dir = tempfile.TemporaryDirectory()

    def extract_and_preprocess(self) -> None:
        """Lit la vidéo, sauvegarde la HD sur disque et garde les proxys en RAM"""
        
        logging.info("Extraction des frames et création des proxys...")
        cap = cv2.VideoCapture(str(self.input_path))
        
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {self.input_path}")
            
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # On définit la largeur cible du proxy
        proxy_w = 128
        # On calcule la hauteur proportionnelle
        proxy_h = int((proxy_w / self.width) * self.height)
        
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sauvegarde HD sur le disque temporaire
            frame_path = Path(self.temp_dir.name) / f"frame_{frame_id:04d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            self.frame_paths[frame_id] = str(frame_path)
            
            # Création du proxy proportionnel en niveaux de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            proxy = cv2.resize(gray, (proxy_w, proxy_h))
            self.proxies[frame_id] = proxy
            
            frame_id += 1
            
        cap.release()
        logging.info(f"-> {frame_id} frames extraites. Dimensions originales: {self.width}x{self.height} @ {self.fps} FPS")

    def pathfinding(self) -> List[int]:
        """Calcule la matrice SSIM et trouve le chemin optimal (Greedy NN Multi-départ)"""
        
        logging.info("Phase 3: Calcul de la matrice de distance SSIM et Pathfinding...")
        n_frames = len(self.proxies)
        dist_matrix = np.zeros((n_frames, n_frames))
        
        # Calcul de la matrice de distance symétrique (1 - SSIM)
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                score = ssim(self.proxies[i], self.proxies[j], data_range=255)
                dist = 1.0 - score
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Greedy NN avec points de départs multiples
        best_path = []
        best_cost = float('inf')
        
        for start_idx in range(n_frames):
            unvisited = set(range(n_frames))
            current = start_idx
            unvisited.remove(current)
            
            path = [current]
            cost = 0.0
            
            while unvisited:
                # Trouve le voisin le plus proche non visité
                nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
                cost += dist_matrix[current, nearest]
                
                current = nearest
                unvisited.remove(current)
                path.append(current)
                
            # Met à jour le meilleur chemin global
            if cost < best_cost:
                best_cost = cost
                best_path = path
                
        logging.info(f"-> Chemin optimal trouvé. Coût total : {best_cost:.4f}")
        return best_path

    def trim(self, optimal_path: List[int]) -> List[int]:
        """Coupe la séquence aux pics de distance (Médiane + 3*MAD) et garde la plus longue."""
        
        logging.info("Phase 4: Élimination des intrus (Shot Boundary Detection)...")
        
        # Calcul des coûts de transition dans le chemin optimal
        transitions = []
        for i in range(len(optimal_path) - 1):
            id_a = optimal_path[i]
            id_b = optimal_path[i+1]
            # On recalcule juste le 1-SSIM entre ces deux là pour aller vite
            score = ssim(self.proxies[id_a], self.proxies[id_b], data_range=255)
            transitions.append(1.0 - score)
            
        transitions = np.array(transitions)
        
        # Seuil dynamique robuste
        median = np.median(transitions)
        mad = np.median(np.abs(transitions - median))
        threshold = median + (3 * mad)
        
        # Découpage du chemin en sous-séquences
        sub_sequences = []
        current_seq = [optimal_path[0]]
        
        for i, cost in enumerate(transitions):
            if cost > threshold:
                # Rupture (Pic de distance) = on clôture la séquence actuelle
                sub_sequences.append(current_seq)
                current_seq = [optimal_path[i+1]]
            else:
                current_seq.append(optimal_path[i+1])
        sub_sequences.append(current_seq)
        
        # On garde uniquement le plus grand bloc (la vidéo principale)
        longest_seq = max(sub_sequences, key=len)
        logging.info(f"-> {len(sub_sequences)} blocs trouvés. Conservation du bloc principal ({len(longest_seq)} frames).")
        
        return longest_seq

    def reconstruct(self, final_sequence: List[int]):
        """Génère le fichier vidéo .mp4 final depuis le disque."""
        logging.info("Phase 5: Reconstruction de la vidéo finale...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        
        for frame_id in final_sequence:
            # Streaming depuis le disque pour épargner la RAM
            frame = cv2.imread(self.frame_paths[frame_id])
            out.write(frame)
            
        out.release()
        logging.info(f"✅ Succès ! Vidéo sauvegardée sous : {self.output_path}")

    def run(self):
        """Exécute l'intégralité du pipeline."""
        try:
            self.extract_and_preprocess()
            optimal_path = self.pathfinding()
            final_sequence = self.trim(optimal_path)
            
            # Limte du projet : si la vidéo est à l'envers, on peut inverser la séquence finale avant reconstruction
            logging.info("Inversion de la séquence (Correction du 50/50)...")
            final_sequence.reverse()
            
            self.reconstruct(final_sequence)
        finally:
            # Nettoyage automatique du disque (même en cas de plantage)
            self.temp_dir.cleanup()

if __name__ == "__main__":
    input_video = "corrupted_video.mp4"
    output_video = "restored_video.mp4"
    
    restorer = VideoRestorer(input_video, output_video)
    restorer.run()