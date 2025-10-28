"""
Module de prédiction pour AgriDetect
Utilise le modèle entraîné pour détecter les maladies
"""

import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import Dict, List, Tuple


class DiseaseDetector:
    """
    Détecteur de maladies des plantes
    """
    
    def __init__(self, model_path: str):
        """
        Initialiser le détecteur
        
        Args:
            model_path: Chemin vers le dossier du modèle
        """
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.class_names = None
        
        self._load_model()
        self._load_metadata()
    
    def _load_model(self):
        """Charger le modèle TensorFlow"""
        model_file = os.path.join(self.model_path, "model.h5")
        
        if not os.path.exists(model_file):
            # Essayer avec SavedModel
            savedmodel_path = os.path.join(self.model_path, "saved_model")
            if os.path.exists(savedmodel_path):
                self.model = tf.keras.models.load_model(savedmodel_path)
            else:
                raise FileNotFoundError(f"Modèle non trouvé dans {self.model_path}")
        else:
            self.model = tf.keras.models.load_model(model_file)
        
        print(f"✓ Modèle chargé depuis {self.model_path}")
    
    def _load_metadata(self):
        """Charger les métadonnées du modèle"""
        metadata_file = os.path.join(self.model_path, "metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            
            # Extraire les noms de classes
            self.class_names = {int(k): v for k, v in self.metadata['classes'].items()}
            print(f"✓ Métadonnées chargées: {len(self.class_names)} classes")
        else:
            print("⚠ Métadonnées non trouvées, utilisation des classes par défaut")
            self.class_names = {}
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Prétraiter une image pour la prédiction
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image prétraitée
        """
        # Charger l'image
        img = Image.open(image_path)
        
        # Redimensionner
        img_height = self.metadata.get('img_height', 224)
        img_width = self.metadata.get('img_width', 224)
        img = img.resize((img_width, img_height))
        
        # Convertir en array et normaliser
        img_array = np.array(img) / 255.0
        
        # Ajouter dimension batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image_path: str, top_k: int = 3) -> List[Dict]:
        """
        Prédire la maladie sur une image
        
        Args:
            image_path: Chemin vers l'image
            top_k: Nombre de prédictions à retourner
            
        Returns:
            Liste des prédictions avec confiance
        """
        # Prétraiter l'image
        img_array = self.preprocess_image(image_path)
        
        # Prédire
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Obtenir les top-k prédictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            disease_name = self.class_names.get(idx, f"classe_{idx}")
            confidence = float(predictions[idx])
            
            results.append({
                'disease_id': f"disease_{idx}",
                'disease_name': disease_name,
                'confidence': confidence
            })
        
        return results
    
    def detect_disease(self, image_path: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Détecter une maladie avec informations complètes
        
        Args:
            image_path: Chemin vers l'image
            confidence_threshold: Seuil de confiance minimum
            
        Returns:
            Résultat de détection complet
        """
        # Obtenir les prédictions
        predictions = self.predict(image_path, top_k=3)
        
        # Meilleure prédiction
        best_prediction = predictions[0]
        
        # Déterminer la sévérité basée sur la confiance
        confidence = best_prediction['confidence']
        if confidence >= 0.9:
            severity = "Élevée"
        elif confidence >= 0.7:
            severity = "Modérée"
        else:
            severity = "Faible"
        
        # Construire le résultat
        result = {
            'disease_id': best_prediction['disease_id'],
            'disease_name': best_prediction['disease_name'],
            'confidence': confidence,
            'severity': severity,
            'alternative_diagnoses': predictions[1:],
            'treatments': self._get_treatments(best_prediction['disease_name']),
            'prevention_tips': self._get_prevention_tips(best_prediction['disease_name']),
            'affected_crop': self._extract_crop_name(best_prediction['disease_name'])
        }
        
        return result
    
    def _get_treatments(self, disease_name: str) -> List[Dict]:
        """
        Obtenir les traitements pour une maladie
        Base de données simplifiée, à étendre avec votre BDD
        """
        treatments_db = {
            'mildiou': [
                {
                    'treatment_id': 'trt_001',
                    'name': 'Fongicide biologique',
                    'description': 'Application de bouillie bordelaise',
                    'organic': True
                },
                {
                    'treatment_id': 'trt_002',
                    'name': 'Fongicide chimique',
                    'description': 'Traitement à base de mancozèbe',
                    'organic': False
                }
            ],
            'oidium': [
                {
                    'treatment_id': 'trt_003',
                    'name': 'Soufre',
                    'description': 'Pulvérisation de soufre mouillable',
                    'organic': True
                }
            ],
            'healthy': []
        }
        
        # Recherche par nom partiel
        disease_lower = disease_name.lower()
        for key, treatments in treatments_db.items():
            if key in disease_lower:
                return treatments
        
        # Traitement par défaut
        return [{
            'treatment_id': 'trt_generic',
            'name': 'Consultation recommandée',
            'description': 'Consultez un agronome pour un traitement adapté',
            'organic': True
        }]
    
    def _get_prevention_tips(self, disease_name: str) -> List[str]:
        """
        Obtenir les conseils de prévention
        """
        tips_db = {
            'mildiou': [
                "Assurer une bonne circulation d'air entre les plants",
                "Éviter l'arrosage par aspersion",
                "Retirer les feuilles infectées"
            ],
            'oidium': [
                "Espacer suffisamment les plants",
                "Éviter l'excès d'azote",
                "Utiliser des variétés résistantes"
            ],
            'healthy': [
                "Continuer les bonnes pratiques culturales",
                "Surveiller régulièrement les plants"
            ]
        }
        
        disease_lower = disease_name.lower()
        for key, tips in tips_db.items():
            if key in disease_lower:
                return tips
        
        return [
            "Maintenir une bonne hygiène culturale",
            "Surveiller régulièrement l'état des plants",
            "Consulter un expert en cas de doute"
        ]
    
    def _extract_crop_name(self, disease_name: str) -> str:
        """
        Extraire le nom de la culture du nom de la maladie
        """
        crops = ['tomate', 'pomme de terre', 'vigne', 'blé', 'maïs', 'riz']
        
        disease_lower = disease_name.lower()
        for crop in crops:
            if crop in disease_lower:
                return crop.capitalize()
        
        return "Non spécifié"


# ========================================
# Exemple d'utilisation
# ========================================

def example_usage():
    """
    Exemple d'utilisation du détecteur
    """
    # Initialiser le détecteur
    model_path = "models/agridetect_model_20250101_120000"  # Remplacer par votre chemin
    detector = DiseaseDetector(model_path)
    
    # Détecter une maladie
    image_path = "path/to/test/image.jpg"
    result = detector.detect_disease(image_path)
    
    # Afficher les résultats
    print(f"Maladie détectée: {result['disease_name']}")
    print(f"Confiance: {result['confidence']:.2%}")
    print(f"Sévérité: {result['severity']}")
    print(f"Culture affectée: {result['affected_crop']}")
    print(f"\nTraitements recommandés:")
    for treatment in result['treatments']:
        print(f"  - {treatment['name']}: {treatment['description']}")
    print(f"\nConseils de prévention:")
    for tip in result['prevention_tips']:
        print(f"  - {tip}")


if __name__ == "__main__":
    example_usage()
