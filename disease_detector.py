import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from PIL import Image
import json
import os
from typing import Tuple, List, Dict

class PlantDiseaseDetector:
    """
    Modèle CNN pour la détection des maladies des plantes
    Utilise MobileNetV2 pour l'efficacité sur mobile
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.class_names = []
        self.image_size = (224, 224)
        self.model_path = model_path
        
        # Dictionnaire des maladies avec traductions
        self.disease_info = {
            "healthy": {
                "fr": "Sain",
                "wo": "Wér",
                "pu": "Cellal",
                "severity": "Aucune",
                "treatments": []
            },
            "leaf_blight": {
                "fr": "Mildiou de la feuille",
                "wo": "Feebar xob bi",
                "pu": "Ñawu leeɗe",
                "severity": "Élevée",
                "treatments": ["fungicide_copper", "neem_oil"]
            },
            "bacterial_spot": {
                "fr": "Tache bactérienne",
                "wo": "Tàkk bakteriya",
                "pu": "Tache bakteriya",
                "severity": "Modérée",
                "treatments": ["copper_spray", "crop_rotation"]
            },
            "rust": {
                "fr": "Rouille",
                "wo": "Xonq",
                "pu": "Dila",
                "severity": "Modérée",
                "treatments": ["fungicide_systemic", "resistant_varieties"]
            },
            "powdery_mildew": {
                "fr": "Oïdium",
                "wo": "Puur weex",
                "pu": "Huɗo peewo",
                "severity": "Faible à Modérée",
                "treatments": ["sulfur_spray", "baking_soda_solution"]
            },
            "mosaic_virus": {
                "fr": "Virus de la mosaïque",
                "wo": "Wirùs mosayik",
                "pu": "Virus mosaïque",
                "severity": "Élevée",
                "treatments": ["remove_infected", "insecticide_vectors"]
            }
        }
        
        # Traitements disponibles
        self.treatments_db = {
            "fungicide_copper": {
                "fr": {
                    "name": "Bouillie bordelaise",
                    "description": "Fongicide à base de cuivre efficace contre les champignons",
                    "application": "Pulvériser tous les 7-10 jours",
                    "organic": True,
                    "cost": "5000 FCFA/ha"
                },
                "wo": {
                    "name": "Garab yu kuivre",
                    "description": "Garab buy fagliku funŋus yi",
                    "application": "Soppi ko 7-10 fan",
                    "organic": True,
                    "cost": "5000 FCFA/ha"
                }
            },
            "neem_oil": {
                "fr": {
                    "name": "Huile de Neem",
                    "description": "Insecticide et fongicide naturel",
                    "application": "Application hebdomadaire le soir",
                    "organic": True,
                    "cost": "3000 FCFA/ha"
                },
                "wo": {
                    "name": "Diw Neem",
                    "description": "Garab buy fagliku gunóor ak funŋus",
                    "application": "Soppi ko bépp ayubés ci ngoon",
                    "organic": True,
                    "cost": "3000 FCFA/ha"
                }
            },
            "crop_rotation": {
                "fr": {
                    "name": "Rotation des cultures",
                    "description": "Alterner les cultures pour briser le cycle des maladies",
                    "application": "Planifier sur 3-4 saisons",
                    "organic": True,
                    "cost": "Minimal"
                }
            }
        }
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.build_model()
    
    def build_model(self, num_classes: int = 10):
        """
        Construit le modèle CNN basé sur MobileNetV2
        """
        # Base model - MobileNetV2 pré-entraîné
        base_model = MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Geler les couches de base
        base_model.trainable = False
        
        # Ajouter des couches personnalisées
        inputs = keras.Input(shape=(*self.image_size, 3))
        
        # Augmentation des données
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ])
        
        x = data_augmentation(inputs)
        
        # Normalisation
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Pooling et couches denses
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Couche de sortie
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compilation
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        return self.model
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Prétraite une image pour la prédiction
        """
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
            
        # Redimensionner
        image = image.resize(self.image_size)
        
        # Convertir en array numpy
        img_array = np.array(image)
        
        # Ajouter dimension batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normaliser
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        return img_array
    
    def predict(self, image_path: str, language: str = "fr") -> Dict:
        """
        Prédit la maladie à partir d'une image
        """
        # Prétraitement
        img_array = self.preprocess_image(image_path)
        
        # Prédiction
        predictions = self.model.predict(img_array)
        
        # Obtenir la classe prédite
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Mapper vers le nom de la maladie
        disease_key = self.class_names[predicted_class_idx] if self.class_names else "unknown"
        
        # Obtenir les informations de la maladie
        disease_data = self.disease_info.get(disease_key, self.disease_info["healthy"])
        
        # Préparer les traitements
        treatments = []
        for treatment_key in disease_data.get("treatments", []):
            treatment = self.treatments_db.get(treatment_key, {})
            if language in treatment:
                treatments.append(treatment[language])
        
        # Construire la réponse
        result = {
            "disease_key": disease_key,
            "disease_name": disease_data.get(language, disease_data.get("fr")),
            "confidence": confidence,
            "severity": disease_data["severity"],
            "treatments": treatments,
            "top_3_predictions": self.get_top_predictions(predictions[0], language),
            "requires_action": disease_key != "healthy"
        }
        
        return result
    
    def get_top_predictions(self, predictions: np.ndarray, language: str = "fr", top_k: int = 3) -> List[Dict]:
        """
        Obtient les top K prédictions
        """
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            disease_key = self.class_names[idx] if idx < len(self.class_names) else "unknown"
            disease_data = self.disease_info.get(disease_key, {"fr": "Inconnu"})
            
            top_predictions.append({
                "disease": disease_data.get(language, disease_data.get("fr")),
                "confidence": float(predictions[idx]),
                "severity": disease_data.get("severity", "Inconnue")
            })
        
        return top_predictions
    
    def train(self, train_dir: str, val_dir: str, epochs: int = 30):
        """
        Entraîne le modèle sur un dataset
        """
        # Générateurs de données
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Chargement des données
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=32,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.image_size,
            batch_size=32,
            class_mode='categorical'
        )
        
        # Sauvegarder les noms de classes
        self.class_names = list(train_generator.class_indices.keys())
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Entraînement
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, path: str):
        """
        Sauvegarde le modèle et les métadonnées
        """
        # Sauvegarder le modèle
        self.model.save(f"{path}/model.h5")
        
        # Sauvegarder les métadonnées
        metadata = {
            "class_names": self.class_names,
            "image_size": self.image_size,
            "disease_info": self.disease_info
        }
        
        with open(f"{path}/metadata.json", 'w') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def load_model(self, path: str):
        """
        Charge un modèle sauvegardé
        """
        # Charger le modèle
        self.model = keras.models.load_model(f"{path}/model.h5")
        
        # Charger les métadonnées
        with open(f"{path}/metadata.json", 'r') as f:
            metadata = json.load(f)
            self.class_names = metadata["class_names"]
            self.image_size = tuple(metadata["image_size"])
            self.disease_info = metadata["disease_info"]
    
    def fine_tune(self, epochs: int = 10):
        """
        Fine-tuning du modèle en dégelant certaines couches
        """
        # Dégeler les dernières couches du modèle de base
        base_model = self.model.layers[4]  # Récupérer le base_model
        base_model.trainable = True
        
        # Geler toutes les couches sauf les 20 dernières
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompiler avec un taux d'apprentissage plus faible
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

# Fonction utilitaire pour tester le modèle
def test_model():
    """
    Fonction de test du modèle
    """
    detector = PlantDiseaseDetector()
    
    # Simuler une prédiction
    test_image = np.random.rand(224, 224, 3) * 255
    test_image = Image.fromarray(test_image.astype('uint8'))
    
    # Initialiser les noms de classes pour le test
    detector.class_names = ["healthy", "leaf_blight", "bacterial_spot", "rust"]
    
    result = detector.predict(test_image, language="fr")
    print("Résultat de détection:", json.dumps(result, indent=2, ensure_ascii=False))
    
    # Test en wolof
    result_wo = detector.predict(test_image, language="wo")
    print("\nRésultat en Wolof:", json.dumps(result_wo, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_model()
