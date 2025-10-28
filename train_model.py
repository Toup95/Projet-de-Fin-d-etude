#!/usr/bin/env python3
"""
Script d'entraînement du modèle AgriDetect
Détection de maladies des cultures par Deep Learning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json

# ========================================
# Configuration
# ========================================

class Config:
    """Configuration pour l'entraînement"""
    
    # Chemins des données
    DATA_DIR = "data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "validation")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    # Chemins de sortie
    OUTPUT_DIR = "models"
    MODEL_NAME = f"agridetect_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Paramètres du modèle
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Architecture
    USE_PRETRAINED = True  # Utiliser transfer learning
    PRETRAINED_MODEL = "MobileNetV2"  # Options: MobileNetV2, EfficientNetB0
    FREEZE_LAYERS = True  # Geler les couches pré-entraînées au début
    
    # Augmentation des données
    AUGMENTATION = True
    ROTATION_RANGE = 40
    WIDTH_SHIFT_RANGE = 0.2
    HEIGHT_SHIFT_RANGE = 0.2
    SHEAR_RANGE = 0.2
    ZOOM_RANGE = 0.2
    HORIZONTAL_FLIP = True
    FILL_MODE = 'nearest'


# ========================================
# Préparation des Données
# ========================================

def create_data_generators(config):
    """
    Créer les générateurs de données avec augmentation
    """
    print("📊 Création des générateurs de données...")
    
    if config.AUGMENTATION:
        # Générateur pour l'entraînement avec augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=config.ROTATION_RANGE,
            width_shift_range=config.WIDTH_SHIFT_RANGE,
            height_shift_range=config.HEIGHT_SHIFT_RANGE,
            shear_range=config.SHEAR_RANGE,
            zoom_range=config.ZOOM_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            fill_mode=config.FILL_MODE
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Générateur pour validation (sans augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Chargement des données
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Sauvegarder les classes
    class_indices = train_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    
    print(f"✓ Classes détectées: {list(class_indices.keys())}")
    print(f"✓ Nombre d'images d'entraînement: {train_generator.samples}")
    print(f"✓ Nombre d'images de validation: {validation_generator.samples}")
    
    return train_generator, validation_generator, class_names


# ========================================
# Construction du Modèle
# ========================================

def build_model(config, num_classes):
    """
    Construire le modèle de détection
    """
    print(f"🔧 Construction du modèle avec {config.PRETRAINED_MODEL}...")
    
    if config.USE_PRETRAINED:
        # Utiliser un modèle pré-entraîné (Transfer Learning)
        if config.PRETRAINED_MODEL == "MobileNetV2":
            base_model = MobileNetV2(
                input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                include_top=False,
                weights='imagenet'
            )
        elif config.PRETRAINED_MODEL == "EfficientNetB0":
            base_model = EfficientNetB0(
                input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS),
                include_top=False,
                weights='imagenet'
            )
        
        # Geler les couches du modèle de base
        if config.FREEZE_LAYERS:
            base_model.trainable = False
            print("✓ Couches pré-entraînées gelées")
        
        # Créer le modèle complet
        inputs = keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
    else:
        # Créer un modèle CNN depuis zéro
        model = keras.Sequential([
            layers.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
    
    print("✓ Modèle construit avec succès")
    return model


# ========================================
# Compilation et Entraînement
# ========================================

def compile_and_train(model, train_gen, val_gen, config, class_names):
    """
    Compiler et entraîner le modèle
    """
    print("🚀 Compilation du modèle...")
    
    # Compiler le modèle
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    print(model.summary())
    
    # Callbacks
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    checkpoint_path = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "checkpoint.h5")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"🎯 Début de l'entraînement pour {config.EPOCHS} epochs...")
    
    # Entraîner
    history = model.fit(
        train_gen,
        epochs=config.EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    print("✓ Entraînement terminé!")
    
    # Sauvegarder l'historique
    history_path = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "history.json")
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']]
        }, f, indent=2)
    
    return history


# ========================================
# Fine-tuning (optionnel)
# ========================================

def fine_tune_model(model, train_gen, val_gen, config):
    """
    Fine-tuner le modèle en dégelant certaines couches
    """
    if not config.USE_PRETRAINED or not config.FREEZE_LAYERS:
        return model
    
    print("🔧 Fine-tuning du modèle...")
    
    # Dégeler les dernières couches
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Geler seulement les premières couches
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompiler avec un taux d'apprentissage plus faible
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraîner quelques epochs supplémentaires
    print("🎯 Fine-tuning en cours...")
    history_fine = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        verbose=1
    )
    
    print("✓ Fine-tuning terminé!")
    return model


# ========================================
# Évaluation et Sauvegarde
# ========================================

def evaluate_and_save(model, val_gen, config, class_names, history):
    """
    Évaluer le modèle et sauvegarder
    """
    print("📊 Évaluation du modèle...")
    
    # Évaluer
    results = model.evaluate(val_gen)
    print(f"✓ Loss: {results[0]:.4f}")
    print(f"✓ Accuracy: {results[1]:.4f}")
    if len(results) > 2:
        print(f"✓ Top-3 Accuracy: {results[2]:.4f}")
    
    # Sauvegarder le modèle complet
    model_path = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "model.h5")
    model.save(model_path)
    print(f"✓ Modèle sauvegardé: {model_path}")
    
    # Sauvegarder aussi au format SavedModel (pour production)
    savedmodel_path = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "saved_model")
    model.save(savedmodel_path, save_format='tf')
    print(f"✓ SavedModel sauvegardé: {savedmodel_path}")
    
    # Sauvegarder les métadonnées
    metadata = {
        'model_name': config.MODEL_NAME,
        'classes': class_names,
        'num_classes': len(class_names),
        'img_height': config.IMG_HEIGHT,
        'img_width': config.IMG_WIDTH,
        'accuracy': float(results[1]),
        'loss': float(results[0]),
        'training_date': datetime.now().isoformat(),
        'architecture': config.PRETRAINED_MODEL if config.USE_PRETRAINED else 'Custom CNN',
        'epochs': config.EPOCHS
    }
    
    metadata_path = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Métadonnées sauvegardées: {metadata_path}")
    
    # Visualiser les courbes d'apprentissage
    plot_training_history(history, config)
    
    return results


# ========================================
# Visualisation
# ========================================

def plot_training_history(history, config):
    """
    Créer des graphiques de l'historique d'entraînement
    """
    print("📈 Création des graphiques...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    plot_path = os.path.join(config.OUTPUT_DIR, config.MODEL_NAME, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graphiques sauvegardés: {plot_path}")
    plt.close()


# ========================================
# Main - Pipeline Complet
# ========================================

def main():
    """
    Pipeline complet d'entraînement
    """
    print("=" * 60)
    print("🌾 AgriDetect - Entraînement du Modèle")
    print("=" * 60)
    print()
    
    # Configuration
    config = Config()
    
    # Vérifier que les données existent
    if not os.path.exists(config.TRAIN_DIR):
        print(f"❌ Erreur: Le dossier {config.TRAIN_DIR} n'existe pas!")
        print("Veuillez créer la structure de données requise.")
        return
    
    # 1. Préparer les données
    train_gen, val_gen, class_names = create_data_generators(config)
    num_classes = len(class_names)
    
    print()
    
    # 2. Construire le modèle
    model = build_model(config, num_classes)
    
    print()
    
    # 3. Entraîner
    history = compile_and_train(model, train_gen, val_gen, config, class_names)
    
    print()
    
    # 4. Fine-tuning (optionnel)
    # model = fine_tune_model(model, train_gen, val_gen, config)
    
    print()
    
    # 5. Évaluer et sauvegarder
    evaluate_and_save(model, val_gen, config, class_names, history)
    
    print()
    print("=" * 60)
    print("✅ Entraînement terminé avec succès!")
    print(f"📁 Modèle sauvegardé dans: {os.path.join(config.OUTPUT_DIR, config.MODEL_NAME)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
