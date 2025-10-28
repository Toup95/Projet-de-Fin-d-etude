"""
Script de test rapide du modèle AgriDetect
"""

from model_predictor import DiseaseDetector
import os

# Charger le modèle
MODEL_PATH = "models/agridetect_model_20251028_125539"
print(f"🔧 Chargement du modèle depuis {MODEL_PATH}...")

detector = DiseaseDetector(MODEL_PATH)

print("✅ Modèle chargé avec succès!")
print(f"📊 Classes disponibles: {len(detector.class_names)}")
print()

# Afficher les classes
print("🌿 Classes de maladies détectables:")
for idx, name in detector.class_names.items():
    print(f"   {idx}: {name}")

print()
print("=" * 60)
print("Modèle prêt à détecter les maladies ! 🎉")
print("=" * 60)
print()

# Test avec une image (si disponible)
test_image_path = "data/test/Tomato_healthy/"
if os.path.exists(test_image_path):
    # Prendre la première image du dossier
    test_images = [f for f in os.listdir(test_image_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if test_images:
        test_img = os.path.join(test_image_path, test_images[0])
        print(f"🧪 Test avec l'image: {test_images[0]}")
        print()
        
        result = detector.detect_disease(test_img)
        
        print(f"🎯 Résultat de la détection:")
        print(f"   Maladie: {result['disease_name']}")
        print(f"   Confiance: {result['confidence']:.1%}")
        print(f"   Sévérité: {result['severity']}")
        print(f"   Culture: {result['affected_crop']}")
        print()
        
        print(f"💊 Traitements recommandés:")
        for treatment in result['treatments']:
            print(f"   - {treatment['name']}")
        print()
        
        print(f"🛡️ Conseils de prévention:")
        for tip in result['prevention_tips']:
            print(f"   - {tip}")

print()
print("✅ Test terminé avec succès!")
