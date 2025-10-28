from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from datetime import datetime
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI(
    title="AgriDetect API",
    description="API pour la détection des maladies des cultures",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic
class DiseaseDetectionResponse(BaseModel):
    disease_id: str
    disease_name: str
    confidence: float
    severity: str
    treatments: List[dict]
    prevention_tips: List[str]
    affected_crop: str
    detection_date: datetime

class TreatmentRecommendation(BaseModel):
    treatment_id: str
    name: str
    description: str
    application_method: str
    frequency: str
    precautions: List[str]
    organic: bool
    cost_estimate: str

class ChatMessage(BaseModel):
    message: str
    language: str = "fr"  # fr, wo (wolof), pu (pulaar)
    context: Optional[dict] = None

class CropAnalysis(BaseModel):
    crop_type: str
    location: Optional[dict] = None
    growth_stage: str
    symptoms: List[str]

class UserProfile(BaseModel):
    user_id: str
    name: str
    farm_location: dict
    preferred_language: str
    crops: List[str]
    farm_size: float

# Routes principales
@app.get("/")
async def root():
    return {
        "message": "Bienvenue sur AgriDetect API",
        "version": "1.0.0",
        "endpoints": {
            "disease_detection": "/api/v1/detect-disease",
            "chat": "/api/v1/chat",
            "treatments": "/api/v1/treatments",
            "crop_analysis": "/api/v1/analyze-crop"
        }
    }

@app.post("/api/v1/detect-disease", response_model=DiseaseDetectionResponse)
async def detect_disease(
    file: UploadFile = File(...),
    crop_type: Optional[str] = None
):
    """
    Détecte les maladies à partir d'une image de feuille
    """
    try:
        # Vérification du type de fichier
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")
        
        # Lecture et prétraitement de l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # TODO: Appeler le modèle CNN pour la détection
        # Pour le moment, réponse simulée
        response = DiseaseDetectionResponse(
            disease_id="leaf_blight_001",
            disease_name="Mildiou de la feuille",
            confidence=0.89,
            severity="Modérée",
            treatments=[
                {
                    "treatment_id": "trt_001",
                    "name": "Fongicide biologique",
                    "description": "Application de bouillie bordelaise",
                    "organic": True
                }
            ],
            prevention_tips=[
                "Assurer une bonne circulation d'air entre les plants",
                "Éviter l'arrosage par aspersion",
                "Retirer les feuilles infectées"
            ],
            affected_crop=crop_type or "Tomate",
            detection_date=datetime.now()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat")
async def chat_with_bot(message: ChatMessage):
    """
    Interface de chat multilingue pour assistance
    """
    try:
        # Détection de la langue et traitement
        language_map = {
            "fr": "Français",
            "wo": "Wolof",
            "pu": "Pulaar"
        }
        
        # TODO: Intégrer le modèle de langage multilingue
        # Réponse simulée pour le moment
        responses = {
            "fr": "Je comprends votre question. Voici mes recommandations...",
            "wo": "Maa ngi lay xamal. Lii mooy sama digal...",
            "pu": "Mi faami nde naamndaa. Ɗum ko mi waawi wallude..."
        }
        
        return {
            "response": responses.get(message.language, responses["fr"]),
            "language": language_map.get(message.language, "Français"),
            "suggestions": [
                "Comment prévenir cette maladie?",
                "Quels sont les symptômes?",
                "Traitement biologique disponible?"
            ],
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/treatments/{disease_id}")
async def get_treatments(disease_id: str):
    """
    Obtenir les traitements recommandés pour une maladie spécifique
    """
    # TODO: Récupérer de la base de données
    treatments = [
        {
            "treatment_id": "trt_001",
            "name": "Bouillie bordelaise",
            "description": "Fongicide à base de cuivre",
            "application_method": "Pulvérisation foliaire",
            "frequency": "Tous les 7-10 jours",
            "precautions": ["Porter des gants", "Éviter par temps venteux"],
            "organic": True,
            "cost_estimate": "5000 FCFA/hectare"
        },
        {
            "treatment_id": "trt_002",
            "name": "Neem Oil",
            "description": "Huile de neem naturelle",
            "application_method": "Pulvérisation",
            "frequency": "Hebdomadaire",
            "precautions": ["Application le soir"],
            "organic": True,
            "cost_estimate": "3000 FCFA/hectare"
        }
    ]
    
    return {"disease_id": disease_id, "treatments": treatments}

@app.post("/api/v1/analyze-crop")
async def analyze_crop(analysis: CropAnalysis):
    """
    Analyse complète de l'état d'une culture
    """
    return {
        "crop_type": analysis.crop_type,
        "health_status": "Attention requise",
        "risk_level": "Moyen",
        "recommendations": [
            "Surveillance accrue recommandée",
            "Traitement préventif conseillé",
            "Améliorer le drainage du sol"
        ],
        "next_inspection": "Dans 3 jours",
        "analysis_date": datetime.now()
    }

@app.post("/api/v1/user/profile")
async def create_user_profile(profile: UserProfile):
    """
    Créer ou mettre à jour le profil utilisateur
    """
    # TODO: Sauvegarder en base de données
    return {
        "status": "success",
        "user_id": profile.user_id,
        "message": "Profil créé avec succès"
    }

@app.get("/api/v1/diseases/common")
async def get_common_diseases(crop_type: Optional[str] = None):
    """
    Obtenir la liste des maladies communes
    """
    diseases = [
        {
            "id": "mld_001",
            "name": "Mildiou",
            "crops_affected": ["Tomate", "Pomme de terre", "Oignon"],
            "season": "Saison des pluies",
            "severity": "Élevée"
        },
        {
            "id": "rlt_002",
            "name": "Rouille",
            "crops_affected": ["Maïs", "Blé", "Haricot"],
            "season": "Toute l'année",
            "severity": "Modérée"
        },
        {
            "id": "anth_003",
            "name": "Anthracnose",
            "crops_affected": ["Mangue", "Papaye", "Avocat"],
            "season": "Saison humide",
            "severity": "Élevée"
        }
    ]
    
    if crop_type:
        diseases = [d for d in diseases if crop_type in d["crops_affected"]]
    
    return {"diseases": diseases, "total": len(diseases)}

@app.get("/api/v1/statistics/dashboard")
async def get_dashboard_stats():
    """
    Statistiques pour le tableau de bord
    """
    return {
        "total_detections": 1543,
        "diseases_detected": 23,
        "success_rate": 87.5,
        "active_users": 342,
        "crops_monitored": ["Tomate", "Oignon", "Maïs", "Mil", "Arachide"],
        "top_diseases": [
            {"name": "Mildiou", "count": 234},
            {"name": "Rouille", "count": 187},
            {"name": "Tache bactérienne", "count": 156}
        ],
        "period": "30 derniers jours"
    }

@app.post("/api/v1/feedback")
async def submit_feedback(
    detection_id: str,
    correct: bool,
    actual_disease: Optional[str] = None
):
    """
    Soumettre un feedback sur une détection
    """
    return {
        "status": "success",
        "message": "Merci pour votre retour",
        "feedback_id": f"fb_{datetime.now().timestamp()}"
    }

@app.get("/health")
async def health_check():
    """
    Vérification de l'état de l'API
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

