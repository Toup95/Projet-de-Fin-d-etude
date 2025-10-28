from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import re

class MultilingualAgriChatbot:
    """
    Chatbot multilingue pour l'assistance agricole
    Supporte: Français, Wolof, Pulaar
    """
    
    def __init__(self):
        self.memory = ConversationBufferMemory()
        self.current_language = "fr"
        
        # Base de connaissances agricoles
        self.knowledge_base = {
            "diseases": self._load_disease_knowledge(),
            "treatments": self._load_treatment_knowledge(),
            "prevention": self._load_prevention_tips(),
            "crops": self._load_crop_info(),
            "seasons": self._load_seasonal_advice()
        }
        
        # Traductions des phrases communes
        self.translations = {
            "greetings": {
                "fr": "Bonjour! Comment puis-je vous aider avec vos cultures aujourd'hui?",
                "wo": "Asalaam aleykum! Nan laa la dimbali ci sa mbay tay?",
                "pu": "Jam tan! Hol ko mi waawi wallude e nder gese maa hannde?"
            },
            "disease_detected": {
                "fr": "J'ai détecté {disease} avec une confiance de {confidence}%",
                "wo": "Maa gis {disease} ak wóolu {confidence}%",
                "pu": "Mi yiɗi {disease} e goonga {confidence}%"
            },
            "treatment_recommendation": {
                "fr": "Je recommande le traitement suivant: {treatment}",
                "wo": "Maa digal ñi garab: {treatment}",
                "pu": "Miɗo waɗdi oo lekki: {treatment}"
            },
            "prevention_advice": {
                "fr": "Pour prévenir cette maladie:",
                "wo": "Ngir faggu feebar bi:",
                "pu": "Ngam haɗde ñawu oo:"
            },
            "need_more_info": {
                "fr": "Pouvez-vous me donner plus de détails sur {topic}?",
                "wo": "Ndax mën nga ma jox yeneen xibaar ci {topic}?",
                "pu": "Aɗa waawi hokku am ɓeyditte e {topic}?"
            },
            "confirmation": {
                "fr": "D'accord, je comprends.",
                "wo": "Waaw, xam naa.",
                "pu": "Eey, mi faamii."
            },
            "thank_you": {
                "fr": "Je vous en prie! N'hésitez pas si vous avez d'autres questions.",
                "wo": "Amul solo! Bul taar laajte yeneen laaj.",
                "pu": "Alaa nde caahu! Woto naamne ngam goɗɗe naamne."
            }
        }
        
        # Intents et patterns
        self.intent_patterns = {
            "greeting": r"(bonjour|salut|bonsoir|asalaam|jam|hello|hi)",
            "disease_inquiry": r"(maladie|feebar|ñawu|symptôme|tache|feuille|xob)",
            "treatment_request": r"(traitement|soigner|garab|lekki|médicament|fongicide)",
            "prevention_question": r"(prévenir|éviter|faggu|haɗde|protection)",
            "crop_info": r"(tomate|oignon|maïs|mil|arachide|mbay|ceeb|tigadega)",
            "weather_concern": r"(pluie|soleil|taw|naaj|temps|climat|saison)",
            "thanks": r"(merci|jërëjëf|a jaaraama|thank)",
            "goodbye": r"(au revoir|bye|ba beneen|haa yeeso|à bientôt)"
        }
    
    def _load_disease_knowledge(self) -> Dict:
        """Charge la base de connaissances sur les maladies"""
        return {
            "mildiou": {
                "fr": {
                    "name": "Mildiou",
                    "symptoms": ["Taches jaunes sur les feuilles", "Moisissure blanche au revers", "Flétrissement"],
                    "causes": "Champignon favorisé par l'humidité",
                    "affected_crops": ["Tomate", "Pomme de terre", "Oignon"]
                },
                "wo": {
                    "name": "Mildiou",
                    "symptoms": ["Tàkk yu mboq ci xob yi", "Puur yu weex ci ginnaaw xob bi", "Xob yi di wow"],
                    "causes": "Funŋus buy gënal ndox",
                    "affected_crops": ["Tomate", "Pomme de terre", "Soble"]
                },
                "pu": {
                    "name": "Mildiou",
                    "symptoms": ["Tache raneeje e leeɗe", "Huɗo peewo les leeɗe", "Leeɗe ɗe koosa"],
                    "causes": "Ñawu funŋus diga ndiyam",
                    "affected_crops": ["Tomate", "Pomme de terre", "Albasal"]
                }
            },
            "rouille": {
                "fr": {
                    "name": "Rouille",
                    "symptoms": ["Pustules orangées", "Jaunissement des feuilles", "Chute prématurée"],
                    "causes": "Champignon de type rouille",
                    "affected_crops": ["Maïs", "Blé", "Haricot"]
                },
                "wo": {
                    "name": "Xonq",
                    "symptoms": ["Poor yu xonq", "Xob yi di mboq", "Xob yi di daanu"],
                    "causes": "Funŋus xonq",
                    "affected_crops": ["Mbay", "Dugub", "Niébé"]
                }
            }
        }
    
    def _load_treatment_knowledge(self) -> Dict:
        """Charge la base de connaissances sur les traitements"""
        return {
            "organic": {
                "fr": {
                    "neem": "Huile de Neem - Pulvériser tous les 7 jours",
                    "copper": "Bouillie bordelaise - Application préventive",
                    "soap": "Savon noir dilué - Contre les insectes",
                    "ash": "Cendre de bois - Saupoudrer autour des plants"
                },
                "wo": {
                    "neem": "Diw Neem - Soppi ko 7 fan",
                    "copper": "Garab kuivre - Soppi ko balaa feebar bi",
                    "soap": "Saabun ñuul - Ngir gunóor yi",
                    "ash": "Tandarma - Saasal ko wër mbay mi"
                },
                "pu": {
                    "neem": "Neɓɓam Neem - Wurta nder 7 ñalawma",
                    "copper": "Lekki kuivre - Huutoro ko adii ñawu",
                    "soap": "Saabun ɓalewo - Fayde kuɓe",
                    "ash": "Toɓɓere - Wurta dow gese"
                }
            },
            "chemical": {
                "fr": {
                    "systemic": "Fongicide systémique - Suivre les doses recommandées",
                    "contact": "Fongicide de contact - Application foliaire",
                    "insecticide": "Insecticide - Respecter le délai avant récolte"
                }
            }
        }
    
    def _load_prevention_tips(self) -> Dict:
        """Charge les conseils de prévention"""
        return {
            "general": {
                "fr": [
                    "Rotation des cultures tous les 3 ans",
                    "Espacement adéquat entre les plants",
                    "Drainage du sol approprié",
                    "Élimination des plants malades",
                    "Utilisation de variétés résistantes"
                ],
                "wo": [
                    "Soppi mbay mi 3 at",
                    "Diggante bu baax ci mbay yi",
                    "Suuf si ndox du des",
                    "Dindi mbay yu feebar",
                    "Jëfandikoo mbay yu mën feebar yi"
                ],
                "pu": [
                    "Waylude gese nder 3 duuɓi",
                    "Seedto gese fof",
                    "Ƴeew ndiyam e leydi",
                    "Ittu gese dogataake",
                    "Huutoro gese tammitooje"
                ]
            },
            "seasonal": {
                "rainy": {
                    "fr": "Pendant la saison des pluies, augmenter l'espacement et améliorer le drainage",
                    "wo": "Ci jamono taw bi, yokk diggante gi te baaxal ndox bi di génne",
                    "pu": "E ndungu, ɓeydu seedgol e moƴƴin ndiyam"
                },
                "dry": {
                    "fr": "En saison sèche, arroser tôt le matin ou tard le soir",
                    "wo": "Ci jamono tanqaay bi, tëj teel ci suba walla ngoon",
                    "pu": "E ceeɗu, wurin subaka walla kiikiiɗe"
                }
            }
        }
    
    def _load_crop_info(self) -> Dict:
        """Charge les informations sur les cultures"""
        return {
            "tomato": {
                "fr": {"name": "Tomate", "cycle": "90-120 jours", "water": "Régulier"},
                "wo": {"name": "Tomate", "cycle": "90-120 fan", "water": "Tëj ko saa yu nekk"},
                "pu": {"name": "Tomate", "cycle": "90-120 ñalawma", "water": "Wurin sahaa fof"}
            },
            "onion": {
                "fr": {"name": "Oignon", "cycle": "120-150 jours", "water": "Modéré"},
                "wo": {"name": "Soble", "cycle": "120-150 fan", "water": "Tëj ko ndank"},
                "pu": {"name": "Albasal", "cycle": "120-150 ñalawma", "water": "Wurin seeɗa"}
            },
            "maize": {
                "fr": {"name": "Maïs", "cycle": "80-100 jours", "water": "Important au début"},
                "wo": {"name": "Mbay", "cycle": "80-100 fan", "water": "Ndox bu bari ci tàmbali"},
                "pu": {"name": "Gawri", "cycle": "80-100 ñalawma", "water": "Ndiyam heewi fuɗɗoode"}
            }
        }
    
    def _load_seasonal_advice(self) -> Dict:
        """Charge les conseils saisonniers"""
        return {
            "planting": {
                "fr": "Meilleure période de plantation: début de la saison des pluies",
                "wo": "Jamono bu gën ci jël: tàmbalit taw bi",
                "pu": "Sahaa moƴƴo hokkude: fuɗɗoode ndungu"
            },
            "harvest": {
                "fr": "Récolter tôt le matin quand il fait frais",
                "wo": "Góob teel ci suba bu sedd bi",
                "pu": "Roƴƴo subaka nde yahdi woni"
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Détecte la langue du message"""
        # Mots-clés spécifiques à chaque langue
        wolof_keywords = ["nanga", "def", "lan", "maa", "ngi", "dëgg", "waaw", "déedéet", "ñaata", "ki"]
        pulaar_keywords = ["ko", "hol", "mi", "ɗo", "ɗum", "nde", "ɓe", "kam", "hannde", "naamne"]
        
        text_lower = text.lower()
        
        # Compter les mots de chaque langue
        wolof_count = sum(1 for word in wolof_keywords if word in text_lower)
        pulaar_count = sum(1 for word in pulaar_keywords if word in text_lower)
        
        if wolof_count > pulaar_count and wolof_count > 0:
            return "wo"
        elif pulaar_count > wolof_count and pulaar_count > 0:
            return "pu"
        else:
            return "fr"  # Français par défaut
    
    def detect_intent(self, text: str) -> str:
        """Détecte l'intention du message"""
        text_lower = text.lower()
        
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text_lower):
                return intent
        
        return "general"
    
    def generate_response(self, message: str, context: Optional[Dict] = None) -> Dict:
        """
        Génère une réponse au message de l'utilisateur
        """
        # Détecter la langue
        detected_language = self.detect_language(message)
        self.current_language = detected_language
        
        # Détecter l'intention
        intent = self.detect_intent(message)
        
        # Préparer le contexte
        if context is None:
            context = {}
        
        # Générer la réponse selon l'intention
        response = self._handle_intent(intent, message, context)
        
        # Ajouter à la mémoire
        self.memory.chat_memory.add_user_message(message)
        self.memory.chat_memory.add_ai_message(response["text"])
        
        # Préparer les suggestions
        suggestions = self._get_suggestions(intent, detected_language)
        
        return {
            "response": response["text"],
            "language": detected_language,
            "intent": intent,
            "suggestions": suggestions,
            "context": response.get("context", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_intent(self, intent: str, message: str, context: Dict) -> Dict:
        """Gère les différentes intentions"""
        lang = self.current_language
        
        if intent == "greeting":
            return {"text": self.translations["greetings"][lang]}
        
        elif intent == "disease_inquiry":
            # Chercher des informations sur les maladies mentionnées
            disease_info = self._extract_disease_info(message, lang)
            if disease_info:
                return {"text": disease_info, "context": {"topic": "disease"}}
            else:
                return {"text": self.translations["need_more_info"][lang].format(topic="les symptômes")}
        
        elif intent == "treatment_request":
            # Proposer des traitements
            treatment_info = self._get_treatment_recommendations(context.get("disease"), lang)
            return {"text": treatment_info, "context": {"topic": "treatment"}}
        
        elif intent == "prevention_question":
            # Donner des conseils de prévention
            prevention_tips = self._get_prevention_tips(lang)
            return {"text": prevention_tips, "context": {"topic": "prevention"}}
        
        elif intent == "crop_info":
            # Informations sur les cultures
            crop_info = self._extract_crop_info(message, lang)
            return {"text": crop_info, "context": {"topic": "crop"}}
        
        elif intent == "thanks":
            return {"text": self.translations["thank_you"][lang]}
        
        elif intent == "goodbye":
            goodbye_messages = {
                "fr": "Au revoir et bonne culture! 🌱",
                "wo": "Ba beneen yoon! Mbay bu baax! 🌱",
                "pu": "Haa yeeso! Gese moƴƴe! 🌱"
            }
            return {"text": goodbye_messages[lang]}
        
        else:
            # Réponse générale
            return self._generate_general_response(message, lang)
    
    def _extract_disease_info(self, message: str, lang: str) -> str:
        """Extrait et retourne les informations sur les maladies"""
        message_lower = message.lower()
        
        for disease_key, disease_data in self.knowledge_base["diseases"].items():
            if disease_key in message_lower or any(symptom.lower() in message_lower 
                                                  for symptom in disease_data.get(lang, disease_data["fr"])["symptoms"]):
                info = disease_data.get(lang, disease_data["fr"])
                response = f"📋 {info['name']}:\n\n"
                response += f"🔍 Symptômes: {', '.join(info['symptoms'])}\n"
                response += f"⚠️ Cause: {info['causes']}\n"
                response += f"🌱 Cultures affectées: {', '.join(info['affected_crops'])}"
                return response
        
        return ""
    
    def _get_treatment_recommendations(self, disease: Optional[str], lang: str) -> str:
        """Génère des recommandations de traitement"""
        treatments = self.knowledge_base["treatments"]["organic"].get(lang, 
                                                                     self.knowledge_base["treatments"]["organic"]["fr"])
        
        response = self.translations["treatment_recommendation"][lang].format(treatment="") + "\n\n"
        response += "🌿 Traitements biologiques:\n"
        
        for key, treatment in list(treatments.items())[:3]:
            response += f"• {treatment}\n"
        
        return response
    
    def _get_prevention_tips(self, lang: str) -> str:
        """Génère des conseils de prévention"""
        tips = self.knowledge_base["prevention"]["general"].get(lang, 
                                                               self.knowledge_base["prevention"]["general"]["fr"])
        
        response = self.translations["prevention_advice"][lang] + "\n\n"
        for tip in tips[:5]:
            response += f"✓ {tip}\n"
        
        return response
    
    def _extract_crop_info(self, message: str, lang: str) -> str:
        """Extrait les informations sur les cultures"""
        message_lower = message.lower()
        
        for crop_key, crop_data in self.knowledge_base["crops"].items():
            crop_info = crop_data.get(lang, crop_data["fr"])
            if crop_info["name"].lower() in message_lower:
                response = f"🌱 {crop_info['name']}:\n"
                response += f"📅 Cycle: {crop_info['cycle']}\n"
                response += f"💧 Arrosage: {crop_info['water']}"
                return response
        
        return "Information non trouvée pour cette culture."
    
    def _generate_general_response(self, message: str, lang: str) -> Dict:
        """Génère une réponse générale"""
        general_responses = {
            "fr": "Je suis là pour vous aider avec vos questions agricoles. Que souhaitez-vous savoir?",
            "wo": "Maa ngi fi ngir dimbali la ci sa laaj yu mbay. Lan nga bëgg xam?",
            "pu": "Miɗo ɗoo ngam wallude e naamne maa gese. Hol ko njiɗ-ɗaa faamde?"
        }
        
        return {"text": general_responses[lang]}
    
    def _get_suggestions(self, intent: str, lang: str) -> List[str]:
        """Génère des suggestions contextuelles"""
        suggestions_map = {
            "fr": {
                "greeting": ["Détecter une maladie", "Conseils de prévention", "Traitements biologiques"],
                "disease_inquiry": ["Voir les traitements", "Mesures préventives", "Cultures affectées"],
                "treatment_request": ["Dosage recommandé", "Alternatives biologiques", "Précautions"],
                "prevention_question": ["Calendrier cultural", "Rotation des cultures", "Variétés résistantes"],
                "general": ["Aide sur les maladies", "Conseils saisonniers", "Guide de culture"]
            },
            "wo": {
                "greeting": ["Xool feebar", "Digal ci faggu", "Garab yu naturel"],
                "disease_inquiry": ["Xool garab yi", "Faggu feebar bi", "Mbay yu mën feebar"],
                "treatment_request": ["Ñaata laa war a jël", "Yeneen garab", "Moytu yi"],
                "general": ["Ndimbal ci feebar yi", "Digal ci jamono", "Yoon wi ngir mbay"]
            },
            "pu": {
                "greeting": ["Yiɗ ñawu", "Ballal haɗde", "Lekki safrooɗi"],
                "disease_inquiry": ["Yiɗ lekki", "Haɗde ñawu", "Gese dogataake"],
                "treatment_request": ["No foti huutoraade", "Goɗɗe lekki", "Reentaade"],
                "general": ["Ballal e ñawɗe", "Ballal sahaa", "Ardude gese"]
            }
        }
        
        default_suggestions = suggestions_map.get(lang, suggestions_map["fr"])
        return default_suggestions.get(intent, default_suggestions["general"])
    
    def reset_conversation(self):
        """Réinitialise la conversation"""
        self.memory.clear()
        self.current_language = "fr"

# Fonction de test
def test_chatbot():
    """Test du chatbot multilingue"""
    chatbot = MultilingualAgriChatbot()
    
    # Tests en différentes langues
    test_messages = [
        "Bonjour, j'ai des taches sur mes tomates",
        "Asalaam aleykum, sama tomate yi am na tàkk",
        "Hol ko mi waawi wallude e albasal am?",
        "Comment traiter le mildiou?",
        "Merci pour votre aide"
    ]
    
    print("=== Test du Chatbot Multilingue ===\n")
    
    for message in test_messages:
        print(f"👤 User: {message}")
        response = chatbot.generate_response(message)
        print(f"🤖 Bot [{response['language']}]: {response['response']}")
        print(f"💡 Suggestions: {', '.join(response['suggestions'][:2])}")
        print("-" * 50)

if __name__ == "__main__":
    test_chatbot()
