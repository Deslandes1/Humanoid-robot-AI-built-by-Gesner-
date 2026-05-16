import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import hashlib
import re
import base64
import csv
import io
import os
import shutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# ========== DATA DIRECTORY ==========
DATA_DIR = ".gesner_data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
DICT_FILE = os.path.join(DATA_DIR, "dictionaries.json")
VOICE_FILE = os.path.join(DATA_DIR, "voice_cache.json")

# ---------- PERSISTENCE FUNCTIONS ----------
def save_training_data():
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.training_data, f, ensure_ascii=False, indent=2)

def load_training_data():
    if os.path.exists(TRAINING_FILE):
        with open(TRAINING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_dictionaries():
    with open(DICT_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.dictionaries, f, ensure_ascii=False, indent=2)

def load_dictionaries():
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"ht": {}, "fr": {}, "en": {}}

def save_voice_cache():
    serializable = {}
    for key, audio_bytes in VOICE_CACHE.items():
        serializable[key] = base64.b64encode(audio_bytes).decode("utf-8")
    with open(VOICE_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)

def load_voice_cache():
    if os.path.exists(VOICE_FILE):
        with open(VOICE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = {}
        for key, b64 in data.items():
            cache[key] = base64.b64decode(b64)
        return cache
    return {}

# ---------- DEFAULT TRAINING FACTS ----------
def get_default_training_facts():
    return [
        "Ti Malice se yon lojisyèl edikatif ki anseye timoun yo Kreyòl Ayisyen atravè jwèt ak istwa.",
        "Ti Malice gen yon liv ki rele 'Ti Malice aprann Kreyòl' ki gen 12 chapit.",
        "Chapit 1 Ti Malice: Alfabè kreyòl la ak pwononsyasyon.",
        "Chapit 2 Ti Malice: Nonm 1 rive 100 an Kreyòl.",
        "Chapit 3 Ti Malice: Koulè ak fòm an Kreyòl.",
        "Chapit 4 Ti Malice: Fanmi ak zanmi.",
        "Chapit 5 Ti Malice: Manje Ayisyen.",
        "Chapit 6 Ti Malice: Bèt ak natir.",
        "Chapit 7 Ti Malice: Vèb ki pi komen yo.",
        "Chapit 8 Ti Malice: Tan pase, tan prezan, tan kap vini.",
        "Chapit 9 Ti Malice: Fraz senp.",
        "Chapit 10 Ti Malice: Konvèsasyon chak jou.",
        "Chapit 11 Ti Malice: Pwovèb ak ekspresyon Kreyòl.",
        "Chapit 12 Ti Malice: Istwa kout pou li.",
        "Ti Malice gen yon seksyon egzèsis ki gen 50 kesyon pou pratike.",
        "Ou ka telechaje Ti Malice sou sitwèb globalinternet.py.",
        "Ti Malice fèt pa Gesner Deslandes pou ede timoun Ayisyen aprann Kreyòl fasilman.",
        "Alfabè kreyòl la gen 32 lèt.",
        "Pwonon pèsonèl an Kreyòl: Mwen, ou, li, nou, yo.",
        "Vèb 'se' (to be) nan prezan: Mwen se, ou se, li se, nou se, yo se.",
        "Vèb 'gen' (to have) nan prezan: Mwen gen, ou gen, li gen, nou gen, yo gen.",
        "Salitasyon debaz: Bonjou (Bondye), Bonswa (Aswe), Kijan ou rele? (Ki jan ou rele?), Mwen rele...",
        "Kesyon debaz: Kijan ou ye? (How are you?), Mwen byen (I'm fine), Mèsi (Thank you), Pa dekwa (You're welcome).",
        "Nonm 1-10: youn, de, twa, kat, senk, sis, sèt, uit, nèf, dis.",
        "Koulè debaz: wouj (red), ble (blue), vèt (green), jòn (yellow), nwa (black), blan (white).",
        "Tan pase (past tense): yo itilize 'te' devan vèb. Egzanp: Mwen te manje (I ate).",
        "Tan kap vini (future tense): yo itilize 'ap' oswa 'pral'. Egzanp: Mwen ap manje (I will eat).",
        "Nègasyon (negation): yo itilize 'pa' apre vèb. Egzanp: Mwen pa manje (I don't eat).",
        "Pwopozisyon komen: nan (in), sou (on), anba (under), devan (in front of), dèyè (behind), bò (beside).",
        "Fraz konplèks: Itilize 'ki' (that/which), 'kote' (where), 'poukisa' (why).",
        "Vèb modèl: vle (to want), kapab (can), dwe (must), konnen (to know), fè (to do/make).",
        "Pawòl konpoze (compound words): pote + chay = potechay (backpack), bwa + chemen = bwachemen (forest path).",
        "Pwovèb Kreyòl popilè: 'Dèyè mòn gen mòn' (Beyond mountains there are mountains - life is full of challenges).",
        "Pwovèb: 'Men anpil, chay pa lou' (Many hands make light work).",
        "Pwovèb: 'Ti ponyen fè gwo chay' (Little by little, big load is carried).",
        "Anplwaye tan ki konpoze: Mwen te ap manje (I was eating).",
        "Vwa pasif: Liv la te ekri pa Jan (The book was written by John).",
        "Sijonktif (subjunctive): Fòk ou vini (You must come).",
        "Liteati kreyòl: ekriven tankou Frankétienne, Gary Victor, ak Lyonel Trouillot.",
        "Diferans ant Kreyòl Ayisyen ak Kreyòl Matinik oswa Giyàn.",
        "Analiz powèm Kreyòl: 'Kreyon mwen' pa Gesner Deslandes.",
        "Rédaksyon avançée: kijan pou ekri yon lèt fòmèl an Kreyòl."
    ]

def initialize_default_training():
    if not st.session_state.training_data:
        default_facts = get_default_training_facts()
        for fact in default_facts:
            if fact.strip():
                embedding = st.session_state.embedding_model.encode([fact])[0]
                st.session_state.training_data.append({"text": fact, "embedding": embedding.tolist()})
        rebuild_index()
        save_training_data()

# ---------- STREAMLIT PAGE CONFIG ----------
st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

# ---------- CSS ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
        border-right: 2px solid #e94560;
    }
    .stMarkdown, .stTextInput label, .stTextArea label, .stSelectbox label, .stButton button, .stCaption,
    h1, h2, h3, h4, h5, h6, p, li, div, span, strong, em, .footer,
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .stSelectbox {
        background-color: #000000 !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background-color: #000000 !important;
        border: 1px solid #e94560 !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: #e94560 !important;
        stroke: #e94560 !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: #000000 !important;
        border: 1px solid #e94560 !important;
    }
    div[data-baseweb="popover"] li {
        color: white !important;
        background-color: #000000 !important;
    }
    div[data-baseweb="popover"] li:hover {
        background-color: #e94560 !important;
        color: white !important;
    }
    .stButton button {
        background-color: #e94560 !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        border: none;
    }
    .chat-row .stButton button {
        background-color: #ffaa33 !important;
        padding: 0px 8px !important;
        border-radius: 20px !important;
        font-size: 1rem !important;
        width: auto !important;
        min-width: 40px;
    }
    .chat-row .stButton button:hover {
        background-color: #ffcc66 !important;
        transform: scale(1.02);
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        color: white !important;
        border-radius: 12px;
        border: 1px solid #e94560;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .user-message {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
    }
    .assistant-message {
        background: linear-gradient(135deg, #0f3460, #1a4a7a);
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
    }
    .char-picker {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .char-btn {
        background-color: #2a5298;
        border: none;
        border-radius: 20px;
        padding: 5px 12px;
        color: white;
        cursor: pointer;
        font-size: 1rem;
        transition: 0.2s;
        margin-right: 5px;
    }
    .char-btn:hover {
        background-color: #e94560;
    }
    @keyframes spin-globe {
        0% { transform: rotate(0deg); filter: drop-shadow(0 0 2px gold); }
        50% { filter: drop-shadow(0 0 15px #ffaa33) drop-shadow(0 0 5px orange); }
        100% { transform: rotate(360deg); filter: drop-shadow(0 0 2px gold); }
    }
    .spinning-brain {
        animation: spin-globe 3s linear infinite;
        display: inline-block;
        font-size: 3rem;
        text-align: center;
        width: 100%;
    }
    .sidebar-info {
        text-align: center;
        margin-top: 1rem;
        padding: 0.5rem;
        border-top: 1px solid #e94560;
        font-size: 0.9rem;
    }
    .sidebar-info a {
        color: #ffaa33 !important;
        text-decoration: none;
    }
    .sidebar-info a:hover {
        text-decoration: underline;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- LANGUAGES AND TEXTS ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

TEXTS = {
    "en": {
        "app_title": "Gesner AI - Kreyòl Assistant",
        "chat_input": "Ask me anything in Kreyòl...",
        "send": "Send",
        "clear": "Clear Chat",
        "dictionary": "Dictionary",
        "voice_training": "Voice Training",
        "bulk_training": "Bulk Training",
        "manage_facts": "Manage Facts",
        "test_training": "Test Training",
        "training_center": "Training Center",
        "train_new": "Train New Fact",
        "fact_text": "Fact text",
        "add_fact": "Add Fact",
        "upload_csv": "Upload CSV",
        "upload_audio": "Upload Audio",
        "record_voice": "Record Voice",
        "save_voice": "Save Voice",
        "play": "Play",
        "delete": "Delete",
        "edit": "Edit",
        "update": "Update",
        "chat_interface_label": "Chat",
        "humanoid_robot": "Humanoid Robot"
    },
    "fr": {
        "app_title": "Gesner IA - Assistant Kreyòl",
        "chat_input": "Posez-moi une question en Kreyòl...",
        "send": "Envoyer",
        "clear": "Effacer",
        "dictionary": "Dictionnaire",
        "voice_training": "Entraînement vocal",
        "bulk_training": "Formation en masse",
        "manage_facts": "Gérer les faits",
        "test_training": "Tester l'entraînement",
        "training_center": "Centre de formation",
        "train_new": "Ajouter un fait",
        "fact_text": "Texte du fait",
        "add_fact": "Ajouter",
        "upload_csv": "Importer CSV",
        "upload_audio": "Importer audio",
        "record_voice": "Enregistrer",
        "save_voice": "Sauvegarder",
        "play": "Écouter",
        "delete": "Supprimer",
        "edit": "Modifier",
        "update": "Mettre à jour",
        "chat_interface_label": "Discussion",
        "humanoid_robot": "Robot humanoïde"
    },
    "ht": {
        "app_title": "Gesner AI - Asistan Kreyòl",
        "chat_input": "Pose m yon kesyon an Kreyòl...",
        "send": "Voye",
        "clear": "Efase",
        "dictionary": "Diksyonè",
        "voice_training": "Fòmasyon Vwa",
        "bulk_training": "Fòmasyon an mas",
        "manage_facts": "Jere reyalite yo",
        "test_training": "Tès fòmasyon",
        "training_center": "Sant Fòmasyon",
        "train_new": "Anseye yon nouvo reyalite",
        "fact_text": "Tèks reyalite a",
        "add_fact": "Ajoute",
        "upload_csv": "Chaje CSV",
        "upload_audio": "Chaje odyo",
        "record_voice": "Anrejistre",
        "save_voice": "Sove",
        "play": "Jwe",
        "delete": "Efase",
        "edit": "Modifye",
        "update": "Mete ajou",
        "chat_interface_label": "Chat",
        "humanoid_robot": "Wobo Imitè Moun"
    }
}

# ---------- SESSION STATE ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model... (first time only)"):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []
if "training_data" not in st.session_state:
    st.session_state.training_data = load_training_data()
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = load_dictionaries()
if "training_access" not in st.session_state:
    st.session_state.training_access = False
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "play_audio" not in st.session_state:
    st.session_state.play_audio = None

VOICE_CACHE = load_voice_cache()

# ---------- PRE‑DEFINED VOICE MAPPING ----------
PREDEFINED_VOICES = {
    "kijan ou rele": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording.wav",
    "site konbyen let ki genhen nan alfabe kreyol la": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(1).wav",
    "konbyen let ki gehen nan alfabe kreyol la": "https://raw.githubusercontent.com/Deslandes1/Gesner-AIx/main/recording%20(3).wav"
}

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def get_predefined_voice_url(user_question):
    norm_q = normalize_text(user_question)
    for key, url in PREDEFINED_VOICES.items():
        if key in norm_q or norm_q.startswith(key):
            return url
    return None

# ---------- HELPER FUNCTIONS ----------
def save_all():
    save_training_data()
    save_dictionaries()
    save_voice_cache()

def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def rebuild_index():
    if st.session_state.training_data:
        st.session_state.texts = [item["text"] for item in st.session_state.training_data]
        embeddings = [np.array(item["embedding"], dtype=np.float32) for item in st.session_state.training_data]
        dim = len(embeddings[0])
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(np.array(embeddings))
        build_tfidf()
    else:
        st.session_state.index = None
        st.session_state.texts = []
        st.session_state.tfidf_vectorizer = None
        st.session_state.tfidf_matrix = None

def add_to_training(text):
    if not text.strip():
        return False
    embedding = st.session_state.embedding_model.encode([text])[0]
    st.session_state.training_data.append({"text": text, "embedding": embedding.tolist()})
    rebuild_index()
    save_training_data()
    return True

def update_training_item(idx, new_text):
    if not new_text.strip():
        return False
    embedding = st.session_state.embedding_model.encode([new_text])[0]
    st.session_state.training_data[idx] = {"text": new_text, "embedding": embedding.tolist()}
    rebuild_index()
    save_training_data()
    return True

def delete_training_item(idx):
    st.session_state.training_data.pop(idx)
    rebuild_index()
    save_training_data()

def get_voice_filename(text):
    norm = text.strip().lower()
    h = hashlib.md5(norm.encode()).hexdigest()
    return h

def save_voice_for_text(text, audio_bytes):
    global VOICE_CACHE
    key = get_voice_filename(text)
    VOICE_CACHE[key] = audio_bytes
    save_voice_cache()

def get_voice_for_text(text):
    if not text:
        return None
    key = get_voice_filename(text)
    return VOICE_CACHE.get(key)

def character_picker(key_prefix, label="Insert Kreyòl characters:"):
    chars = ["e","è","E","È","o","ò","O","Ò","an","An","AN","en","En","EN","on","On","ON","oun","Oun","OUN"]
    st.markdown(f"**{label}**")
    cols = st.columns(len(chars))
    for i, ch in enumerate(chars):
        with cols[i]:
            if st.button(ch, key=f"char_{key_prefix}_{ch}"):
                if key_prefix.startswith("edit_"):
                    idx = key_prefix.split("_")[1]
                    key = f"edit_text_{idx}"
                    current = st.session_state.get(key, "")
                    st.session_state[key] = current + ch
                st.rerun()

def retrieve_facts_hybrid(query, k=5):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    query_embedding = st.session_state.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state.index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(st.session_state.texts) and distances[0][i] < 1.2:
            results.append(st.session_state.texts[idx])
    if st.session_state.tfidf_vectorizer is not None and st.session_state.tfidf_matrix is not None:
        q_vec = st.session_state.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-k:][::-1]
        for idx in top_indices:
            if scores[idx] > 0.1 and st.session_state.texts[idx] not in results:
                results.append(st.session_state.texts[idx])
    return results[:k]

def direct_keyword_answer(query):
    q_lower = query.lower().strip()
    if "ti malice" in q_lower:
        if "kiyès" in q_lower or "who" in q_lower or "kreyatè" in q_lower:
            return "Ti Malice se yon lojisyèl edikatif ki fèt pa Gesner Deslandes pou anseye Kreyòl Ayisyen atravè jwèt ak istwa."
        if "chapit" in q_lower or "chapter" in q_lower:
            return "Ti Malice gen 12 chapit. Chapit 1: Alfabè, Chapit 2: Nonm, Chapit 3: Koulè, Chapit 4: Fanmi, Chapit 5: Manje, Chapit 6: Bèt, Chapit 7: Vèb, Chapit 8: Tan, Chapit 9: Fraz senp, Chapit 10: Konvèsasyon, Chapit 11: Pwovèb, Chapit 12: Istwa."
        if "telechaje" in q_lower or "download" in q_lower:
            return "Ou ka telechaje Ti Malice sou sitwèb globalinternet.py."
        return "Ti Malice se yon lojisyèl k ap anseye Kreyòl Ayisyen. Li gen 12 chapit ak egzèsis. Pou plis enfòmasyon, mande m 'chapit Ti Malice' oswa 'telechaje Ti Malice'."
    if any(w in q_lower for w in ["beginner", "debutan", "debutant", "aprann kreyòl deba"]):
        return "Kou Kreyòl pou debitan (Beginner): Alfabè 32 lèt, pwonon (mwen, ou, li, nou, yo), vèb 'se' ak 'gen', salitasyon (Bonjou, Bonswa), nonm 1-10, koulè debaz. Kisa ou ta renmen aprann an premye?"
    if any(w in q_lower for w in ["intermediate", "entèmedyè", "mwayen", "intermédiaire"]):
        return "Kou Kreyòl entèmedyè: Tan pase ak 'te', tan kap vini ak 'ap' oswa 'pral', nègasyon ak 'pa', pwopozisyon (nan, sou, anba), fraz konplèks ak 'ki', 'kote', 'poukisa'. Vle w pran yon egzèsis?"
    if any(w in q_lower for w in ["advanced", "avanse", "avancé"]):
        return "Kou Kreyòl avansé: Pawòl konpoze, pwovèb popilè (Dèyè mòn gen mòn, Men anpil chay pa lou), tan ki konpoze (Mwen te ap manje), vwa pasif, sijonktif (Fòk ou vini), literati kreyòl, ak analiz powèm. Eksplore youn nan sijè sa yo."
    if any(q in q_lower for q in ["kijan ou rele", "kiyès ou ye", "kisa ou ye", "ki moun ou ye", "what is your name", "who are you"]):
        return "Non pa mwen se Gesner L’IA, kreyatè mwen an se Gesner Deslandes nan GlobalInternet.py."
    if any(q in q_lower for q in ["kiyès ki kreye ou", "ki moun ki fè ou", "who created you", "ki moun ki devlope ou", "kiyès ki te kreye ou"]):
        return "Mwen te kreye pa Gesner Deslandes, fondatè GlobalInternet.py. Li se yon enjenyè ki renmen edike Ayiti."
    if q_lower in ["bonjou","bonswa","hello","hi","salut"]:
        return "Bonjou! Kijan ou ye? Mwen la pou reponn kesyon ou."
    return None

def reason_about_question(query):
    q = query.lower().strip()
    math_match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q)
    if math_match:
        try:
            a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
            if op == '+': res = a + b
            elif op == '-': res = a - b
            elif op == '*': res = a * b
            elif op == '/': res = a / b
            else: res = None
            if res is not None:
                if isinstance(res, float) and res.is_integer():
                    res = int(res)
                return f"Repons lan se {res}."
        except: pass
    if "kapital" in q or "capital" in q:
        capitals = {"france":"Paris","ayiti":"Pòtoprens","haiti":"Port‑au‑Prince","etazini":"Washington, D.C.","usa":"Washington, D.C.","kanada":"Ottawa","brezil":"Brasília","alman":"Bèlen","itali":"Wòm","espay":"Madrid","angle":"Londr","japon":"Tokiyo"}
        for country, cap in capitals.items():
            if country in q:
                return f"Kapital {country.title()} se {cap}."
    if "ki lè li ye" in q or "what time" in q:
        now = datetime.now().strftime("%H:%M")
        return f"Kounye a li {now}."
    return None

def reason_answer(query, retrieved_facts):
    if not retrieved_facts:
        return None
    if len(retrieved_facts) == 1:
        return retrieved_facts[0]
    q_lower = query.lower()
    if any(w in q_lower for w in ["beginner", "debutan", "debutant"]):
        beginner_facts = [f for f in retrieved_facts if "beginner" in f.lower() or "debitan" in f.lower() or "alfabè" in f.lower() or "pwonon" in f.lower()]
        if beginner_facts:
            return ". ".join(beginner_facts[:3])
    if any(w in q_lower for w in ["intermediate", "entèmedyè"]):
        inter_facts = [f for f in retrieved_facts if "intermediate" in f.lower() or "entèmedyè" in f.lower() or "tan pase" in f.lower()]
        if inter_facts:
            return ". ".join(inter_facts[:3])
    if any(w in q_lower for w in ["advanced", "avanse"]):
        adv_facts = [f for f in retrieved_facts if "advanced" in f.lower() or "avanse" in f.lower() or "pwovèb" in f.lower()]
        if adv_facts:
            return ". ".join(adv_facts[:3])
    if "ti malice" in q_lower:
        malice_facts = [f for f in retrieved_facts if "ti malice" in f.lower()]
        if malice_facts:
            return ". ".join(malice_facts[:3])
    if any(word in q_lower for word in ["raconte", "rakonte", "istwa", "history", "histoire"]):
        history_facts = [f for f in retrieved_facts if any(kw in f.lower() for kw in ["endepandan", "revolisyon", "duvalier", "tranblemanntè", "1804", "1915", "1957", "bwa kayiman"])]
        if history_facts:
            combined = ". ".join(history_facts[:3])
            return combined + "."
        else:
            return retrieved_facts[0]
    return retrieved_facts[0]

def generate_response(user_input):
    normalized = user_input.strip().lower()
    patterns = [
        "site konbyen let ki genhen nan alfabe kreyol la",
        "site konbyen let ki genhen nan alfabe kreyol",
        "site konbyen let ki genhen nan alfabe kreyòl la",
        "site konbyen let ki genhen nan alfabe kreyòl",
        "konbyen let ki gehen nan alfabe kreyol la",
        "site konbyen let ki genhen"
    ]
    for pat in patterns:
        if pat in normalized:
            answer = "A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z"
            return answer, False, False

    with st.spinner("🧠 Gesner AI ap reflechi... (thinking...)"):
        time.sleep(0.8)
        direct = direct_keyword_answer(user_input)
        if direct:
            return direct, False, False
        math_result = reason_about_question(user_input)
        if math_result and ("+" in user_input or "-" in user_input or "*" in user_input or "/" in user_input):
            return math_result, False, False
        facts = retrieve_facts_hybrid(user_input, k=5)
        if facts:
            reasoned = reason_answer(user_input, facts)
            return reasoned, False, False
        logic = reason_about_question(user_input)
        if logic:
            return logic, False, False
    return "Mwen poko konn sa. Tanpri anseye m nan Sant Fòmasyon.", True, False

# ---------- AUDIO PLAYBACK ----------
def show_audio_button(text, user_question, key_suffix):
    url = get_predefined_voice_url(user_question) if user_question else None
    if url:
        if st.button("🔊", key=f"audio_btn_{key_suffix}", help="Play audio"):
            st.session_state.play_audio = ("url", url)
            st.rerun()
        return
    audio_bytes = get_voice_for_text(text)
    if audio_bytes:
        if st.button("🔊", key=f"audio_btn_{key_suffix}", help="Play audio"):
            st.session_state.play_audio = ("bytes", audio_bytes, "audio/wav")
            st.rerun()
        return

def render_audio_player():
    if st.session_state.play_audio:
        audio_type = st.session_state.play_audio[0]
        if audio_type == "url":
            url = st.session_state.play_audio[1]
            st.audio(url, format="audio/wav")
        elif audio_type == "bytes":
            _, data, mime = st.session_state.play_audio
            st.audio(data, format=mime)
        st.session_state.play_audio = None

# ---------- HUMANOID ROBOT PAGE (no speech bubble) ----------
def humanoid_robot_page():
    st.markdown("## 🤖 GlobalInternet.py Humanoid Robot")
    st.markdown("*A realistic humanoid robot that speaks, blinks, and gestures – created by Gesner Deslandes*")
    
    robot_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { margin: 0; overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            #info {
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 10px 15px;
                border-radius: 8px;
                pointer-events: none;
                z-index: 100;
                font-size: 14px;
                backdrop-filter: blur(5px);
            }
            #company {
                position: absolute;
                bottom: 20px;
                left: 20px;
                background: rgba(0,0,0,0.7);
                color: #ffaa33;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 12px;
                pointer-events: none;
                z-index: 100;
                backdrop-filter: blur(5px);
            }
            button {
                position: absolute;
                bottom: 30px;
                right: 30px;
                background: #e94560;
                color: white;
                border: none;
                border-radius: 30px;
                padding: 10px 20px;
                font-weight: bold;
                cursor: pointer;
                z-index: 200;
                font-family: inherit;
                transition: transform 0.2s;
            }
            button:hover {
                transform: scale(1.05);
                background: #ff6b6b;
            }
        </style>
    </head>
    <body>
        <div id="info">
            🤖 Humanoid Robot | Created by Gesner Deslandes | GlobalInternet.py<br>
            I teach English, French, Spanish, and Haitian Creole.
        </div>
        <div id="company">
            🌐 GlobalInternet.py – AI Education • Software Development • Industrial Robotics
        </div>
        <button id="speakBtn">🗣️ Make Robot Speak</button>
        
        <script type="importmap">
            {
                "imports": {
                    "three": "https://unpkg.com/three@0.128.0/build/three.module.js",
                    "three/addons/": "https://unpkg.com/three@0.128.0/examples/jsm/"
                }
            }
        </script>

        <script type="module">
            import * as THREE from 'three';
            import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
            import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

            // --- Setup Scene ---
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0a2a);
            scene.fog = new THREE.FogExp2(0x0a0a2a, 0.008);

            const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(2.8, 2.2, 4.5);
            camera.lookAt(0, 1.2, 0);

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.shadowMap.enabled = true;
            document.body.appendChild(renderer.domElement);

            const labelRenderer = new CSS2DRenderer();
            labelRenderer.setSize(window.innerWidth, window.innerHeight);
            labelRenderer.domElement.style.position = 'absolute';
            labelRenderer.domElement.style.top = '0px';
            labelRenderer.domElement.style.left = '0px';
            labelRenderer.domElement.style.pointerEvents = 'none';
            document.body.appendChild(labelRenderer.domElement);

            // Controls
            const controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.autoRotate = false;
            controls.enableZoom = true;
            controls.target.set(0, 1.2, 0);

            // --- Lighting ---
            const ambientLight = new THREE.AmbientLight(0x404060);
            scene.add(ambientLight);
            const mainLight = new THREE.DirectionalLight(0xffffff, 1);
            mainLight.position.set(3, 5, 2);
            mainLight.castShadow = true;
            mainLight.receiveShadow = true;
            scene.add(mainLight);
            const fillLight = new THREE.PointLight(0x8866cc, 0.3);
            fillLight.position.set(0, 1, 1);
            scene.add(fillLight);
            const backLight = new THREE.PointLight(0xffaa66, 0.5);
            backLight.position.set(-1, 1.5, -2);
            scene.add(backLight);
            
            // --- Industrial floor and environment ---
            const gridHelper = new THREE.GridHelper(12, 20, 0x88aaff, 0x335588);
            gridHelper.position.y = -0.9;
            scene.add(gridHelper);
            const groundPlane = new THREE.Mesh(
                new THREE.PlaneGeometry(10, 10),
                new THREE.ShadowMaterial({ opacity: 0.5, color: 0x000000, transparent: true })
            );
            groundPlane.rotation.x = -Math.PI / 2;
            groundPlane.position.y = -0.89;
            groundPlane.receiveShadow = true;
            scene.add(groundPlane);
            
            // Simple industrial props
            const pillarMat = new THREE.MeshStandardMaterial({ color: 0x88aacc, metalness: 0.8 });
            const pillar = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.3, 2, 6), pillarMat);
            pillar.position.set(2.5, -0.2, -2);
            pillar.castShadow = true;
            scene.add(pillar);
            const gearMat = new THREE.MeshStandardMaterial({ color: 0xccaa77, metalness: 0.6 });
            const gear = new THREE.Mesh(new THREE.CylinderGeometry(0.5, 0.5, 0.15, 24), gearMat);
            gear.position.set(-2, 0, -2);
            gear.castShadow = true;
            scene.add(gear);
            
            // --- Humanoid Robot Model (realistic human-like) ---
            const robotGroup = new THREE.Group();
            
            // Torso
            const torsoGeom = new THREE.CylinderGeometry(0.55, 0.5, 1.1, 12);
            const torsoMat = new THREE.MeshStandardMaterial({ color: 0xDEB887, roughness: 0.4, metalness: 0.1 });
            const torso = new THREE.Mesh(torsoGeom, torsoMat);
            torso.castShadow = true;
            torso.receiveShadow = true;
            torso.position.y = 0.6;
            robotGroup.add(torso);
            
            // Neck
            const neckMat = new THREE.MeshStandardMaterial({ color: 0xDEB887 });
            const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.22, 0.2, 8), neckMat);
            neck.position.y = 1.15;
            neck.castShadow = true;
            robotGroup.add(neck);
            
            // Head
            const headMat = new THREE.MeshStandardMaterial({ color: 0xDEB887, roughness: 0.3 });
            const head = new THREE.Mesh(new THREE.SphereGeometry(0.45, 48, 48), headMat);
            head.position.y = 1.45;
            head.castShadow = true;
            robotGroup.add(head);
            
            // Hair
            const hairMat = new THREE.MeshStandardMaterial({ color: 0x4a2c2c });
            const hair = new THREE.Mesh(new THREE.SphereGeometry(0.48, 32, 32), hairMat);
            hair.position.y = 1.75;
            hair.scale.set(1, 0.4, 1);
            robotGroup.add(hair);
            
            // Eyes
            const eyeWhiteMat = new THREE.MeshStandardMaterial({ color: 0xffffff });
            const eyeIrisMat = new THREE.MeshStandardMaterial({ color: 0x4a2c1e });
            const eyePupilMat = new THREE.MeshStandardMaterial({ color: 0x000000 });
            
            const leftEyeWhite = new THREE.Mesh(new THREE.SphereGeometry(0.12, 32, 32), eyeWhiteMat);
            leftEyeWhite.position.set(-0.16, 1.58, 0.45);
            robotGroup.add(leftEyeWhite);
            const rightEyeWhite = new THREE.Mesh(new THREE.SphereGeometry(0.12, 32, 32), eyeWhiteMat);
            rightEyeWhite.position.set(0.16, 1.58, 0.45);
            robotGroup.add(rightEyeWhite);
            
            const leftIris = new THREE.Mesh(new THREE.SphereGeometry(0.08, 32, 32), eyeIrisMat);
            leftIris.position.set(-0.16, 1.58, 0.57);
            robotGroup.add(leftIris);
            const rightIris = new THREE.Mesh(new THREE.SphereGeometry(0.08, 32, 32), eyeIrisMat);
            rightIris.position.set(0.16, 1.58, 0.57);
            robotGroup.add(rightIris);
            
            const leftPupil = new THREE.Mesh(new THREE.SphereGeometry(0.05, 32, 32), eyePupilMat);
            leftPupil.position.set(-0.16, 1.57, 0.64);
            robotGroup.add(leftPupil);
            const rightPupil = new THREE.Mesh(new THREE.SphereGeometry(0.05, 32, 32), eyePupilMat);
            rightPupil.position.set(0.16, 1.57, 0.64);
            robotGroup.add(rightPupil);
            
            // Eyebrows
            const browMat = new THREE.MeshStandardMaterial({ color: 0x4a2c2c });
            const leftBrow = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.05, 0.1), browMat);
            leftBrow.position.set(-0.2, 1.7, 0.48);
            robotGroup.add(leftBrow);
            const rightBrow = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.05, 0.1), browMat);
            rightBrow.position.set(0.2, 1.7, 0.48);
            robotGroup.add(rightBrow);
            
            // Jaw (moving part)
            const jawMat = new THREE.MeshStandardMaterial({ color: 0xDEB887 });
            const jaw = new THREE.Mesh(new THREE.SphereGeometry(0.28, 32, 32), jawMat);
            jaw.position.y = 1.28;
            jaw.scale.set(0.9, 0.5, 0.7);
            jaw.castShadow = true;
            robotGroup.add(jaw);
            
            // Nose
            const noseMat = new THREE.MeshStandardMaterial({ color: 0xDEB887 });
            const nose = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.1, 0.12, 8), noseMat);
            nose.position.set(0, 1.45, 0.5);
            robotGroup.add(nose);
            
            // Arms
            const armMat = new THREE.MeshStandardMaterial({ color: 0xDEB887 });
            const leftArmUpper = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.12, 0.65, 8), armMat);
            leftArmUpper.position.set(-0.65, 1.0, 0);
            leftArmUpper.rotation.z = 0.4;
            leftArmUpper.castShadow = true;
            robotGroup.add(leftArmUpper);
            const rightArmUpper = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.12, 0.65, 8), armMat);
            rightArmUpper.position.set(0.65, 1.0, 0);
            rightArmUpper.rotation.z = -0.4;
            rightArmUpper.castShadow = true;
            robotGroup.add(rightArmUpper);
            
            const leftForearm = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.09, 0.55, 8), armMat);
            leftForearm.position.set(-0.95, 0.68, 0);
            leftForearm.rotation.z = 0.6;
            leftForearm.castShadow = true;
            robotGroup.add(leftForearm);
            const rightForearm = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.09, 0.55, 8), armMat);
            rightForearm.position.set(0.95, 0.68, 0);
            rightForearm.rotation.z = -0.6;
            rightForearm.castShadow = true;
            robotGroup.add(rightForearm);
            
            // Hands
            const handMatObj = new THREE.MeshStandardMaterial({ color: 0xDEB887 });
            const leftHand = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), handMatObj);
            leftHand.position.set(-1.2, 0.45, 0);
            robotGroup.add(leftHand);
            const rightHand = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), handMatObj);
            rightHand.position.set(1.2, 0.45, 0);
            robotGroup.add(rightHand);
            
            // Legs and feet
            const legMatObj = new THREE.MeshStandardMaterial({ color: 0xCDA87C });
            const leftLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.14, 0.8, 8), legMatObj);
            leftLeg.position.set(-0.25, 0.1, 0);
            leftLeg.castShadow = true;
            robotGroup.add(leftLeg);
            const rightLeg = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.14, 0.8, 8), legMatObj);
            rightLeg.position.set(0.25, 0.1, 0);
            rightLeg.castShadow = true;
            robotGroup.add(rightLeg);
            
            const footMatObj = new THREE.MeshStandardMaterial({ color: 0x8B5A2B });
            const leftFoot = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.12, 0.5), footMatObj);
            leftFoot.position.set(-0.28, -0.3, 0.1);
            leftFoot.castShadow = true;
            robotGroup.add(leftFoot);
            const rightFoot = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.12, 0.5), footMatObj);
            rightFoot.position.set(0.28, -0.3, 0.1);
            rightFoot.castShadow = true;
            robotGroup.add(rightFoot);
            
            scene.add(robotGroup);
            
            // CSS2D label above head
            const nameDiv = document.createElement('div');
            nameDiv.textContent = '🤖 Gesner AI Robot';
            nameDiv.style.backgroundColor = 'rgba(0,0,0,0.6)';
            nameDiv.style.color = '#ffaa33';
            nameDiv.style.padding = '2px 8px';
            nameDiv.style.borderRadius = '20px';
            nameDiv.style.fontSize = '12px';
            nameDiv.style.border = '1px solid #ffaa33';
            const nameLabel = new CSS2DObject(nameDiv);
            nameLabel.position.set(0, 2.1, 0);
            scene.add(nameLabel);
            
            // --- Speech Synthesis (Web Speech API) ---
            let synth = window.speechSynthesis;
            let currentUtterance = null;
            
            function stopSpeaking() {
                if (synth.speaking || synth.pending) {
                    synth.cancel();
                }
                if (currentUtterance) {
                    currentUtterance = null;
                }
            }
            
            function speakMessage(message) {
                stopSpeaking();
                // Start visual speaking animation
                startSpeakingAnimation();
                
                // Speak using browser's speech synthesis
                const utterance = new SpeechSynthesisUtterance(message);
                utterance.lang = 'en-US';
                utterance.rate = 0.9;
                utterance.pitch = 1.1;
                utterance.volume = 1;
                utterance.onend = function() {
                    currentUtterance = null;
                    stopSpeakingAnimation();
                };
                utterance.onerror = function() {
                    currentUtterance = null;
                    stopSpeakingAnimation();
                };
                currentUtterance = utterance;
                synth.speak(utterance);
            }
            
            // Animation state
            let isAnimating = false;
            let jawInterval = null;
            let browInterval = null;
            
            function startSpeakingAnimation() {
                if (isAnimating) return;
                isAnimating = true;
                
                // Jaw movement
                let jawOpen = 0;
                jawInterval = setInterval(() => {
                    if (!isAnimating) {
                        clearInterval(jawInterval);
                        jaw.position.y = 1.28;
                        return;
                    }
                    jawOpen = jawOpen === 0 ? 0.06 : 0;
                    jaw.position.y = 1.28 + jawOpen;
                }, 200);
                
                // Eyebrow wiggle and arm wave
                let browTime = 0;
                browInterval = setInterval(() => {
                    if (!isAnimating) {
                        clearInterval(browInterval);
                        leftBrow.position.y = 1.7;
                        rightBrow.position.y = 1.7;
                        leftArmUpper.rotation.z = 0.4;
                        rightArmUpper.rotation.z = -0.4;
                        leftForearm.rotation.z = 0.6;
                        rightForearm.rotation.z = -0.6;
                        return;
                    }
                    browTime += 0.3;
                    const lift = Math.sin(browTime) * 0.03;
                    leftBrow.position.y = 1.7 + lift;
                    rightBrow.position.y = 1.7 + lift;
                    const wave = Math.sin(browTime * 2) * 0.2;
                    rightArmUpper.rotation.z = -0.4 + wave * 0.5;
                    leftArmUpper.rotation.z = 0.4 - wave * 0.3;
                    rightForearm.rotation.z = -0.6 + wave * 0.4;
                    leftForearm.rotation.z = 0.6 - wave * 0.3;
                }, 150);
            }
            
            function stopSpeakingAnimation() {
                isAnimating = false;
                if (jawInterval) clearInterval(jawInterval);
                if (browInterval) clearInterval(browInterval);
                jaw.position.y = 1.28;
                leftBrow.position.y = 1.7;
                rightBrow.position.y = 1.7;
                leftArmUpper.rotation.z = 0.4;
                rightArmUpper.rotation.z = -0.4;
                leftForearm.rotation.z = 0.6;
                rightForearm.rotation.z = -0.6;
            }
            
            // --- Messages (only spoken on button click) ---
            const messageList = [
                "Hello! I am the GlobalInternet.py Humanoid Robot, created by Gesner Deslandes. Our company builds Python-based software on demand for clients worldwide. Like Silicon Valley, but with a Haitian touch and outstanding outcomes. We offer AI-powered solutions – chatbots, data analysis, automation; complete election and voting systems – secure, multi-language, real-time; web applications – dashboards, internal tools, online platforms; and full package delivery – we email you the complete code and guide you through installation. Whether you need a company website, a custom software tool, or a full-scale online platform – we build it, you own it. Founder and CEO: Gesner Deslandes – Engineer, AI Enthusiast, Python Expert. Contact: (509) 4738-5663, email: deslandes78@gmail.com. I can teach you four languages: English, French, Spanish, and Haitian Creole. Just ask me for beginner, intermediate, or advanced lessons in any of them.",
                "I can teach you English, from beginner to advanced. Would you like a lesson?",
                "I can teach you French. Ask me for French lessons.",
                "I can teach you Spanish. Would you like to learn Spanish?",
                "I can teach you Haitian Creole. Do you want to learn Kreyòl?",
                "Visit our website globalinternet.py to see our projects and services."
            ];
            
            // Button click: speak a random message
            document.getElementById('speakBtn').addEventListener('click', () => {
                const randomMsg = messageList[Math.floor(Math.random() * messageList.length)];
                speakMessage(randomMsg);
            });
            
            // Blink eyes periodically
            function blinkEyes() {
                leftPupil.scale.set(1, 0.05, 1);
                rightPupil.scale.set(1, 0.05, 1);
                setTimeout(() => {
                    leftPupil.scale.set(1, 1, 1);
                    rightPupil.scale.set(1, 1, 1);
                }, 100);
            }
            setInterval(blinkEyes, 4000);
            
            // Animate environment (gear rotation) and idle head movement
            let time = 0;
            function animateScene() {
                requestAnimationFrame(animateScene);
                time += 0.02;
                gear.rotation.y += 0.02;
                head.rotation.z = Math.sin(time * 0.8) * 0.02;
                controls.update();
                renderer.render(scene, camera);
                labelRenderer.render(scene, camera);
            }
            animateScene();
            
            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
                labelRenderer.setSize(window.innerWidth, window.innerHeight);
            });
        </script>
    </body>
    </html>
    """
    st.components.v1.html(robot_html, height=650, scrolling=False)

# ---------- UI COMPONENTS ----------
def dictionary_manager(t):
    st.subheader(t['dictionary'])
    lang = st.selectbox("Select language", list(LANGUAGES.keys()), key="dict_lang")
    lang_code = LANGUAGES[lang]
    word = st.text_input("Word / Phrase", key="dict_word")
    meaning = st.text_area("Meaning / Translation", key="dict_meaning")
    if st.button("Add / Update", key="dict_add"):
        if word and meaning:
            st.session_state.dictionaries[lang_code][word] = meaning
            save_dictionaries()
            st.success("Saved!")
            st.rerun()
    st.markdown("---")
    st.write("**Existing entries**")
    for w, m in st.session_state.dictionaries[lang_code].items():
        col1, col2 = st.columns([3,1])
        with col1:
            st.write(f"**{w}**: {m}")
        with col2:
            if st.button("Delete", key=f"del_{lang_code}_{w}"):
                del st.session_state.dictionaries[lang_code][w]
                save_dictionaries()
                st.rerun()

def voice_training(t):
    st.subheader(t['voice_training'])
    fact_text = st.text_area(t['fact_text'], key="voice_fact")
    uploaded_audio = st.file_uploader(t['upload_audio'], type=["wav", "mp3"], key="voice_upload")
    if uploaded_audio:
        audio_bytes = uploaded_audio.read()
        st.audio(audio_bytes, format="audio/wav")
        if st.button(t['save_voice'], key="save_voice_btn"):
            save_voice_for_text(fact_text, audio_bytes)
            st.success("Voice saved!")
    st.markdown("---")
    st.write("**Existing voice mappings**")
    for idx, item in enumerate(st.session_state.training_data):
        text = item["text"]
        if get_voice_for_text(text):
            col1, col2 = st.columns([3,1])
            with col1:
                st.write(text[:60] + "..." if len(text) > 60 else text)
            with col2:
                if st.button(t['play'], key=f"play_voice_{idx}"):
                    audio_bytes = get_voice_for_text(text)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")

def bulk_training(t):
    st.subheader(t['bulk_training'])
    uploaded_file = st.file_uploader(t['upload_csv'], type=["csv"], key="bulk_csv")
    if uploaded_file:
        df = csv.DictReader(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
        facts = [row.get("fact") or row.get("text") for row in df]
        if facts:
            if st.button("Import facts", key="bulk_import"):
                count = 0
                for fact in facts:
                    if fact and fact.strip():
                        if add_to_training(fact.strip()):
                            count += 1
                st.success(f"Imported {count} facts.")
                st.rerun()

def manage_trained_facts(t):
    st.subheader(t['manage_facts'])
    for idx, item in enumerate(st.session_state.training_data):
        col1, col2, col3 = st.columns([4,1,1])
        with col1:
            if f"edit_{idx}" in st.session_state and st.session_state[f"edit_{idx}"]:
                new_text = st.text_area("Edit", value=item["text"], key=f"edit_text_{idx}")
                if st.button("Save", key=f"save_edit_{idx}"):
                    update_training_item(idx, new_text)
                    st.session_state[f"edit_{idx}"] = False
                    st.rerun()
            else:
                st.write(item["text"])
        with col2:
            if st.button(t['edit'], key=f"edit_btn_{idx}"):
                st.session_state[f"edit_{idx}"] = True
                st.rerun()
        with col3:
            if st.button(t['delete'], key=f"del_btn_{idx}"):
                delete_training_item(idx)
                st.rerun()

def test_training_section(t):
    st.subheader(t['test_training'])
    query = st.text_input("Test query", key="test_query")
    if st.button("Test", key="test_btn"):
        if query:
            facts = retrieve_facts_hybrid(query, k=3)
            if facts:
                st.write("**Retrieved facts:**")
                for f in facts:
                    st.write(f"- {f}")
            else:
                st.write("No relevant facts found.")

def training_center(t):
    st.markdown(f"## {t['training_center']}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {t['train_new']}")
        new_fact = st.text_area(t['fact_text'], key="new_fact")
        if st.button(t['add_fact'], key="add_fact_btn"):
            if new_fact.strip():
                add_to_training(new_fact.strip())
                st.success("Fact added!")
                st.rerun()
    with col2:
        bulk_training(t)
    manage_trained_facts(t)
    test_training_section(t)

def chat_interface(t):
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['app_title']}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Mwen reponn sèlman an Kreyòl. Poze m kesyon sou alfabè, gramè, istwa Ayiti, oswa nenpòt bagay ou te anseye m.</p>", unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.container():
                col_text, col_btn = st.columns([10, 1])
                with col_text:
                    st.markdown(f'<div class="assistant-message" style="padding:0.5rem; border-radius:20px;">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
                with col_btn:
                    if not msg.get("skip_audio", False):
                        user_q = st.session_state.conversation_history[idx-1]["content"] if idx > 0 else ""
                        show_audio_button(msg["content"], user_q, f"chat_{idx}")
            st.markdown("")
    render_audio_player()
    user_input = st.text_input(t['chat_input'], key="chat_input")
    if st.button(t['send'], use_container_width=True, key="send_btn"):
        if user_input.strip():
            answer, is_fallback, skip_audio = generate_response(user_input)
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "is_fallback": is_fallback,
                "skip_audio": skip_audio
            })
            st.rerun()
    if st.button(t['clear'], use_container_width=True, key="clear_btn"):
        st.session_state.conversation_history = []
        st.rerun()

def show_sidebar():
    with st.sidebar:
        st.markdown('<div class="spinning-brain">🧠</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-info">
                <strong>Gesner AI</strong><br>
                Created by <strong>Gesner Deslandes</strong><br>
                Founder of <strong>GlobalInternet.py</strong><br>
                ✉️ <a href="mailto:deslandes78@gmail.com">deslandes78@gmail.com</a><br>
                📞 +509 4738-5663<br>
                🌐 <a href="https://globalinternet.py" target="_blank">globalinternet.py</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        lang_choice = st.selectbox("🌐 Interface Language", list(LANGUAGES.keys()), key="lang_select")
        st.session_state.ui_language = LANGUAGES[lang_choice]
        t = TEXTS.get(st.session_state.ui_language, TEXTS["en"])
        menu = st.radio("Menu", [t['chat_interface_label'], t['dictionary'], t['voice_training'], t['training_center'], t['humanoid_robot']])
        return menu, t

def main():
    if not st.session_state.training_data:
        initialize_default_training()
    menu, t = show_sidebar()
    if menu == t.get('chat_interface_label', "Chat"):
        chat_interface(t)
    elif menu == t['dictionary']:
        dictionary_manager(t)
    elif menu == t['voice_training']:
        voice_training(t)
    elif menu == t['training_center']:
        training_center(t)
    elif menu == t['humanoid_robot']:
        humanoid_robot_page()

if __name__ == "__main__":
    main()
