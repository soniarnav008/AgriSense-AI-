
import json
import os
import hashlib

import streamlit as st
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from src.storage import init_db, save_record, get_records
from src.plant_doctor import predict_leaf_disease, model_status
from src.chatbot import ask_farmer_assistant
from src.monitoring import assess_conditions
from src.reports import make_markdown_report

load_dotenv()

st.set_page_config(
    page_title="AgriSense AI | Precision Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()


# ==================================================
# AGRISENSE AI - DARK EDGE AI THEME
# ==================================================

st.markdown("""
<style>
:root {
    --bg: #090e13;
    --sidebar: #0d141d;
    --surface: #121c28;
    --surface2: #172533;
    --cyan: #19d9f5;
    --cyan-soft: #8bdcf0;
    --text: #f4f7fb;
    --muted: #8c9baa;
    --border: #263747;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

header[data-testid="stHeader"] {
    background: rgba(9,14,19,0.96);
}

.block-container {
    max-width: 1450px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

section[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
    color: var(--text);
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

p, label, .stMarkdown {
    color: #aab7c5;
}

/* SIDEBAR RADIO NAVIGATION */

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 8px;
    padding: 5px 8px;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #172533;
    color: var(--cyan) !important;
}

/* HERO PANEL */

.hero-panel {
    background:
        radial-gradient(circle at 90% 10%,
        rgba(25,217,245,0.10), transparent 35%),
        linear-gradient(115deg, #101b27, #162b38);
    border: 1px solid #263e50;
    border-radius: 18px;
    padding: 36px;
    margin-bottom: 38px;
    min-height: 310px;
}

.hero-label {
    color: var(--cyan);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(36px, 4.5vw, 58px);
    font-weight: 850;
    line-height: 1.15;
    margin: 24px 0 12px;
}

.hero-subtitle {
    color: #a8ddf0;
    font-size: 21px;
    margin-bottom: 20px;
}

.hero-description {
    color: #a9bfd0;
    font-size: 15px;
    line-height: 1.8;
    max-width: 750px;
}

.hero-tag {
    display: inline-block;
    color: var(--cyan);
    background: rgba(25,217,245,0.07);
    border: 1px solid #315465;
    border-radius: 30px;
    padding: 9px 15px;
    margin: 20px 7px 0 0;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

/* SECTION LABELS */

.section-eyebrow {
    color: var(--cyan);
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 10px;
}

/* METRIC CARDS */

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #15212e, #101923);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    min-height: 135px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #f4f7fb !important;
}

[data-testid="stMetricDelta"] {
    color: var(--cyan) !important;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    background: #122b38;
    color: var(--cyan) !important;
    border: 1px solid #315568;
    border-radius: 9px;
    min-height: 43px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #164153;
    border-color: var(--cyan);
}

/* INPUT FIELDS */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stChatInput textarea {
    background: #111d29 !important;
    color: white !important;
    border: 1px solid #304354 !important;
    border-radius: 9px !important;
}

.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"] {
    background: #111d29;
}

/* FILE UPLOADER */

div[data-testid="stFileUploader"] {
    background: #111d29;
    border: 1px dashed #426174;
    border-radius: 13px;
    padding: 16px;
}

/* CONTAINERS AND EXPANDERS */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #101923;
    border: 1px solid var(--border);
    border-radius: 13px;
}

div[data-testid="stExpander"] {
    background: #111d29;
    border: 1px solid var(--border);
    border-radius: 11px;
}

[data-testid="stChatMessage"] {
    background: #121e2a;
    border: 1px solid var(--border);
    border-radius: 12px;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: var(--border);
}

.stCaption, small {
    color: var(--muted) !important;
}

/* STATUS DOT */

.status-dot {
    color: #49e6a1;
    font-size: 20px;
}

/* MOBILE */

@media(max-width:700px) {
    .hero-panel {
        padding: 23px;
    }

    .hero-title {
        font-size: 34px;
    }
}

/* =====================================
   FINAL ALIGNMENT & POLISH
===================================== */

/* Consistent page spacing */
.block-container {
    max-width: 1450px;
    padding: 42px 42px 55px;
}

/* Consistent heading spacing */
h1, h2, h3 {
    margin-top: 0 !important;
    margin-bottom: 14px !important;
    line-height: 1.3 !important;
}

/* Hero alignment */
.hero-panel {
    min-height: 310px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 38px 40px;
    margin-bottom: 38px;
    box-sizing: border-box;
}

.hero-title {
    margin: 18px 0 10px !important;
    line-height: 1.12 !important;
}

.hero-subtitle {
    margin: 0 0 16px !important;
    line-height: 1.4 !important;
}

.hero-description {
    margin: 0;
    line-height: 1.75;
}

/* Section spacing */
.section-eyebrow {
    margin-top: 10px;
    margin-bottom: 12px;
}

/* Equal metric cards */
[data-testid="stMetric"] {
    height: 142px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
    padding: 22px;
}

/* Equal tool cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
    min-height: 255px;
    box-sizing: border-box;
    padding: 20px;
}

/* Consistent card content */
div[data-testid="stVerticalBlockBorderWrapper"] h3 {
    min-height: 32px;
    margin-bottom: 12px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] p {
    line-height: 1.65;
}

/* Buttons aligned at bottom */
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stButton"] {
    margin-top: auto;
}

/* Consistent columns */
div[data-testid="stHorizontalBlock"] {
    align-items: stretch;
    gap: 16px;
}

/* Better vertical rhythm */
div[data-testid="stVerticalBlock"] {
    gap: 12px;
}

/* Workflow number alignment */
div[data-testid="stVerticalBlock"] h3 {
    line-height: 1.3;
}

/* Responsive spacing */
@media (max-width: 900px) {
    .block-container {
        padding: 25px 20px;
    }

    .hero-panel {
        padding: 26px;
        min-height: auto;
    }
}

@media (max-width: 600px) {
    .hero-title {
        font-size: 34px;
    }

    [data-testid="stMetric"] {
        height: auto;
        min-height: 120px;
    }
}

/* FINAL DASHBOARD POLISH */

.block-container {
    padding-top: 2.1rem;
    padding-left: 40px;
    padding-right: 40px;
}

.hero-panel {
    margin-bottom: 42px;
}

.hero-title {
    font-size: clamp(38px, 4vw, 54px);
    letter-spacing: -1.5px;
}

.section-eyebrow {
    margin-top: 0;
    margin-bottom: 12px;
}

[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}

.stButton > button {
    width: 100%;
    min-height: 44px;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 7px 9px;
    margin-bottom: 3px;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] input {
    accent-color: #19d9f5;
}

@media(max-width:900px) {
    .block-container {
        padding-left: 20px;
        padding-right: 20px;
    }
}

/* ==================================
   TOOL AND WORKFLOW FINAL ALIGNMENT
================================== */

/* Remove excessive global card spacing */
div[data-testid="stVerticalBlockBorderWrapper"] {
    min-height: 0;
    height: auto;
    padding: 16px;
}

/* Tool cards */
div[data-testid="stHorizontalBlock"] {
    align-items: stretch;
}

/* Tool descriptions */
div[data-testid="stMarkdownContainer"] p {
    line-height: 1.65;
}

/* Buttons */
div[data-testid="stButton"] button {
    min-height: 43px;
    font-size: 13px;
    font-weight: 650;
}

/* Workflow cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #263747;
    border-radius: 13px;
    background: linear-gradient(145deg, #121d29, #0f1822);
}

/* Consistent section spacing */
.section-eyebrow {
    margin-top: 8px;
    margin-bottom: 10px;

}
@media(max-width: 850px) {
    .block-container {
        padding-left: 20px;
        padding-right: 20px;
    }
}

/* ==========================================
   AGRISENSE AI — ALL PAGE DESIGN SYSTEM
========================================== */

/* Consistent page headings */
h1 {
    font-size: 34px !important;
    font-weight: 800 !important;
    letter-spacing: -1px !important;
    margin-bottom: 12px !important;
}

h2 {
    font-size: 25px !important;
    font-weight: 750 !important;
    margin-top: 25px !important;
    margin-bottom: 15px !important;
}

h3 {
    font-size: 19px !important;
    font-weight: 700 !important;
}

/* Page descriptions */
div[data-testid="stAppViewContainer"] p {
    line-height: 1.7;
}

/* Result panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg,#14212e,#101923);
    border: 1px solid #263747;
    border-radius: 14px;
    padding: 18px;
}

/* Forms */
[data-testid="stForm"] {
    background: #111c28;
    border: 1px solid #263747;
    border-radius: 14px;
    padding: 22px;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #9aabba !important;
    font-weight: 650 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #19d9f5 !important;
}

/* Progress bars */
div[data-testid="stProgressBar"] > div {
    background: #263747;
    border-radius: 20px;
}

div[data-testid="stProgressBar"] > div > div {
    background: #19d9f5;
    border-radius: 20px;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #111c28;
    border: 1px solid #263747;
    border-radius: 12px;
}

/* Download buttons */
div[data-testid="stDownloadButton"] button {
    background: #122b38;
    color: #19d9f5 !important;
    border: 1px solid #315568;
    border-radius: 9px;
    min-height: 44px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #121e2a;
    border: 1px solid #263747;
    border-radius: 13px;
    padding: 16px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 11px;
}

/* Consistent section spacing */
hr {
    margin: 28px 0 !important;
    border-color: #263747 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d141d;
    border-right: 1px solid #263747;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #172533;
    border-radius: 8px;
    color: #19d9f5 !important;
}

/* Mobile */
@media(max-width:700px) {
    h1 {
        font-size: 27px !important;
    }

    .block-container {
        padding-left: 16px;
        padding-right: 16px;
    }
}

/* ===== TEXT SIZE IMPROVEMENT ===== */

/* Main headings */
h1 {
    font-size: 38px !important;
    font-weight: 800 !important;
}

/* Section headings */
h2 {
    font-size: 28px !important;
    font-weight: 750 !important;
}

h3 {
    font-size: 22px !important;
    font-weight: 700 !important;
}

/* Normal paragraph text */
p {
    font-size: 17px !important;
    line-height: 1.8 !important;
}

/* Labels and captions */
label,
[data-testid="stCaptionContainer"] {
    font-size: 15px !important;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-size: 32px !important;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-size: 16px !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    font-size: 16px !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    font-size: 16px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 16px !important;
}

/* Input fields */
input, textarea {
    font-size: 16px !important;
}

/* Mobile adjustment */
@media(max-width:700px) {
    h1 { font-size: 30px !important; }
    h2 { font-size: 24px !important; }
    p { font-size: 16px !important; }
}

/* ===== CHAT INPUT PLACEHOLDER VISIBILITY ===== */

[data-testid="stChatInput"] textarea {
    background-color: #172635 !important;
    color: #ffffff !important;
    border: 1px solid #315568 !important;
    font-size: 17px !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #a9c1d0 !important;
    opacity: 1 !important;
}

/* Chat input focus */
[data-testid="stChatInput"] textarea:focus {
    border: 1px solid #19d9f5 !important;
    box-shadow: 0 0 8px rgba(25,217,245,0.15) !important;
}
/* Dashboard heading hierarchy fix */

.dashboard-section-label {
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    color: #55d6f5 !important;
    text-transform: uppercase;
}

.dashboard-section-title {
    font-size: 32px !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: -0.5px !important;
    margin-top: 5px !important;
}

.dashboard-description {
    font-size: 17px !important;
    color: #b8cbd9 !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# PLANT DISEASE KNOWLEDGE BASE
# ==================================================

DISEASE_GUIDE = {

    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "cause": "Fungus Alternaria solani.",
        "description": "A fungal disease that commonly affects older leaves first and can reduce plant vigour and tuber yield.",
        "symptoms": [
            "Brown spots with concentric ring patterns (target-like appearance).",
            "Yellowing around affected leaf areas.",
            "Older leaves are often affected first.",
            "Severely affected leaves may dry and fall."
        ],
        "solutions": [
            "Remove heavily infected leaves and dispose of them away from the field.",
            "Avoid overhead irrigation and prolonged leaf wetness.",
            "Maintain adequate spacing and airflow between plants.",
            "Use a locally registered fungicide only after confirming the disease and following its label.",
            "Consult an agricultural expert if symptoms continue spreading."
        ],
        "prevention": [
            "Use healthy, disease-free seed tubers.",
            "Rotate crops with non-host crops.",
            "Remove infected crop debris after harvest.",
            "Avoid excessive nitrogen fertilisation.",
            "Monitor lower and older leaves regularly."
        ]
    },

    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "cause": "Oomycete Phytophthora infestans.",
        "description": "A destructive potato disease that can spread rapidly in cool, humid and wet conditions.",
        "symptoms": [
            "Irregular, water-soaked dark patches on leaves.",
            "Brown or black lesions that can expand rapidly.",
            "White growth may appear on the underside of leaves in humid conditions.",
            "Dark lesions may develop on stems.",
            "Tubers may develop firm, brown or reddish-brown rot."
        ],
        "solutions": [
            "Remove severely infected plant material carefully.",
            "Avoid overhead irrigation and reduce leaf wetness.",
            "Do not move infected plant material through healthy crop areas.",
            "Seek prompt local agricultural advice about an appropriate registered late-blight control product.",
            "Follow product label instructions and pre-harvest intervals."
        ],
        "prevention": [
            "Plant certified disease-free seed potatoes.",
            "Remove volunteer potato plants and infected crop debris.",
            "Maintain adequate spacing and field drainage.",
            "Inspect crops frequently during cool, wet weather.",
            "Avoid saving tubers from infected plants for future planting."
        ]
    },

    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy",
        "cause": "No disease detected by the classifier.",
        "description": "The model classified the uploaded leaf as healthy. This does not rule out every possible plant disease.",
        "symptoms": [
            "Leaves generally retain their normal green colour.",
            "No obvious characteristic disease lesions.",
            "No clear signs of widespread leaf damage."
        ],
        "solutions": [
            "No disease-specific treatment is indicated by this prediction.",
            "Continue regular crop monitoring.",
            "Remove damaged or dead leaves where appropriate.",
            "Investigate any unusual changes that appear later."
        ],
        "prevention": [
            "Use healthy seed tubers.",
            "Maintain appropriate irrigation and drainage.",
            "Follow balanced nutrient management.",
            "Inspect leaves and stems regularly."
        ]
    },

    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "cause": "Xanthomonas species of bacteria.",
        "description": "A bacterial disease that affects tomato foliage and may also damage stems and fruits.",
        "symptoms": [
            "Small dark brown or black spots on leaves.",
            "Yellow halos may surround leaf spots.",
            "Spots may merge and cause leaf damage.",
            "Raised or scabby lesions can develop on fruits."
        ],
        "solutions": [
            "Remove severely affected plant material.",
            "Avoid handling plants while foliage is wet.",
            "Avoid overhead irrigation.",
            "Use disease-management products only after local expert confirmation.",
            "Clean tools before using them on healthy plants."
        ],
        "prevention": [
            "Use certified disease-free seeds and healthy transplants.",
            "Avoid working in wet fields.",
            "Rotate crops with non-host plants.",
            "Remove infected crop debris.",
            "Maintain good plant spacing."
        ]
    },

    "Tomato_Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "cause": "Fungus Alternaria solani.",
        "description": "A fungal disease that commonly begins on older tomato leaves.",
        "symptoms": [
            "Brown circular spots with concentric rings.",
            "Yellowing around infected areas.",
            "Older lower leaves are often affected first.",
            "Leaves may dry and fall prematurely."
        ],
        "solutions": [
            "Remove heavily infected lower leaves.",
            "Avoid wetting foliage during irrigation.",
            "Improve airflow around plants.",
            "Use an appropriate registered fungicide only with local expert guidance.",
            "Monitor nearby plants for spreading symptoms."
        ],
        "prevention": [
            "Rotate crops with non-host plants.",
            "Remove infected plant debris.",
            "Use healthy seedlings.",
            "Maintain proper spacing.",
            "Avoid excessive nitrogen fertilisation."
        ]
    },

    "Tomato_healthy": {
        "crop": "Tomato",
        "disease": "Healthy",
        "cause": "No disease detected by the classifier.",
        "description": "The uploaded leaf was classified as healthy. Continue monitoring because a single image cannot rule out all diseases.",
        "symptoms": [
            "Generally green and intact leaf tissue.",
            "No obvious characteristic disease spots.",
            "No clear widespread leaf damage."
        ],
        "solutions": [
            "No disease-specific treatment is indicated.",
            "Continue normal crop care.",
            "Monitor for new spots, curling or discolouration."
        ],
        "prevention": [
            "Use healthy seedlings.",
            "Maintain balanced irrigation and nutrition.",
            "Provide sufficient plant spacing.",
            "Inspect plants regularly."
        ]
    },

    "Tomato_Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "cause": "Oomycete Phytophthora infestans.",
        "description": "A rapidly spreading disease that can damage tomato leaves, stems and fruits, particularly in cool and wet conditions.",
        "symptoms": [
            "Irregular water-soaked patches on leaves.",
            "Dark brown lesions that expand quickly.",
            "White growth may appear underneath leaves in humid conditions.",
            "Dark lesions can develop on stems.",
            "Fruits may develop firm, brown or greasy-looking patches."
        ],
        "solutions": [
            "Remove severely infected plant material.",
            "Avoid overhead watering.",
            "Reduce prolonged leaf wetness.",
            "Seek prompt expert guidance on locally registered late-blight control products.",
            "Follow all product label and harvest-safety instructions."
        ],
        "prevention": [
            "Use healthy seedlings.",
            "Remove infected plant debris.",
            "Maintain good airflow.",
            "Avoid prolonged wet foliage.",
            "Monitor crops closely during humid weather."
        ]
    },

    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "cause": "Fungus Passalora fulva.",
        "description": "A fungal disease favoured by high humidity and poor air circulation.",
        "symptoms": [
            "Pale green or yellow patches on upper leaf surfaces.",
            "Olive-green or brown velvety growth underneath leaves.",
            "Affected leaves may curl and dry.",
            "Severe infections can cause premature leaf drop."
        ],
        "solutions": [
            "Remove heavily affected leaves.",
            "Improve ventilation around plants.",
            "Reduce humidity where practical.",
            "Avoid overhead irrigation.",
            "Seek local advice before using a registered fungicide."
        ],
        "prevention": [
            "Maintain sufficient plant spacing.",
            "Improve greenhouse ventilation where applicable.",
            "Remove infected plant debris.",
            "Avoid prolonged leaf wetness.",
            "Use healthy planting material."
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "cause": "Fungus Septoria lycopersici.",
        "description": "A fungal leaf disease that often starts on older, lower leaves.",
        "symptoms": [
            "Small circular spots with dark margins.",
            "Grey or tan centres within spots.",
            "Tiny dark dots may be visible inside lesions.",
            "Yellowing and premature leaf drop."
        ],
        "solutions": [
            "Remove affected lower leaves without spreading infected material.",
            "Avoid overhead watering.",
            "Keep foliage as dry as practical.",
            "Improve airflow around plants.",
            "Consult an agricultural expert about appropriate registered disease-control products."
        ],
        "prevention": [
            "Rotate crops with non-host plants.",
            "Remove infected crop debris.",
            "Use disease-free seedlings.",
            "Maintain proper plant spacing.",
            "Disinfect gardening tools appropriately."
        ]
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mites",
        "cause": "Infestation by Tetranychus urticae, a plant-feeding mite.",
        "description": "A pest infestation rather than a fungal or bacterial disease. Mites commonly thrive in hot, dry conditions.",
        "symptoms": [
            "Fine pale speckling on leaf surfaces.",
            "Yellow or bronze discolouration.",
            "Fine webbing on the undersides of leaves in heavier infestations.",
            "Leaves may dry and fall."
        ],
        "solutions": [
            "Inspect leaf undersides to confirm mite presence.",
            "Remove heavily infested leaves where practical.",
            "Use a strong but gentle water spray where suitable to dislodge mites.",
            "Protect beneficial predatory insects.",
            "Seek local guidance before using a registered miticide or other control product."
        ],
        "prevention": [
            "Inspect plants regularly, especially leaf undersides.",
            "Avoid excessive plant stress from water shortage.",
            "Control weeds that may host mites.",
            "Avoid unnecessary broad-spectrum pesticide use.",
            "Encourage beneficial predatory mites where appropriate."
        ]
    },

    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "cause": "Fungus Corynespora cassiicola.",
        "description": "A fungal disease that can affect tomato leaves, stems and fruits, particularly under favourable humidity.",
        "symptoms": [
            "Brown circular lesions with concentric rings.",
            "Spots may enlarge and merge.",
            "Leaf yellowing and premature leaf drop.",
            "Dark lesions may develop on fruits."
        ],
        "solutions": [
            "Remove severely affected leaves.",
            "Avoid overhead irrigation.",
            "Improve airflow and reduce prolonged leaf wetness.",
            "Seek expert guidance on locally registered fungicide options."
        ],
        "prevention": [
            "Remove infected plant debris.",
            "Maintain good plant spacing.",
            "Use healthy seedlings.",
            "Rotate crops where practical.",
            "Monitor leaves regularly."
        ]
    },

    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "cause": "Tomato mosaic virus (ToMV).",
        "description": "A viral disease that can affect leaf development and plant growth. There is no curative treatment for an infected plant.",
        "symptoms": [
            "Light and dark green mosaic patterns on leaves.",
            "Leaf distortion or curling.",
            "Reduced plant growth.",
            "Uneven fruit development in some cases."
        ],
        "solutions": [
            "Remove and safely dispose of confirmed infected plants where appropriate.",
            "Avoid handling healthy plants immediately after infected plants.",
            "Clean tools and hands to reduce mechanical transmission.",
            "Do not attempt to cure viral infection using fungicides.",
            "Seek expert confirmation before removing valuable plants."
        ],
        "prevention": [
            "Use certified virus-free seeds and seedlings.",
            "Sanitise tools and equipment.",
            "Avoid tobacco contamination when handling plants.",
            "Remove infected plant debris.",
            "Use resistant varieties where available."
        ]
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "cause": "Tomato yellow leaf curl virus complex, commonly transmitted by whiteflies.",
        "description": "A viral disease that can severely affect tomato growth and fruit production.",
        "symptoms": [
            "Upward curling of leaves.",
            "Yellowing of leaf margins.",
            "Smaller leaves and shortened internodes.",
            "Stunted plant growth.",
            "Poor fruit setting."
        ],
        "solutions": [
            "Remove confirmed infected plants where appropriate to reduce sources of infection.",
            "Inspect plants for whiteflies.",
            "Use suitable insect-exclusion netting where practical.",
            "Seek local advice for integrated whitefly management.",
            "Do not use fungicides as a treatment for viral infection."
        ],
        "prevention": [
            "Use healthy seedlings.",
            "Grow resistant varieties where available.",
            "Monitor and manage whitefly populations.",
            "Remove nearby weed hosts.",
            "Use insect-proof nursery protection where practical."
        ]
    },

    "Pepper__bell___Bacterial_spot": {
        "crop": "Pepper",
        "disease": "Bacterial Spot",
        "cause": "Xanthomonas species of bacteria.",
        "description": "A bacterial disease affecting pepper leaves, stems and fruits.",
        "symptoms": [
            "Small dark water-soaked spots on leaves.",
            "Spots may develop yellow halos.",
            "Leaf lesions may merge and cause defoliation.",
            "Raised or scabby fruit lesions."
        ],
        "solutions": [
            "Remove severely infected plant material.",
            "Avoid handling plants when wet.",
            "Reduce overhead irrigation.",
            "Clean tools between plants.",
            "Consult local experts about suitable registered management products."
        ],
        "prevention": [
            "Use disease-free seeds and transplants.",
            "Maintain good plant spacing.",
            "Rotate crops with non-host plants.",
            "Remove infected debris.",
            "Avoid working in wet fields."
        ]
    },

    "Pepper__bell___healthy": {
        "crop": "Pepper",
        "disease": "Healthy",
        "cause": "No disease detected by the classifier.",
        "description": "The model classified the pepper leaf as healthy. Regular monitoring is still necessary.",
        "symptoms": [
            "Generally green leaf tissue.",
            "No obvious characteristic disease lesions.",
            "No clear widespread leaf damage."
        ],
        "solutions": [
            "No disease-specific treatment is indicated.",
            "Continue normal crop care.",
            "Monitor for new spots, curling or discolouration."
        ],
        "prevention": [
            "Use healthy seedlings.",
            "Maintain balanced watering and nutrition.",
            "Ensure adequate spacing.",
            "Inspect plants regularly."
        ]
    }
}


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "latest_assessment" not in st.session_state:
    st.session_state.latest_assessment = {}

if "last_plant_result" not in st.session_state:
    st.session_state.last_plant_result = None

if "last_farm_result" not in st.session_state:
    st.session_state.last_farm_result = None

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "navigation" not in st.session_state:
    st.session_state.navigation = st.session_state.page


def navigate_to(target):
    st.session_state["page"] = target
    st.session_state["navigation"] = target



# ==================================================
# SIDEBAR NAVIGATION
# ==================================================

pages = [
    "Dashboard",
    "Plant Doctor",
    "Farmer Assistant",
    "Farm Monitoring",
    "Reports & History",
    "Edge AI Lab",
]

with st.sidebar:

    st.markdown("""
    <div style="padding:18px 0 22px;">
        <div style="font-size:28px;">🌱</div>
        <div style="font-size:23px;font-weight:850;color:#ffffff;">
            AgriSense AI
        </div>
        <div style="font-size:10px;letter-spacing:2px;
                    color:#19d9f5;margin-top:7px;">
            PRECISION AGRICULTURE
        </div>
        <div style="font-size:11px;color:#8295a5;margin-top:10px;">
            Intelligence workspace · Prototype
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        pages,
        key="navigation",
        label_visibility="collapsed"
    )

    st.session_state["page"] = page

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div style="font-size:12px;color:#9aa9b7;line-height:2;">
        Local plant inference + online AI assistant
        <br><br>
        No physical sensors are currently connected.
    </div>
    """, unsafe_allow_html=True)





# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    status = model_status()
    records = get_records()
    api_ready = bool(os.getenv("GEMINI_API_KEY"))

    model_value = "Ready" if status["ready"] else "Offline"
    api_value = "Connected" if api_ready else "Offline"

    model_color = "#42e6a0" if status["ready"] else "#ff647c"
    api_color = "#42e6a0" if api_ready else "#ff647c"

    # --------------------------------------------------
    # HERO SECTION
    # --------------------------------------------------

    st.html("""
    <div style="
        background:
            radial-gradient(circle at 90% 10%,#17465a 0%,transparent 40%),
            linear-gradient(120deg,#102b3a,#142331,#101923);
        border:1px solid #245064;
        border-radius:20px;
        padding:34px 30px;
        margin-bottom:28px;
        position:relative;
        overflow:hidden;
    ">
        <div style="
            color:#19d9f5;
            font-size:12px;
            font-weight:800;
            letter-spacing:3px;
        ">
            AGRISENSE AI · SMART AGRICULTURE
        </div>

        <div style="
            color:white;
            font-size:42px;
            font-weight:850;
            margin-top:20px;
            line-height:1.15;
        ">
            Grow smarter.
        </div>

        <div style="
            color:#19d9f5;
            font-size:25px;
            font-weight:650;
            margin-top:5px;
        ">
            Think beyond the field.
        </div>

        <div style="
            color:#b8cbd7;
            font-size:15px;
            line-height:1.8;
            max-width:650px;
            margin-top:18px;
        ">
            An intelligent agriculture platform combining local plant
            disease classification, AI-powered farmer assistance
            and farm-condition monitoring.
        </div>

        <div style="margin-top:23px;">
            <span style="background:#173c4d;color:#19d9f5;
                padding:8px 12px;border-radius:20px;
                font-size:11px;margin:4px;display:inline-block;">
                LOCAL EDGE AI
            </span>
            <span style="background:#173c4d;color:#19d9f5;
                padding:8px 12px;border-radius:20px;
                font-size:11px;margin:4px;display:inline-block;">
                15 PLANT CLASSES
            </span>
            <span style="background:#173c4d;color:#19d9f5;
                padding:8px 12px;border-radius:20px;
                font-size:11px;margin:4px;display:inline-block;">
                GEMINI AI
            </span>
            <span style="background:#173c4d;color:#19d9f5;
                padding:8px 12px;border-radius:20px;
                font-size:11px;margin:4px;display:inline-block;">
                FARM MONITORING
            </span>
        </div>
    </div>
    """)

    # --------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-eyebrow">SYSTEM OVERVIEW</div>',
        unsafe_allow_html=True
    )

    st.subheader("Your intelligence workspace")
    st.caption("Current application and service status.")

    st.html(f"""
    <style>
    .agri-status-grid {{
        display:grid;
        grid-template-columns:repeat(3,minmax(0,1fr));
        gap:14px;
        margin:16px 0 28px;
    }}

    .agri-status-card {{
        background:linear-gradient(145deg,#172635,#101923);
        border:1px solid #294052;
        border-radius:16px;
        padding:22px;
        min-height:145px;
        box-sizing:border-box;
    }}

    .agri-status-label {{
        color:#aabac9;
        font-size:13px;
        margin-bottom:18px;
    }}

    .agri-status-value {{
        color:#f4f7fb;
        font-size:26px;
        font-weight:800;
        display:flex;
        align-items:center;
        gap:10px;
    }}

    .agri-status-dot {{
        width:10px;
        height:10px;
        border-radius:50%;
        display:inline-block;
    }}

    .agri-status-description {{
        color:#8495a5;
        font-size:12px;
        margin-top:12px;
    }}

    @media(max-width:700px) {{
        .agri-status-grid {{
            grid-template-columns:1fr;
        }}
    }}
    </style>

    <div class="agri-status-grid">

        <div class="agri-status-card">
            <div class="agri-status-label">🌿 Plant AI Model</div>
            <div class="agri-status-value">
                <span class="agri-status-dot"
                    style="background:{model_color};"></span>
                {model_value}
            </div>
            <div class="agri-status-description">
                Local plant disease classification
            </div>
        </div>

        <div class="agri-status-card">
            <div class="agri-status-label">🤖 AI Farmer Assistant</div>
            <div class="agri-status-value">
                <span class="agri-status-dot"
                    style="background:{api_color};"></span>
                {api_value}
            </div>
            <div class="agri-status-description">
                Gemini API connection status
            </div>
        </div>

        <div class="agri-status-card">
            <div class="agri-status-label">📁 Saved Assessments</div>
            <div class="agri-status-value">
                {len(records)}
            </div>
            <div class="agri-status-description">
                Stored assessment records
            </div>
        </div>

    </div>
    """)

    # --------------------------------------------------
    # QUICK ACCESS TOOLS
    # --------------------------------------------------

    st.markdown(
        '<div class="section-eyebrow">YOUR TOOLS</div>',
        unsafe_allow_html=True
    )

    st.subheader("Explore AgriSense")
    st.caption("Select a module to access its features.")

    tool_data = [
        {
            "icon": "🌿",
            "title": "Plant Doctor",
            "description": "Analyze plant leaf images and identify possible diseases using a local AI model.",
            "button": "Open Plant Doctor →",
            "page": "Plant Doctor",
            "tag": "EDGE AI"
        },
        {
            "icon": "🤖",
            "title": "Farmer Assistant",
            "description": "Ask agricultural questions and receive AI-powered crop-management guidance.",
            "button": "Open Assistant →",
            "page": "Farmer Assistant",
            "tag": "GEMINI AI"
        },
        {
            "icon": "📊",
            "title": "Farm Monitoring",
            "description": "Monitor environmental readings, review historical trends and identify potential risks.",
            "button": "Open Monitoring →",
            "page": "Farm Monitoring",
            "tag": "FIELD INTELLIGENCE"
        }
    ]

    tool_cols = st.columns(3, gap="medium")

    for col, tool in zip(tool_cols, tool_data):

        with col:

            with st.container(border=True):

                st.markdown(
                    f"""
                    <div style="
                        font-size:32px;
                        margin-bottom:14px;
                    ">
                        {tool['icon']}
                    </div>

                    <div style="
                        color:#f4f7fb;
                        font-size:21px;
                        font-weight:800;
                        margin-bottom:10px;
                    ">
                        {tool['title']}
                    </div>

                    <div style="
                        color:#19d9f5;
                        font-size:10px;
                        font-weight:800;
                        letter-spacing:1.5px;
                        margin-bottom:12px;
                    ">
                        {tool['tag']}
                    </div>

                    <div style="
                        color:#9aabba;
                        font-size:13px;
                        line-height:1.7;
                        min-height:88px;
                    ">
                        {tool['description']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.button(
                    tool["button"],
                    key=f"tool_{tool['page']}",
                    use_container_width=True,
                    on_click=navigate_to,
                    args=(tool["page"],)
                )

    # --------------------------------------------------
    # PLATFORM CAPABILITIES
    # --------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-eyebrow">PLATFORM CAPABILITIES</div>',
        unsafe_allow_html=True
    )

    st.subheader("One platform. Multiple intelligence tools.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🧠 AI-powered analysis")
        st.write(
            "Plant disease classification, agricultural question "
            "answering and automated condition screening."
        )

        st.markdown("### 🌱 Agriculture-focused insights")
        st.write(
            "Designed to help users understand plant health and "
            "environmental conditions."
        )

    with c2:
        st.markdown("### ⚡ Edge AI approach")
        st.write(
            "Plant image classification uses a locally loaded model "
            "rather than requiring a cloud inference API."
        )

        st.markdown("### 📡 Expandable architecture")
        st.write(
            "Designed with a future pathway for ESP32-based "
            "environmental sensor integration."
        )

    st.caption(
        "AgriSense AI is an academic prototype. Its outputs are "
        "preliminary decision-support information, not guaranteed "
        "agronomic diagnoses."
    )


    
    
    # WORKFLOW

    st.markdown(
        '<div class="section-eyebrow">WORKFLOW</div>',
        unsafe_allow_html=True
    )

    st.subheader("How AgriSense works")
    st.caption(
        "A simple process to turn plant data into actionable insights."
    )

    workflow = [
        ("01", "📤", "Upload", "Upload a clear plant leaf image."),
        ("02", "🔬", "Analyze", "AI analyzes the image locally."),
        ("03", "📋", "Review", "Review disease predictions and guidance."),
        ("04", "💾", "Save", "Store your assessment for future reference.")
    ]

    w1, w2, w3, w4 = st.columns(4, gap="medium")

    for col, (number, emoji, title, description) in zip(
        [w1, w2, w3, w4], workflow
    ):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<p style='color:#19d9f5;font-size:12px;"
                    f"font-weight:800;letter-spacing:2px;'>"
                    f"STEP {number}</p>",
                    unsafe_allow_html=True
                )

                st.markdown(f"## {emoji}")
                st.markdown(f"### {title}")
                st.caption(description)

    if not status["ready"]:
        st.warning(
            "Plant model not configured. Check your model files."
        )

    if not api_ready:
        st.info(
            "Gemini API key not configured. Check your .env file."
        )






# ==================================================
# PLANT DOCTOR
# ==================================================

elif page == "Plant Doctor":

    
    st.html("""
    <div style="
        background:linear-gradient(120deg,#102b3a,#142331,#10202b);
        border:1px solid #245064;
        border-radius:18px;
        padding:28px;
        margin-bottom:22px;
    ">
        <div style="color:#19d9f5;font-size:14px;font-weight:700;
                    letter-spacing:2px;">
            AGRISENSE AI · EDGE INTELLIGENCE
        </div>
        <div style="font-size:36px;font-weight:800;color:#ffffff;
                    margin-top:10px;">
            🌿 AI Plant Doctor
        </div>
        <div style="font-size:18px;color:#b8cbd7;margin-top:10px;
                    line-height:1.7;">
            Detect plant diseases, understand symptoms and explore
            AI-powered crop health guidance.
        </div>
        <div style="font-size:15px;color:#19d9f5;margin-top:18px;">
            15 CLASSES &nbsp; · &nbsp; 3 CROPS &nbsp; · &nbsp; LOCAL AI MODEL
        </div>
    </div>
    """)

    st.subheader("Upload Plant Image")
    st.write(
        "Upload a clear image of a tomato, potato or pepper leaf "
        "for AI-based disease classification."
    )

    st.caption(
        "Supported categories: 10 Tomato classes, 3 Potato classes "
        "and 2 Pepper classes."
    )


    uploaded = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"],
        key="plant_image_upload"
    )

    if uploaded:

        from PIL import Image

        image_bytes = uploaded.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        image = Image.open(uploaded).convert("RGB")

        st.markdown("### 📷 Image Preview")

        with st.container(border=True):
            st.image(
                image,
                caption="Uploaded plant image",
                width=420,
                use_container_width=False
            )
            st.caption("Image uploaded successfully. Ready for AI analysis.")

        status = model_status()

        if not status["ready"]:

            st.warning(
                "15-class plant disease model or labels are missing."
            )

        else:

            if st.button("🔍 Analyze Plant", type="primary"):

                with st.spinner("Analyzing plant image locally..."):

                    try:
                        result = predict_leaf_disease(image)

                    except Exception as error:
                        result = {
                            "error": str(error)
                        }

                st.session_state.last_plant_result = {
                    "result": result,
                    "image_name": uploaded.name,
                    "image_hash": image_hash
                }

                if not result.get("error"):

                    st.session_state.latest_assessment = {
                        "type": "plant_image",
                        "crop": result.get("crop", "Unknown"),
                        "result": result
                    }

            saved = st.session_state.last_plant_result

            if saved and saved.get("image_hash") == image_hash:

                result = saved["result"]

                if result.get("error"):

                    st.error(result["error"])

                else:

                    label = result.get("label", "Unknown")
                    guide = DISEASE_GUIDE.get(label)

                    st.divider()
                    st.header("🌿 Plant Health Report")

                    # DIAGNOSIS SUMMARY

                    st.subheader("Diagnosis Summary")

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.metric(
                            "Identified Crop",
                            result.get("crop", "Unknown")
                        )

                    with c2:
                        st.metric(
                            "Predicted Condition",
                            result.get("display_label", label)
                        )

                    with c3:
                        confidence = result.get("confidence", 0.0)

                        st.metric(
                            "Prediction Score",
                            f"{confidence:.2%}"
                        )

                    st.caption(
                        "Prediction score represents the model's confidence "
                        "output, not a confirmed disease probability."
                    )

                    # CONFIDENCE WARNING

                    if result.get("low_confidence"):

                        st.warning(
                            "⚠️ Low Confidence Prediction: "
                            + result.get(
                                "confidence_message",
                                "The model is uncertain. Upload a clearer "
                                "image or consult an agriculture expert."
                            )
                        )

                    else:

                        st.success(
                            result.get(
                                "confidence_message",
                                "The model has relatively high confidence "
                                "in this prediction."
                            )
                        )

                    # DISEASE INFORMATION

                    st.subheader("Disease Information")

                    if guide:

                        overview_tab, symptoms_tab, solutions_tab, prevention_tab = st.tabs(
                            ["Overview", "Symptoms", "Solutions", "Prevention"]
                        )

                        with overview_tab:

                            st.markdown(f"### {guide['disease']}")

                            st.write(guide["description"])

                            st.info(
                                f"**Likely cause:** {guide['cause']}"
                            )

                            if guide["disease"] == "Healthy":

                                st.success(
                                    "The classifier predicts a healthy leaf. "
                                    "Continue routine monitoring. This does not "
                                    "rule out every possible plant health issue."
                                )

                        with symptoms_tab:

                            for symptom in guide["symptoms"]:
                                st.markdown(f"• {symptom}")

                        with solutions_tab:

                            for index, solution in enumerate(
                                guide["solutions"], start=1
                            ):
                                st.markdown(f"**{index}.** {solution}")

                        with prevention_tab:

                            for prevention in guide["prevention"]:
                                st.markdown(f"• {prevention}")

                        if guide["disease"] != "Healthy":

                            st.warning(
                                "Use these recommendations as preliminary "
                                "guidance. Confirm the disease before applying "
                                "chemical treatments."
                            )

                    else:

                        st.warning(
                            "No disease information is available for this "
                            "model label. Check DISEASE_GUIDE in app.py."
                        )

                    # ALTERNATIVE PREDICTIONS

                    st.divider()
                    st.subheader("Other Model Predictions")

                    st.caption(
                        "These are the model's highest-scoring alternative "
                        "classes. They are not independent disease diagnoses."
                    )

                    predictions = result.get("top_predictions", [])

                    if predictions:

                        seen_predictions = set()

                        for item in predictions:

                            raw_label = item.get("label", "Unknown")

                            if raw_label in seen_predictions:
                                continue

                            seen_predictions.add(raw_label)

                            if raw_label.startswith("Tomato"):
                                crop_name = "Tomato"

                            elif raw_label.startswith("Potato"):
                                crop_name = "Potato"

                            elif raw_label.startswith("Pepper"):
                                crop_name = "Pepper"

                            else:
                                crop_name = "Unknown"

                            display = item.get(
                                "display_label",
                                raw_label
                            )

                            prediction_confidence = item.get(
                                "confidence", 0.0
                            )

                            st.write(
                                f"**{crop_name} - {display}** — "
                                f"{prediction_confidence:.2%}"
                            )

                            st.progress(
                                min(max(prediction_confidence, 0.0), 1.0)
                            )

                    else:

                        st.caption(
                            "Alternative predictions are not available."
                        )

                    # SAVE ASSESSMENT

                    st.divider()
                    st.subheader("Assessment Record")

                    if st.button(
                        "💾 Save Plant Assessment",
                        key="save_plant_assessment"
                    ):

                        save_record(
                            "plant_image",
                            result.get("crop", "Unknown"),
                            {
                                "image_name": saved["image_name"],
                                "label": label,
                                "display_label": result.get(
                                    "display_label", label
                                ),
                                "confidence": confidence,
                                "top_predictions": predictions,
                                "disease_information": guide,
                                "latency_ms": result.get("latency_ms")
                            }
                        )

                        st.success(
                            "Detailed plant assessment saved successfully."
                        )

                    # DISCLAIMER

                    st.caption(
                        "Disclaimer: AgriSense AI provides an AI-based "
                        "classification and general agricultural guidance. "
                        "It is not a laboratory-confirmed diagnosis. "
                        "Actual symptoms and treatment should be verified "
                        "by a qualified agricultural expert."
                    )


# ==================================================
# FARMER ASSISTANT
# ==================================================

elif page == "Farmer Assistant":

    
    st.html("""
    <div style="
        background:linear-gradient(120deg,#102b3a,#142331,#10202b);
        border:1px solid #245064;
        border-radius:18px;
        padding:28px;
        margin-bottom:22px;
    ">
        <div style="color:#19d9f5;font-size:14px;font-weight:700;
                    letter-spacing:2px;">
            AGRISENSE AI · INTELLIGENT ASSISTANCE
        </div>
        <div style="font-size:36px;font-weight:800;color:#ffffff;
                    margin-top:10px;">
            💬 Farmer Assistant
        </div>
        <div style="font-size:18px;color:#b8cbd7;margin-top:10px;
                    line-height:1.7;">
            Your AI-powered agricultural companion. Ask questions
            about crop health, diseases, irrigation and farming practices.
        </div>
        <div style="font-size:15px;color:#19d9f5;margin-top:18px;">
            GEMINI AI &nbsp; · &nbsp; CROP GUIDANCE &nbsp; · &nbsp; SMART SUPPORT
        </div>
    </div>
    """)

    st.subheader("Ask Your Agricultural Question")
    st.caption(
        "You can also discuss your latest Plant Doctor or Farm Monitoring assessment."
    )


    with st.expander("How answers are generated"):

        st.write(
            "The assistant uses an online AI model and available application "
            "context. Results are informational, not a confirmed diagnosis. "
            "The application has no live sensors."
        )

    context_parts = []

    latest = st.session_state.get("latest_assessment")

    if latest:

        context_parts.append(
            "Latest assessment from the application:\n"
            + json.dumps(
                latest,
                ensure_ascii=False,
                default=str
            )
        )

    else:

        context_parts.append(
            "No recent Plant Doctor or Farm Monitoring assessment is available."
        )

    context_parts.append(
        "Farm readings may be manually entered and are not verified live "
        "sensor measurements. Treat model predictions as uncertain."
    )

    context = "\n\n".join(context_parts)

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
    "🌱 Ask anything about your crops, diseases or farming..."
)

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Consulting the AI assistant..."):

                answer = ask_farmer_assistant(
                    question=question,
                    farm_context=context,
                    history=st.session_state.messages[:-1],
                )

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })

    if st.button("🗑️ Clear Chat", type="secondary"):

        st.session_state.messages = []
        st.rerun()



# ==================================================
# FARM MONITORING
# ==================================================

elif page == "Farm Monitoring":

    import pandas as pd
    from datetime import datetime

    # Initialize session history
    if "monitoring_history" not in st.session_state:
        st.session_state.monitoring_history = []

    if "last_farm_result" not in st.session_state:
        st.session_state.last_farm_result = None

    st.html("""
    <div style="
        background:linear-gradient(120deg,#102b3a,#142331,#10202b);
        border:1px solid #245064;
        border-radius:18px;
        padding:28px;
        margin-bottom:22px;
    ">
        <div style="color:#19d9f5;font-size:14px;font-weight:700;
                    letter-spacing:2px;">
            AGRISENSE AI · FIELD INTELLIGENCE
        </div>
        <div style="font-size:36px;font-weight:800;color:#ffffff;
                    margin-top:10px;">
            📊 Farm Monitoring
        </div>
        <div style="font-size:18px;color:#b8cbd7;margin-top:10px;
                    line-height:1.7;">
            Monitor environmental conditions and identify potential
            crop risks through intelligent rule-based screening.
        </div>
        <div style="font-size:15px;color:#19d9f5;margin-top:18px;">
            SOIL MOISTURE · TEMPERATURE · HUMIDITY · RAINFALL
        </div>
    </div>
    """)

    st.info("📡 Current Mode: Manual input / simulated readings. No physical IoT sensors are connected.")

    # --------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------

    st.subheader("🌱 Enter Farm Conditions")

    with st.form("farm_form"):

        a, b = st.columns(2)

        with a:
            crop = st.selectbox(
                "Crop",
                ["Wheat", "Rice", "Tomato", "Potato", "Maize", "Other"],
            )

            moisture = st.slider(
                "Soil moisture (%)", 0, 100, 25
            )

            temperature = st.slider(
                "Temperature (°C)", -5, 55, 30
            )

        with b:
            humidity = st.slider(
                "Relative humidity (%)", 0, 100, 55
            )

            rainfall = st.number_input(
                "Recent rainfall (mm)",
                min_value=0.0,
                max_value=500.0,
                value=0.0,
                step=1.0,
            )

            notes = st.text_input("Optional field notes")

        submitted = st.form_submit_button(
            "🔍 Analyze Farm Conditions",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------------------
    # ANALYSIS AND HISTORY STORAGE
    # --------------------------------------------------

    if submitted:
        try:
            result = assess_conditions(
                crop,
                moisture,
                temperature,
                humidity,
                rainfall,
                notes
            )

            st.session_state.last_farm_result = result

            st.session_state.latest_assessment = {
                "type": "farm_conditions",
                "inputs": result["inputs"],
                "alerts": result["alerts"],
            }

            # Save reading in session history
            st.session_state.monitoring_history.append({
                "Time": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M:%S"),
                "Crop": crop,
                "Temperature": temperature,
                "Humidity": humidity,
                "Soil Moisture": moisture,
                "Rainfall": rainfall,
            })

            st.success("Farm conditions analyzed successfully!")

        except Exception as error:
            st.error(f"Unable to assess farm conditions: {error}")

    result = st.session_state.last_farm_result

    # --------------------------------------------------
    # INTERACTIVE MONITORING DASHBOARD
    # --------------------------------------------------

    if result:

        inputs = result["inputs"]

        st.markdown("---")
        st.subheader("📊 Live Monitoring Dashboard")
        st.caption("Latest submitted farm readings")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "🌡️ Temperature",
                f'{inputs["temperature_c"]} °C'
            )

        with c2:
            st.metric(
                "💧 Humidity",
                f'{inputs["relative_humidity_percent"]}%'
            )

        with c3:
            st.metric(
                "🌱 Soil Moisture",
                f'{inputs["soil_moisture_percent"]}%'
            )

        st.caption(f'Crop: {inputs["crop"]}')

        # --------------------------------------------------
        # SMART RECOMMENDATIONS
        # --------------------------------------------------

        st.markdown("---")
        st.subheader("🧠 Smart Recommendations")

        if result.get("alerts"):

            for alert in result["alerts"]:

                message = (
                    f'**{alert["title"]}:** {alert["detail"]}'
                )

                if alert["level"] == "High":
                    st.error(message)

                elif alert["level"] == "Moderate":
                    st.warning(message)

                else:
                    st.info(message)

        else:
            st.success(
                "No rule-based alerts were triggered for these inputs. "
                "This does not guarantee the crop is healthy."
            )

        st.caption(
            "These are preliminary screening heuristics, not "
            "crop-specific agronomic recommendations."
        )

        # --------------------------------------------------
        # HISTORICAL CHARTS
        # --------------------------------------------------

        st.markdown("---")
        st.subheader("📈 Historical Monitoring Charts")

        history = st.session_state.monitoring_history

        if len(history) >= 2:

            df = pd.DataFrame(history)

            st.caption(
                "Session history: manually submitted readings only."
            )

            st.markdown("**Temperature Trend (°C)**")
            st.line_chart(
                df.set_index("Time")[["Temperature"]]
            )

            st.markdown("**Humidity Trend (%)**")
            st.line_chart(
                df.set_index("Time")[["Humidity"]]
            )

            st.markdown("**Soil Moisture Trend (%)**")
            st.line_chart(
                df.set_index("Time")[["Soil Moisture"]]
            )

        else:
            st.info(
                "Submit at least two farm assessments to generate "
                "historical charts."
            )

        # --------------------------------------------------
        # SAVE ASSESSMENT
        # --------------------------------------------------

        if st.button("💾 Save Monitoring Assessment"):

            save_record(
                "farm_conditions",
                inputs["crop"],
                result,
            )

            st.success("Monitoring assessment saved successfully!")

    # --------------------------------------------------
    # FUTURE IOT INTEGRATION
    # --------------------------------------------------

    st.markdown("---")
    st.subheader("📡 Future IoT Integration")

    st.caption(
        "Planned architecture — hardware integration is not currently active."
    )

    with st.expander("Explore the proposed IoT architecture"):

        st.markdown("""
        **Proposed hardware components**

        - ESP32 microcontroller — processes and transmits sensor readings.
        - DHT22 sensor — measures temperature and humidity.
        - Capacitive soil moisture sensor — measures soil moisture.

        **Data flow**

        1. Sensors collect environmental readings.
        2. ESP32 reads and processes sensor data.
        3. Data is transmitted through Wi-Fi using HTTP or MQTT.
        4. AgriSense AI receives the readings.
        5. The monitoring dashboard displays updated values.
        6. Rule-based screening identifies potential risks.
        """)

        st.code(
            "DHT22 + Soil Moisture Sensor\n"
            "          ↓\n"
            "        ESP32\n"
            "          ↓\n"
            "      Wi-Fi / MQTT\n"
            "          ↓\n"
            "   AgriSense AI Backend\n"
            "          ↓\n"
            " Monitoring Dashboard + Alerts",
            language="text"
        )

        st.info(
            "This is a proposed future extension. The current application "
            "uses manual inputs and does not receive live sensor data."
        )



# ==================================================
# REPORTS & HISTORY
# ==================================================

elif page == "Reports & History":

    import json

    st.html("""
    <div style="
        background:linear-gradient(120deg,#102b3a,#142331,#10202b);
        border:1px solid #245064;
        border-radius:18px;
        padding:28px;
        margin-bottom:22px;
    ">
        <div style="color:#19d9f5;font-size:14px;font-weight:700;
                    letter-spacing:2px;">
            AGRISENSE AI · DATA CENTER
        </div>
        <div style="font-size:36px;font-weight:800;color:#ffffff;
                    margin-top:10px;">
            📚 Reports & History
        </div>
        <div style="font-size:18px;color:#b8cbd7;margin-top:10px;
                    line-height:1.7;">
            Review previous plant diagnoses, farm assessments
            and export your agricultural reports.
        </div>
        <div style="font-size:15px;color:#19d9f5;margin-top:18px;">
            SAVED RECORDS · ANALYTICS · REPORT EXPORT
        </div>
    </div>
    """)

    records = get_records()

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------

    plant_records = [
        r for r in records
        if r["record_type"] == "plant_image"
    ]

    farm_records = [
        r for r in records
        if r["record_type"] == "farm_conditions"
    ]

    st.subheader("📊 Assessment Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Assessments", len(records))
    c2.metric("Plant Diagnoses", len(plant_records))
    c3.metric("Farm Assessments", len(farm_records))

    st.markdown("---")

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    st.subheader("📥 Export Reports")

    if records:

        report_data = {
            "application": "AgriSense AI",
            "report_type": "Assessment History",
            "total_assessments": len(records),
            "plant_diagnoses": len(plant_records),
            "farm_assessments": len(farm_records),
            "records": records
        }

        st.download_button(
            "⬇️ Download Complete Assessment Report (JSON)",
            data=json.dumps(
                report_data,
                indent=4,
                default=str
            ),
            file_name="AgriSense_Assessment_Report.json",
            mime="application/json",
            use_container_width=True
        )

    else:
        st.info("Complete and save an assessment to enable report export.")

    st.markdown("---")

    # --------------------------------------------------
    # ASSESSMENT HISTORY
    # --------------------------------------------------

    st.subheader("📁 Saved Assessment Records")

    if not records:

        st.info(
            "No records available yet. Complete and save a Plant Doctor "
            "or Farm Monitoring assessment to see your history here."
        )

    else:

        st.caption(f"{len(records)} saved assessment(s) available.")

        for index, record in enumerate(records):

            with st.expander(
                f'{record["created_at"]} · '
                f'{record["record_type"]} · {record["crop"]}'
            ):

                payload = record["payload"]

                # PLANT DISEASE RECORDS

                if record["record_type"] == "plant_image":

                    st.markdown("### 🌿 Plant Disease Assessment")

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Crop", record["crop"])

                    c2.metric(
                        "Predicted Condition",
                        payload.get("display_label", "Unknown")
                    )

                    c3.metric(
                        "Prediction Score",
                        f'{payload.get("confidence", 0):.2%}'
                    )

                    st.caption(
                        f'Uploaded image: '
                        f'{payload.get("image_name", "Not recorded")}'
                    )

                    latency = payload.get("latency_ms")

                    if latency is not None:
                        st.caption(
                            f"Inference time: {latency:.1f} ms"
                        )

                    with st.expander("View detailed assessment data"):
                        st.json(payload)

                # FARM MONITORING RECORDS

                elif record["record_type"] == "farm_conditions":

                    st.markdown("### 📊 Farm Condition Assessment")

                    inputs = payload.get("inputs", {})

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Temperature",
                        f'{inputs.get("temperature_c", 0)} °C'
                    )

                    c2.metric(
                        "Humidity",
                        f'{inputs.get("relative_humidity_percent", 0)}%'
                    )

                    c3.metric(
                        "Soil Moisture",
                        f'{inputs.get("soil_moisture_percent", 0)}%'
                    )

                    st.write(
                        f'**Crop:** {inputs.get("crop", "Unknown")}'
                    )

                    st.write(
                        f'**Rainfall:** {inputs.get("recent_rainfall_mm", 0)} mm'
                    )

                    alerts = payload.get("alerts", [])

                    if alerts:

                        st.markdown("**Screening Alerts**")

                        for alert in alerts:

                            message = (
                                f'**{alert.get("level", "Info")}:** '
                                f'{alert.get("title", "")} — '
                                f'{alert.get("detail", "")}'
                            )

                            if alert.get("level") == "High":
                                st.error(message)

                            elif alert.get("level") == "Moderate":
                                st.warning(message)

                            else:
                                st.info(message)

                    else:
                        st.success(
                            "No rule-based alerts were triggered."
                        )

                    with st.expander("View complete farm data"):
                        st.json(payload)

                else:
                    st.json(payload)

    st.markdown("---")

    st.caption(
        "AgriSense AI · Assessment records are retrieved from the "
        "application database. Exported JSON files contain saved "
        "assessment data."
    )

# ==================================================
# REPORTS & HISTORY
# ==================================================

elif page == "Reports & History":

    import json

    st.html("""
    <div style="
        background:linear-gradient(120deg,#102b3a,#142331,#10202b);
        border:1px solid #245064;
        border-radius:18px;
        padding:28px;
        margin-bottom:22px;
    ">
        <div style="color:#19d9f5;font-size:14px;font-weight:700;
                    letter-spacing:2px;">
            AGRISENSE AI · DATA CENTER
        </div>
        <div style="font-size:36px;font-weight:800;color:#ffffff;
                    margin-top:10px;">
            📚 Reports & History
        </div>
        <div style="font-size:18px;color:#b8cbd7;margin-top:10px;
                    line-height:1.7;">
            Review previous plant diagnoses, farm assessments
            and export your agricultural reports.
        </div>
        <div style="font-size:15px;color:#19d9f5;margin-top:18px;">
            SAVED RECORDS · ANALYTICS · REPORT EXPORT
        </div>
    </div>
    """)

    records = get_records()

    # --------------------------------------------------
    # STATISTICS
    # --------------------------------------------------

    plant_records = [
        r for r in records
        if r["record_type"] == "plant_image"
    ]

    farm_records = [
        r for r in records
        if r["record_type"] == "farm_conditions"
    ]

    st.subheader("📊 Assessment Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Assessments", len(records))
    c2.metric("Plant Diagnoses", len(plant_records))
    c3.metric("Farm Assessments", len(farm_records))

    st.markdown("---")

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    st.subheader("📥 Export Reports")

    if records:

        report_data = {
            "application": "AgriSense AI",
            "report_type": "Assessment History",
            "total_assessments": len(records),
            "plant_diagnoses": len(plant_records),
            "farm_assessments": len(farm_records),
            "records": records
        }

        st.download_button(
            "⬇️ Download Complete Assessment Report (JSON)",
            data=json.dumps(
                report_data,
                indent=4,
                default=str
            ),
            file_name="AgriSense_Assessment_Report.json",
            mime="application/json",
            use_container_width=True
        )

    else:
        st.info("Complete and save an assessment to enable report export.")

    st.markdown("---")

    # --------------------------------------------------
    # ASSESSMENT HISTORY
    # --------------------------------------------------

    st.subheader("📁 Saved Assessment Records")

    if not records:

        st.info(
            "No records available yet. Complete and save a Plant Doctor "
            "or Farm Monitoring assessment to see your history here."
        )

    else:

        st.caption(f"{len(records)} saved assessment(s) available.")

        for index, record in enumerate(records):

            with st.expander(
                f'{record["created_at"]} · '
                f'{record["record_type"]} · {record["crop"]}'
            ):

                payload = record["payload"]

                # PLANT DISEASE RECORDS

                if record["record_type"] == "plant_image":

                    st.markdown("### 🌿 Plant Disease Assessment")

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Crop", record["crop"])

                    c2.metric(
                        "Predicted Condition",
                        payload.get("display_label", "Unknown")
                    )

                    c3.metric(
                        "Prediction Score",
                        f'{payload.get("confidence", 0):.2%}'
                    )

                    st.caption(
                        f'Uploaded image: '
                        f'{payload.get("image_name", "Not recorded")}'
                    )

                    latency = payload.get("latency_ms")

                    if latency is not None:
                        st.caption(
                            f"Inference time: {latency:.1f} ms"
                        )

                    with st.expander("View detailed assessment data"):
                        st.json(payload)

                # FARM MONITORING RECORDS

                elif record["record_type"] == "farm_conditions":

                    st.markdown("### 📊 Farm Condition Assessment")

                    inputs = payload.get("inputs", {})

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Temperature",
                        f'{inputs.get("temperature_c", 0)} °C'
                    )

                    c2.metric(
                        "Humidity",
                        f'{inputs.get("relative_humidity_percent", 0)}%'
                    )

                    c3.metric(
                        "Soil Moisture",
                        f'{inputs.get("soil_moisture_percent", 0)}%'
                    )

                    st.write(
                        f'**Crop:** {inputs.get("crop", "Unknown")}'
                    )

                    st.write(
                        f'**Rainfall:** {inputs.get("recent_rainfall_mm", 0)} mm'
                    )

                    alerts = payload.get("alerts", [])

                    if alerts:

                        st.markdown("**Screening Alerts**")

                        for alert in alerts:

                            message = (
                                f'**{alert.get("level", "Info")}:** '
                                f'{alert.get("title", "")} — '
                                f'{alert.get("detail", "")}'
                            )

                            if alert.get("level") == "High":
                                st.error(message)

                            elif alert.get("level") == "Moderate":
                                st.warning(message)

                            else:
                                st.info(message)

                    else:
                        st.success(
                            "No rule-based alerts were triggered."
                        )

                    with st.expander("View complete farm data"):
                        st.json(payload)

                else:
                    st.json(payload)

    st.markdown("---")

    st.caption(
        "AgriSense AI · Assessment records are retrieved from the "
        "application database. Exported JSON files contain saved "
        "assessment data."
    )


        # EXPORT REPORTS
        st.subheader("📤 Export Your Reports")
        st.caption("Download your saved data for future reference.")
        st.divider()

        report = make_markdown_report(records)

        st.download_button(
            "Download history report (.md)",
            report,
            file_name="agrisense_report.md",
            mime="text/markdown",
        )

        st.download_button(
            "Download raw records (.json)",
            json.dumps(
                records,
                indent=2,
                ensure_ascii=False,
                default=str
            ),
            file_name="agrisense_history.json",
            mime="application/json",
        )

        # CLEAR HISTORY

        if st.button(
            "Delete all saved history",
            type="secondary"
        ):

            from src.storage import clear_records

            clear_records()

            st.success("History cleared.")

            st.rerun()


# ==================================================
# EDGE AI LAB
# ==================================================

elif page == "Edge AI Lab":

    
    st.html("""
    <div style="
        background:linear-gradient(120deg,#102b3a,#142331,#10202b);
        border:1px solid #245064;
        border-radius:18px;
        padding:28px;
        margin-bottom:22px;
    ">
        <div style="color:#19d9f5;font-size:14px;font-weight:700;
                    letter-spacing:2px;">
            AGRISENSE AI · TECHNOLOGY CENTER
        </div>
        <div style="font-size:36px;font-weight:800;color:#ffffff;
                    margin-top:10px;">
            ⚙️ Edge AI Lab
        </div>
        <div style="font-size:18px;color:#b8cbd7;margin-top:10px;
                    line-height:1.7;">
            Explore the technology powering AgriSense AI,
            from local plant disease classification to intelligent
            agricultural assistance.
        </div>
        <div style="font-size:15px;color:#19d9f5;margin-top:18px;">
            MOBILENETV2 · TENSORFLOW · GEMINI AI · SQLITE
        </div>
    </div>
    """)

    st.subheader("🧠 Model Performance & Configuration")


    status = model_status()

    c1, c2 = st.columns(2)

    c1.metric(
        "Local Image Model",
        "Ready" if status["ready"] else "Missing"
    )

    c2.metric(
        "Model File Size",
        f'{status["size_mb"]:.2f} MB'
        if status["ready"] else "—"
    )

    st.divider()

    
    st.subheader("🧠 Model Configuration")

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("### ⚡ Model Architecture")
            st.markdown("**MobileNetV2**")
            st.caption("Deep learning image classification")

    with c2:
        with st.container(border=True):
            st.markdown("### 🖼️ Input Dimensions")
            st.markdown("**RGB · 160 × 160 px**")
            st.caption("Standardized image preprocessing")

    c3, c4 = st.columns(2)

    with c3:
        with st.container(border=True):
            st.markdown("### 🌿 Supported Classes")
            st.markdown("**15 Classes**")
            st.caption("Tomato · Potato · Pepper")

    with c4:
        with st.container(border=True):
            st.markdown("### 💻 Inference Engine")
            st.markdown("**Local Device**")
            st.caption("CPU / GPU as supported")

    st.markdown("### 🧠 AI Model Information")

    with st.container(border=True):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Architecture**")
            st.markdown("MobileNetV2")
            st.markdown("**Input Resolution**")
            st.markdown("160 × 160 pixels")
            st.markdown("**Processing**")
            st.markdown("On-device")

        with c2:
            st.markdown("**Classification**")
            st.markdown("15 Classes")
            st.markdown("**Supported Crops**")
            st.markdown("Tomato · Potato · Pepper")
            st.markdown("**Model Status**")
            st.markdown("🟢 Ready" if status["ready"] else "🔴 Missing")

    if status["ready"]:
        st.success("15-class model and labels found.")
    else:
        st.error("Model weights or labels are missing.")

    st.divider()

    # SYSTEM ARCHITECTURE

    st.markdown("---")
    st.subheader("🏗️ System Architecture")
    st.caption("Four integrated components powering the platform.")

    architecture = [
        ("🌿", "Plant Disease Detection", "MobileNetV2 based image classification."),
        ("💬", "Farmer Assistant", "Gemini-powered agricultural guidance."),
        ("📊", "Farm Monitoring", "Rule-based analysis of farm conditions."),
        ("💾", "Data Management", "SQLite storage and report generation.")
    ]

    cols = st.columns(2)

    for i, (emoji, title, description) in enumerate(architecture):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"## {emoji}")
                st.markdown(f"**{title}**")
                st.caption(description)

    st.divider()

    # MODEL EVALUATION

    st.markdown("---")
    st.subheader("📈 Model Evaluation")
    st.caption("Validation performance of the plant disease classifier.")

    st.metric(
        "Validation Accuracy",
        "92.17%"
    )

    st.caption(
        "Measured on the held-out PlantVillage validation split. "
        "This is not a guarantee of real-world field accuracy."
    )

    st.write(
        "**Important:** Model confidence is not the same as "
        "diagnostic accuracy. Real-world images may produce "
        "incorrect predictions."
    )

    st.divider()

    # LAST INFERENCE

    st.subheader("Last Measured Inference")

    latest = st.session_state.latest_assessment

    if (
        latest.get("type") == "plant_image"
        and latest.get("result", {}).get("latency_ms") is not None
    ):

        st.metric(
            "Inference Latency",
            f'{latest["result"]["latency_ms"]:.1f} ms'
        )

    else:

        st.caption(
            "No image inference has been recorded in this session."
        )
