# -*- coding: utf-8 -*-
import shap
import py3Dmol
import streamlit as st
import base64
import pandas as pd
import numpy as np


# Patch for SHAP compatibility with new NumPy (>=1.24)
if not hasattr(np, "int"):
    np.int = int

import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors, QED, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import pickle
import warnings
import streamlit.components.v1 as components
from functools import lru_cache
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="ToxicSmiles Advanced - ML-Powered Molecular Toxicity Prediction",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Streamlined CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
    .main-content{ background: white; border-radius: 20px; padding: 2rem; margin: 1rem; box-shadow: 0 20px 40px rgba(0,0,0,0.1); backdrop-filter: blur(10px);}
    .stApp { font-family: 'Inter', sans-serif; }
    h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3.5rem !important; font-weight: 700 !important; text-align: center; margin-bottom: 0.5rem !important;}
    .subtitle { text-align: center; color: #6b7280; font-size: 1.2rem; margin-bottom: 2rem; font-weight: 400;}
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #f8fafc; border-radius: 12px; padding: 4px; margin-bottom: 2rem;}
    .stTabs [data-baseweb="tab"] { height: 50px; background: transparent; border-radius: 8px; color: #64748b; font-weight: 500; transition: all 0.3s ease; padding: 0 20px;}
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);}
    .input-section { background: white; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 2px solid #e2e8f0; transition: all 0.3s ease;}
    .input-section:hover { border-color: #667eea; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);}
    .stTextInput > div > div > input { border-radius: 8px; border: 2px solid #e2e8f0; padding: 12px 16px; font-size: 16px; transition: all 0.3s ease;}
    .stTextInput > div > div > input:focus { border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);}
    .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; padding: 12px 32px; font-weight: 600; font-size: 16px; transition: all 0.3s ease; width: 100%;}
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);}
    .ensemble-prediction { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white !important; padding: 2rem; border-radius: 16px; text-align: center; margin: 1rem 0 2rem 0; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);}
    .ensemble-prediction h2 { color: white !important; margin-bottom: 1rem; background: none !important; -webkit-text-fill-color: white !important;}
    .ensemble-prediction h1 { color: white !important; font-size: 2.5rem; margin: 0.5rem 0; background: none !important; -webkit-text-fill-color: white !important;}
    .ensemble-prediction p { font-size: 1.2rem; margin: 0; color: white !important;}
    .metric-card { background: white; border-radius: 12px; padding: 1.5rem; text-align: center; border: 2px solid #e2e8f0; transition: all 0.3s ease; margin: 0.5rem 0;}
    .metric-card:hover { transform: translateY(-4px); box-shadow: 0 12px 24px rgba(0,0,0,0.1); border-color: #667eea;}
    .metric-number { font-size: 2rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;}
    .metric-label { color: #64748b; font-weight: 500;}
    .uncertainty-box { background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border: 2px solid #f59e0b; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; text-align: center;}
</style>
""", unsafe_allow_html=True)

# --- Markdown Content ---
why_this_matters_md = """
## Why This Matters

### The Translation Problem
Drug development has a prediction problem. 90% of drugs that pass animal safety tests fail in human trials, costing an average of $1.2 billion per successful drug and 10-15 years of development time.

The core issue: biological differences between species create translational gaps. For instance, mouse metabolism follows different pathways than human liver function. This creates a system where we test on animals who cannot consent to develop drugs for those who can, and ironically often failing both.

### A Computational Solution
Modern infrastructure enables new approaches:
- Molecular modeling for detailed structure-activity analysis
- Human datasets with verified effects of hundreds of thousands of compounds
- Machine learning to identify toxicity patterns in human-relevant data
- Uncertainty quantification to assess prediction reliability

ToxicSmiles Advanced leverages existing human data rather than generating new animal experiments, developing more direct methods for predicting human toxicity responses with confidence assessment.

### Beyond Single Endpoints
Our roadmap aims to expand on this foundation:
- Multi-endpoint models for hepatotoxicity, cardiotoxicity, neurotoxicity, and genotoxicity
- Molecular-to-omics integration linking chemical structure to transcriptional signatures
- Endocrine disruption assessment for hormonal pathway interference
- Explainable AI for mechanistic understanding of toxicity predictions

### The Vision
Query a molecular structure. Receive comprehensive toxicity assessment with uncertainty bounds and confidence-weighted predictions for informed decision-making.
"""

how_it_works_md = """
## ToxicSmiles Advanced: Next-Generation Digital Toxicology

ToxicSmiles Advanced is a comprehensive computational platform that predicts chemical effects on mitochondrial membrane potential (MMP) using ensemble machine learning with advanced uncertainty quantification.

### For Everyone: The Enhanced Experience
Input a molecule's SMILES code. Get toxicity prediction with confidence intervals and uncertainty assessment.

**Process:**
1. Submit molecular SMILES string
2. Platform extracts 4000+ structural features
3. Ensemble of 5+ ML models generates predictions with uncertainty
4. Receive toxicity score with confidence bounds

### For Toxicologists: Advanced Analytics

**Feature Engineering (4000+ Features)**
- Morgan fingerprints (ECFP4, 2048 bits)
- MACCS keys (167 structural alerts)
- Atom pair fingerprints (1024 bits)
- RDKit topological fingerprints
- Physicochemical descriptors (MW, LogP, TPSA, QED, etc.)
- ADMET properties and drug-likeness metrics

**Advanced Preprocessing**
- Scaffold-aware train/test splitting to prevent data leakage
- StandardScaler normalization with feature selection
- SMOTE for intelligent class balancing

**Enhanced Ensemble Architecture**
Five algorithms with uncertainty quantification:
- Random Forest with balanced classes
- XGBoost with optimized hyperparameters  
- LightGBM for gradient boosting
- Support Vector Machine with RBF kernel
- Multi-head Attention Network for chemical feature relationships

**Uncertainty Quantification**
- Model disagreement-based uncertainty quantification
- Confidence intervals around predictions

### For ML Professionals: Technical Implementation

**Advanced Pipeline Architecture:**
- Input: SMILES strings with validation
- Featurization: 4000+ molecular descriptors via RDKit
- Preprocessing: Scaffold-aware splitting, standardization, feature selection
- Models: Voting ensemble + attention network with uncertainty quantification
- Output: Calibrated probabilities with confidence bounds

**Key Innovations:**
- Scaffold-aware data splitting prevents overfitting to chemical scaffolds
- Uncertainty quantification via ensemble disagreement and confidence intervals
- Multi-head attention for learning complex feature interactions

The system is optimized for mitochondrial toxicity prediction with comprehensive uncertainty assessment.
"""

about_us_md = """
## About Us

I'm Chaitanya, and honestly, my career path has been anything but predictable. I started in animal rights advocacy, fell in love with cellular agriculture (basically trying to make meat without killing animals), then somehow ended up in environmental chemistry, and now I'm deep in toxicology and AI.

The thing that ties it all together? A deep care for animals and the environment, plus never stopping asking 'but why?' like a 5-year-old. It made me wonder why we're still testing on animals in labs when AI can already beat grandmasters at chess and drive cars. Seemed like a problem worth fixing.

When I'm not drowning in datasets, debugging code at 2AM, I'm probably on some hiking trail, going down philosophy rabbit holes, or watching horror movies to decompress. I've realized my best ideas don't come when I'm trying to force them - they show up when I'm completely disconnected from work, usually when riding my motorcycle into the horizon.

The goal isn't just better predictions - it's building AI systems that toxicologists can trust, understand, and use to make better decisions for human and environmental health.

If you've got ideas about where computational toxicology should go next, or want to collaborate on making this technology more accessible and impactful, I'm all ears. The conversations I love most start with "what if we could..." [Connect with me](https://www.linkedin.com/in/chaitanya-c-b67a73178/) and let's see where it leads us!
"""

# --- Toxicity Predictor Class ---
class AdvancedToxicityPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k='all')
        self.models = {}
        self.ensemble_model = None
        self.attention_model = None
        self.uncertainty_models = {}
        self.best_params = {}
        self.feature_names = None
        self.is_trained = False
        self.X_train_final = None
        self.y_train_final = None

    def get_advanced_toxicity_descriptors(self, mol):
        descriptors = {}
        try:
            descriptors.update({
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'MolMR': Descriptors.MolMR(mol),
                'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
                'RingCount': Descriptors.RingCount(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'LabuteASA': Descriptors.LabuteASA(mol),
                'BalabanJ': Descriptors.BalabanJ(mol),
                'MaxEStateIndex': Descriptors.MaxEStateIndex(mol),
                'MinEStateIndex': Descriptors.MinEStateIndex(mol),
            })
            descriptors.update({
                'QED': QED.qed(mol),
                'Lipinski_violations': sum([
                    Descriptors.MolWt(mol) > 500,
                    Descriptors.MolLogP(mol) > 5,
                    Descriptors.NumHDonors(mol) > 5,
                    Descriptors.NumHAcceptors(mol) > 10
                ]),
                'Veber_violations': sum([
                    Descriptors.NumRotatableBonds(mol) > 10,
                    Descriptors.TPSA(mol) > 140
                ]),
            })
            toxicity_fragments = [
                'fr_Al_COO', 'fr_Al_OH', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO',
                'fr_C_O', 'fr_C_S', 'fr_NH2', 'fr_N_O', 'fr_SH', 'fr_aldehyde',
                'fr_benzene', 'fr_halogen', 'fr_ketone', 'fr_nitro', 'fr_phenol',
                'fr_pyridine', 'fr_sulfide', 'fr_thiophene', 'fr_urea', 'fr_ester',
                'fr_ether', 'fr_Imine', 'fr_quatN', 'fr_NH1', 'fr_NH0'
            ]
            for fragment in toxicity_fragments:
                if hasattr(Descriptors, fragment):
                    descriptors[fragment] = getattr(Descriptors, fragment)(mol)
            for key, value in descriptors.items():
                if np.isnan(value) or np.isinf(value):
                    descriptors[key] = 0.0
        except Exception as e:
            st.error(f"Error calculating descriptors: {e}")
        return descriptors

    @lru_cache(maxsize=10000)
    def get_molecular_fingerprints_cached(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        fingerprints = {}
        try:
            morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
            morgan_fp = morgan_gen.GetFingerprint(mol)
            fingerprints.update({f'Morgan_{i}': int(morgan_fp[i]) for i in range(2048)})
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            fingerprints.update({f'MACCS_{i}': int(maccs_fp[i]) for i in range(len(maccs_fp))})
            atom_pair_gen = GetAtomPairGenerator(maxDistance=20, fpSize=1024)
            atom_pair_fp = atom_pair_gen.GetFingerprint(mol)
            fingerprints.update({f'AtomPair_{i}': int(atom_pair_fp[i]) for i in range(1024)})
        except Exception as e:
            st.error(f"Error generating fingerprints: {e}")
        return fingerprints

    def predict_with_uncertainty(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {'error': 'Invalid SMILES string'}
            descriptors = self.get_advanced_toxicity_descriptors(mol)
            fingerprints = self.get_molecular_fingerprints_cached(smiles)
            features = {}
            features.update(descriptors)
            features.update(fingerprints)
            if not self.feature_names:
                return {'error': 'Model not properly loaded - missing feature names'}
            feature_vector = np.array([features.get(name, 0) for name in self.feature_names])
            feature_vector = feature_vector.reshape(1, -1)
            feature_vector_scaled = self.scaler.transform(feature_vector)
            feature_vector_selected = self.feature_selector.transform(feature_vector_scaled)
            predictions = {}
            probabilities = []
            for name, model in self.models.items():
                if name == 'attention':
                    if hasattr(model, 'predict'):
                        pred_proba = model.predict(feature_vector_selected)[0][0]
                        pred = 1 if pred_proba > 0.5 else 0
                    else:
                        continue
                else:
                    pred = model.predict(feature_vector_selected)[0]
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(feature_vector_selected)[0][1]
                    else:
                        pred_proba = 0.5
                predictions[name] = {'prediction': int(pred), 'probability': pred_proba}
                probabilities.append(pred_proba)
            if self.ensemble_model:
                ensemble_pred = int(self.ensemble_model.predict(feature_vector_selected)[0])
                ensemble_proba = self.ensemble_model.predict_proba(feature_vector_selected)[0][1]
            else:
                ensemble_pred = int(np.mean([p['prediction'] for p in predictions.values()]) > 0.5)
                ensemble_proba = np.mean(probabilities)
            uncertainty = np.std(probabilities) if len(probabilities) > 1 else 0.0
            ci_lower = max(0, ensemble_proba - 2*uncertainty)
            ci_upper = min(1, ensemble_proba + 2*uncertainty)
            return {
                'ensemble_prediction': ensemble_pred,
                'ensemble_probability': ensemble_proba,
                'individual_predictions': predictions,
                'uncertainty': uncertainty,
                'confidence_interval': (ci_lower, ci_upper)
            }
        except Exception as e:
            return {'error': f"Prediction error: {str(e)}"}

@st.cache_data
def load_model():
    try:
        import pickle5 as pickle
        model_pkl_path = "dev_advanced_toxicity_model.pkl"
        with open(model_pkl_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except ImportError:
        try:
            import pickle
            model_pkl_path = "dev_advanced_toxicity_model.pkl"
            with open(model_pkl_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

def enhanced_interactive_prediction(user_input):
    model_data = load_model()
    if model_data is None:
        return None
    predictor = AdvancedToxicityPredictor()
    predictor.models = model_data.get('models', {})
    predictor.ensemble_model = model_data.get('ensemble_model')
    predictor.scaler = model_data.get('scaler', StandardScaler())
    predictor.feature_selector = model_data.get('feature_selector', SelectKBest())
    predictor.feature_names = model_data.get('feature_names', [])
    predictor.is_trained = True
    try:
        from tensorflow.keras.models import load_model as tf_load_model
        attention_path = "advanced_toxicity_model_attention.h5"
        if 'attention.h5' in str(attention_path):
            predictor.attention_model = tf_load_model(attention_path, compile=False)
            predictor.models['attention'] = predictor.attention_model
    except:
        pass
    result = predictor.predict_with_uncertainty(user_input)
    return result

def get_shap_top_features(user_input, model_data):
    """
    Compute SHAP values and return Top-10 *pro-toxic* features.
    Uses XGBoost booster when available to avoid SHAP decode errors.
    Falls back to LightGBM or RandomForest if needed.
    """
    predictor = AdvancedToxicityPredictor()
    predictor.models = model_data.get('models', {})
    predictor.ensemble_model = model_data.get('ensemble_model')
    predictor.scaler = model_data.get('scaler', StandardScaler())
    predictor.feature_selector = model_data.get('feature_selector', SelectKBest(f_classif))
    predictor.feature_names = model_data.get('feature_names', [])

    # Choose a tree model for SHAP (prefer XGB → LGB → RF)
    model = (
        predictor.models.get('xgb')
        or predictor.models.get('lgb')
        or predictor.models.get('rf')
    )
    if model is None:
        return {'error': 'No tree model found (xgb/lgb/rf) for SHAP analysis.'}

    # Featurize input
    mol = Chem.MolFromSmiles(user_input)
    if mol is None:
        return {'error': 'Invalid SMILES string'}

    descriptors = predictor.get_advanced_toxicity_descriptors(mol)
    fingerprints = predictor.get_molecular_fingerprints_cached(user_input)
    features = {**descriptors, **fingerprints}

    # Build vector in training feature order → scale → select
    x = np.array([features.get(name, 0) for name in predictor.feature_names]).reshape(1, -1)
    x_scaled = predictor.scaler.transform(x)
    x_sel = predictor.feature_selector.transform(x_scaled)
    mask = predictor.feature_selector.get_support()
    selected_features = [name for i, name in enumerate(predictor.feature_names) if mask[i]]

    # Build a TreeExplainer robustly
    explainer = None
    shap_values = None
    try:
        # If it's an XGBoost sklearn wrapper, prefer its Booster
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
            explainer = shap.TreeExplainer(booster)
            # Newer SHAP versions prefer the callable API; try that first
            try:
                shap_values = explainer(x_sel).values
            except Exception:
                shap_values = explainer.shap_values(x_sel)
        else:
            # LightGBM / RF (sklearn tree ensembles)
            explainer = shap.TreeExplainer(model)
            try:
                shap_values = explainer(x_sel).values
            except Exception:
                shap_values = explainer.shap_values(x_sel)
    except Exception as e:
        # Last-resort fallback (slower but safe): model-agnostic Explainer
        try:
            explainer = shap.Explainer(model.predict_proba, masker=x_sel)
            shap_values = explainer(x_sel).values
            # For binary classifiers, take the contribution to class 1 if needed
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]
        except Exception as e2:
            return {'error': f'SHAP failed: {e2}'}

    # shap_values shape handling (binary: (1, n_features) or (n_classes, ...))
    if isinstance(shap_values, list):
        # Some SHAP versions return list per class; pick toxic class (1)
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_row = np.array(shap_values[0]) if shap_values is not None else None
    if shap_row is None or shap_row.shape[0] != len(selected_features):
        return {'error': 'SHAP output shape mismatch.'}

    # Keep only positive (pro-toxic) contributions, sort desc, top 10
    idx_desc = np.argsort(shap_row)[::-1]
    top = []
    for i in idx_desc:
        val = float(shap_row[i])
        if val > 0:
            top.append({'feature': selected_features[i], 'shap_value': val})
        if len(top) >= 10:
            break

    return {'top10': top}



def display_streamlined_results(result):
    if 'error' in result:
        st.error(f"Error: {result['error']}")
        return
    ensemble_pred = result.get('ensemble_prediction')
    ensemble_prob = result.get('ensemble_probability', 0)
    uncertainty = result.get('uncertainty', 0)
    pred_text = "TOXIC" if ensemble_pred == 1 else "NON-TOXIC"
    confidence_percent = ensemble_prob * 100 if ensemble_pred == 1 else (1 - ensemble_prob) * 100
    st.markdown(f"""
    <div class="ensemble-prediction">
        <h2>Ensemble Prediction</h2>
        <h1>{pred_text}</h1>
        <p>Confidence: {confidence_percent:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="uncertainty-box">
        <h3>Uncertainty Analysis</h3>
        <div class="metric-number">{uncertainty*100:.1f}%</div>
        <div class="metric-label">Model Disagreement</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Individual Model Predictions")
    individual_preds = result.get('individual_predictions', {})
    if individual_preds:
        cols = st.columns(len(individual_preds))
        for i, (name, pred_info) in enumerate(individual_preds.items()):
            with cols[i]:
                prediction = pred_info.get('prediction')
                prob = pred_info.get('probability', 0)
                pred_text = "Toxic" if prediction == 1 else "Non-toxic"
                confidence = prob * 100 if prediction == 1 else (1 - prob) * 100
                bg_color = "#fee2e2" if prediction == 1 else "#dcfce7"
                border_color = "#ef4444" if prediction == 1 else "#22c55e"
                st.markdown(f"""
                <div class="metric-card" style="background: {bg_color}; border-color: {border_color};">
                    <h4>{name.upper()}</h4>
                    <div class="metric-number">{pred_text}</div>
                    <div class="metric-label">{confidence:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

def render_3d_molecule(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Cannot generate 3D structure from SMILES")
            return
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=400, height=350)
        viewer.addModel(block, "mol")
        viewer.setStyle({'stick': {'radius': 0.2}, 'sphere': {'radius': 0.4}})
        viewer.setBackgroundColor("white")
        viewer.zoomTo()
        viewer.spin(True)
        components.html(viewer._make_html(), height=350)
    except Exception as e:
        st.error(f"Could not generate 3D structure: {str(e)}")

def main():
    st.markdown("""
    <div class="hero-section">
        <h1>ToxicSmiles</h1>
        <div class="subtitle">Next-Generation ML-Powered Molecular Toxicity Prediction</div>
    </div>
    """, unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs([
        "Prediction", 
        "Why ToxicSmiles?", 
        "How It Works", 
        "About Us"
    ])
    with tab1:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Enter SMILES String")
            st.markdown("Input your molecular SMILES notation for toxicity analysis")
            st.markdown("**Try these examples:**")
            example_cols = st.columns(3)
            examples = [
                ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
                ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
                ("CCCP", "C1=CC(=CC=C1NN=C(C#N)C#N)OC(F)(F)F"),
            ]
            for i, (name, smiles) in enumerate(examples):
                with example_cols[i]:
                    if st.button(name):
                        st.session_state.smiles_input = smiles
            user_input = st.text_input(
                "SMILES Input:",
                value=st.session_state.get('smiles_input', ''),
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                help="Enter a valid SMILES string representing your molecule"
            )
            predict_btn = st.button("Predict Toxicity", type="primary")
        with col2:
            st.markdown("### Molecular Visualization")
            if user_input:
                render_3d_molecule(user_input)
            else:
                st.info("Enter a SMILES string to view 3D structure")
        if predict_btn and user_input:
            with st.spinner("Running toxicity analysis..."):
                result = enhanced_interactive_prediction(user_input)
            if result:
                display_streamlined_results(result)
                st.markdown("### 🔍 Top 10 Pro-Toxic Features (SHAP)")
                model_data = load_model()
                shap_result = get_shap_top_features(user_input, model_data)
                if shap_result and 'top10' in shap_result:
                    top10 = shap_result['top10']
                    if len(top10) == 0:
                        st.info("No pro-toxic features detected for this molecule (all SHAP values ≤ 0).")
                    else:
                        feat_names = [row['feature'] for row in top10][::-1]
                        shap_vals  = [row['shap_value'] for row in top10][::-1]
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=shap_vals,
                            y=feat_names,
                            orientation='h',
                            text=[f"{v:.3f}" for v in shap_vals],
                            textposition='auto'
                        ))
                        fig.update_layout(
                            title="Top 10 Features Driving Toxicity",
                            xaxis_title="SHAP value (higher = stronger toxic contribution)",
                            yaxis_title="Feature",
                            height=420,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown("#### Details")
                        df_top = pd.DataFrame(top10)
                        df_top['shap_value'] = df_top['shap_value'].map(lambda v: f"{v:.4f}")
                        st.dataframe(df_top, use_container_width=True)
                elif shap_result and 'error' in shap_result:
                    st.warning(f"SHAP explanation unavailable: {shap_result['error']}")
                st.markdown("---")
                st.markdown("### Understanding Your Results")
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.markdown("""
                    #### **Ensemble Prediction**
                    This is your most reliable result - it combines insights from multiple ML approaches including:
                    - Random Forest for feature importance
                    - XGBoost for gradient boosting
                    - SVM for non-linear patterns
                    - LightGBM for efficient learning
                    - Neural attention networks for complex relationships
                    """)
                with info_cols[1]:
                    st.markdown("""
                    #### **Uncertainty Analysis**
                    - **Uncertainty**: How much models disagree (lower = more confident)
                    - **Individual Models**: Breakdown of each algorithm's prediction
                    - **Confidence**: Overall certainty in the prediction
                    """)
                if result.get('uncertainty', 0) > 0.15:
                    st.info("""
                    **Moderate Uncertainty**: Models show some disagreement. The prediction is reasonable 
                    but consider the confidence levels when making decisions.
                    """)
                else:
                    st.success("""
                    **High Confidence**: Models are in strong agreement. This prediction is highly reliable.
                    """)
            else:
                st.error("Failed to generate prediction. Please check your SMILES string and try again.")
        st.markdown('</div>', unsafe_allow_html=True)
    with tab2:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown(why_this_matters_md, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab3:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown(how_it_works_md, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with tab4:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown(about_us_md, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
