import streamlit as st
import base64
import pandas as pd

import lightgbm as lgb # Added LightGBM import
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetAtomPairGenerator
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
import numpy as np
import pickle
import warnings
import streamlit.components.v1 as components
warnings.filterwarnings("ignore")

# Configure page
st.set_page_config(
    page_title="ToxicSmiles - Machine Learning to Predict Molecular Toxicity",
    page_icon="☣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
   
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
   
    .main-content {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
   
    .stApp {
        font-family: 'Inter', sans-serif;
    }
   
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
   
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        border-radius: 12px;
        padding: 4px;
        margin-bottom: 2rem;
    }
   
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 0 20px;
    }
   
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
   
    .prediction-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
   
    .input-section {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
   
    .input-section:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
   
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 12px 16px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
   
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
   
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
    }
   
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
   
    .section-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        color: #1e293b;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
   
    .sub-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        color: #334155;
    }
   
    .content-section {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
   
    .highlight-box {
        background: linear-gradient(135deg, #667eea10 0%, #764ba220 100%);
        border: 1px solid #667eea30;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
   
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
   
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
    }
   
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
   
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
   
    .stat-label {
        color: #64748b;
        font-weight: 500;
    }
   
    .feature-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 0.5rem;
        margin: 1rem 0;
    }
   
    .feature-item {
        background: #f8fafc;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 3px solid #667eea;
        transition: all 0.3s ease;
    }
   
    .feature-item:hover {
        background: #e2e8f0;
        transform: translateX(4px);
    }
   
    .quote {
        font-style: italic;
        color: #475569;
        font-size: 1.1rem;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 1.5rem 0;
        background: #f8fafc;
        padding: 1rem 1rem 1rem 2rem;
        border-radius: 0 8px 8px 0;
    }
   
    .ref {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
   
    .hero-section {
        text-align: center;
        background: linear-gradient(135deg, #667eea10 0%, #764ba220 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
   
    .hero-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }

    .ensemble-prediction {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    .ensemble-prediction h2 {
        color: white !important;
        margin-bottom: 1rem;
        background: none !important;
        -webkit-text-fill-color: white !important;
    }
    
    .ensemble-prediction h1 {
        color: white !important;
        font-size: 2.5rem;
        margin: 0.5rem 0;
        background: none !important;
        -webkit-text-fill-color: white !important;
    }
    
    .ensemble-prediction p {
        font-size: 1.2rem;
        margin: 0;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Content definitions with enhanced styling
why_this_matters_md = """
## The Ethical Paradox We've Normalized

We've built our entire pharma industry on a moral contradiction: inflicting deliberate harm on sentient beings who cannot consent to benefit those who can. Every year, millions of laboratory animals undergo toxicity testing; creatures capable of fear, pain, and suffering subjected to distress not for their benefit, but for ours.

**However, it's not just ethically problematic—it's scientifically flawed as well.**

## The Scientific Inadequacy

We test on animals to predict human responses, yet our predictions are systematically unreliable. Species differences aren't minor mutations; they're significant biological variations. A mouse liver metabolizes compounds through divergent enzymatic pathways than a human liver. Rat cardiovascular systems operate under different physiological constraints than human hearts.

**Here's the reality:**
- **90%** of drugs that pass animal safety tests fail in human trials
- **$1.2B** average cost per successful drug, with substantial waste on false leads
- **10-15 years** of development time regularly derailed by species-specific predictions

We're failing both animals and patients who wait while we pursue leads that animal biology suggested were promising but human biology rejects.

## The False Necessity

The strongest defense rests on necessity: that animal testing represents our best available method. This deserves serious challenge.

The infrastructure is already here:
- Sophisticated computational systems can precisely analyze molecular structures
- Vast databases contain hundreds of thousands of compounds with verified human effects
- Advanced machine learning algorithms are increasingly adept at identifying complex toxicity patterns

**The question isn't whether these alternatives could be perfect—neither are animal models. The question is whether they provide equal or better predictive value while eliminating significant ethical costs.**

## The Path Forward

**ToxicSmiles** aims to leverage existing data intelligently instead of harming sentient beings to generate new data. We're trying to replace ineffective methods with more effective ones while resolving an ethical problem that persisted only because we assumed it was scientifically necessary.

We advance as a species when our methods align with our values, not when we sacrifice one for the other.

---

*References:*
- Akhtar A. The flaws and human harms of animal experimentation. *Camb Q Healthc Ethics.* 2015;24(4):407–19.
- Wouters OJ, McKee M, Luyten J. R&D Investment for New Medicine. *JAMA.* 2020;323(9):844–853.
- Paul SM et al. R&D productivity challenge. *Nature Reviews Drug Discovery.* 2010;9(3):203–214.
"""

how_it_works_md = """
## ToxicSmiles: Your Digital Toxicology Lab

**ToxicSmiles** is an advanced computational platform designed specifically to predict chemical effects on mitochondrial membrane potential (MMP)—a critical biomarker of cellular health and toxicity—using state-of-the-art machine learning models trained on human-relevant data.

## For Everyone: The Big Picture

Imagine you discover a new chemical and want to know if it disrupts mitochondrial membrane potential, a key driver of cell viability and toxicity. Instead of animal testing, **ToxicSmiles** operates as a "digital lab."

**How it works:** Provide the molecule's SMILES code (its digital blueprint), and the platform analyzes its features against a vast library of chemicals known to affect MMP in humans. ToxicSmiles then delivers:

- A scientific toxicity prediction
- A confidence score  
- An explanation of key reasons behind the result

**No animal testing required.**

## For Toxicologists: The Technical Details

### Feature Extraction
SMILES are translated into detailed profiles using hundreds of molecular descriptors and chemical fingerprints that capture:
- Molecular weight
- Precise structural topologies
- Functional groups known to influence mitochondrial activity

### Data Preprocessing
Features are standardized and filtered. Class imbalance in mitochondrial toxicity datasets is addressed using **SMOTE**, enhancing true toxicant detection.

### Ensemble Modeling
ToxicSmiles uses a powerful ensemble of machine learning models:
- Random Forest
- XGBoost
- LightGBM
- Support Vector Machine (SVM)
- Deep Neural Network

### Consensus Prediction
Individual model outputs are combined via *soft voting* to produce a consensus MMP disruption score.

### Interpretability
ToxicSmiles explains which molecular features most strongly drive predicted mitochondrial toxicity, helping researchers identify potential mitochondrial toxicophores and design safer molecules.

## For Machine Learning Professionals

**Technical Architecture:**
- **Input & Featurization:** Accepts SMILES strings. Uses RDKit to generate descriptors and fingerprints enriched for mitochondrial toxicity
- **Preprocessing:** Applies normalization, feature selection, and SMOTE for handling class imbalance
- **Ensemble Design:** A VotingClassifier with soft voting combines five models
- **Prediction Aggregation:** Probabilities are averaged for calibrated MMP toxicity scores
- **Interpretability Module:** Uses feature importance, regression coefficients, and statistical metrics
- **Deployment:** Supports both interactive use and batch processing. Models are stored via pickle and Keras

Supports both single-compound queries and large-scale screening to accelerate toxicology research.
"""


about_us_md = """
## About Us

I'm Chaitanya!, and my path here has been anything but conventional. My journey spans through animal rights, cellular agriculture (working on meat without murder) then moved into environmental chemistry, and now I'm in toxicology. 

What connects all these dots? A simple philosophy: the best solutions come from questioning everything like a 5-year-old would. That relentless curiosity led me to ask why, if AI can beat grandmasters at chess and drive cars, we're still using animal testing in toxicology labs. It seemed like a problem worth solving, so ToxicSmiles became my proof of concept—using machine learning to make toxicology less toxic to, well, everything.

When I'm not knee-deep in data or debugging experiments or stubborn code, you'll find me wandering hiking trails, falling into 2 AM philosophy rabbit holes, or calming my nerves with horror movies. Over the years, I've learned that the most interesting discoveries show up in the most unexpected places, whether that's connecting seemingly unrelated fields or having breakthrough moments while completely unplugged from work.

If you've got a bold idea or want to tackle something that actually matters, there's nothing I love more than conversations that start with "what if we could..." [Connect with me](https://www.linkedin.com/in/chaitanya-c-b67a73178/) and let's see where the questions take us.
"""


# Rest of your existing classes and functions remain the same
class EnhancedToxicityPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k='all')
        self.models = {}
        self.ensemble_model = None
        self.best_params = {}
        self.feature_names = None
        self.is_trained = False
        self.nn_model = None

        self.pre_optimized_params = {
            'rf_n_estimators': 236,
            'rf_max_depth': 23,
            'rf_min_samples_split': 18,
            'rf_min_samples_leaf': 8,
            'xgb_n_estimators': 140,
            'xgb_max_depth': 9,
            'xgb_learning_rate': 0.2507830493666598,
            'xgb_subsample': 0.6094213865403939,
            'xgb_colsample_bytree': 0.7071271107568533,
            'svm_C': 59.03505324844556,
            'svm_gamma': 0.593644676772998,
            'svm_kernel': 'rbf',
            'lgb_n_estimators': 100,
            'lgb_num_leaves': 31,
            'lgb_learning_rate': 0.1
        }
    def get_molecular_fingerprints(self, mol):
        """Get various molecular fingerprints using updated RDKit generators"""
        fingerprints = {}

        try:
            # Morgan fingerprints (ECFP4)
            morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
            morgan_fp = morgan_gen.GetFingerprint(mol)
            fingerprints.update({f'Morgan_{i}': int(morgan_fp[i]) for i in range(1024)})

            # MACCS keys
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            fingerprints.update({f'MACCS_{i}': int(maccs_fp[i]) for i in range(len(maccs_fp))})

            # Atom pairs
            atom_pair_gen = GetAtomPairGenerator(maxDistance=30, fpSize=1024)
            atom_pair_fp = atom_pair_gen.GetFingerprint(mol)
            fingerprints.update({f'AtomPair_{i}': int(atom_pair_fp[i]) for i in range(1024)})

            # Topological fingerprints (RDKit)
            rdkit_fp = FingerprintMols.FingerprintMol(mol)
            rdkit_bits = rdkit_fp.GetOnBits()
            for i, bit in enumerate(rdkit_bits):
                if i < 512:
                    fingerprints[f'RDKit_{bit}'] = 1

        except Exception as e:
            print(f"Error generating fingerprints: {e}")

        return fingerprints

    def get_toxicity_descriptors(self, mol):
        """Get descriptors specifically relevant to toxicity prediction"""
        descriptors = {}

        try:
            # Basic physicochemical properties
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
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
                'LabuteASA': Descriptors.LabuteASA(mol),
                'BalabanJ': Descriptors.BalabanJ(mol),
                'Chi0': Descriptors.Chi0(mol),
                'Chi1': Descriptors.Chi1(mol),
                'Chi0v': Descriptors.Chi0v(mol),
                'Chi1v': Descriptors.Chi1v(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'Kappa2': Descriptors.Kappa2(mol),
                'Kappa3': Descriptors.Kappa3(mol),
                'MaxEStateIndex': Descriptors.MaxEStateIndex(mol),
                'MinEStateIndex': Descriptors.MinEStateIndex(mol),
                'SlogP_VSA2': Descriptors.SlogP_VSA2(mol),
                'SMR_VSA3': Descriptors.SMR_VSA3(mol),
                'PEOE_VSA6': Descriptors.PEOE_VSA6(mol),
                'EState_VSA2': Descriptors.EState_VSA2(mol),
            })

            # Toxicity-specific descriptors
            descriptors.update({
                'NumRadicalElectrons': Descriptors.NumRadicalElectrons(mol),
                'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
                'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
                'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
                'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles(mol),
                'fr_Al_COO': Descriptors.fr_Al_COO(mol),
                'fr_Al_OH': Descriptors.fr_Al_OH(mol),
                'fr_Ar_N': Descriptors.fr_Ar_N(mol),
                'fr_Ar_NH': Descriptors.fr_Ar_NH(mol),
                'fr_Ar_OH': Descriptors.fr_Ar_OH(mol),
                'fr_COO': Descriptors.fr_COO(mol),
                'fr_COO2': Descriptors.fr_COO2(mol),
                'fr_C_O': Descriptors.fr_C_O(mol),
                'fr_C_S': Descriptors.fr_C_S(mol),
                'fr_HOCCN': Descriptors.fr_HOCCN(mol),
                'fr_Imine': Descriptors.fr_Imine(mol),
                'fr_NH0': Descriptors.fr_NH0(mol),
                'fr_NH1': Descriptors.fr_NH1(mol),
                'fr_NH2': Descriptors.fr_NH2(mol),
                'fr_N_O': Descriptors.fr_N_O(mol),
                'fr_Ndealkylation1': Descriptors.fr_Ndealkylation1(mol),
                'fr_Ndealkylation2': Descriptors.fr_Ndealkylation2(mol),
                'fr_Nhpyrrole': Descriptors.fr_Nhpyrrole(mol),
                'fr_SH': Descriptors.fr_SH(mol),
                'fr_aldehyde': Descriptors.fr_aldehyde(mol),
                'fr_benzene': Descriptors.fr_benzene(mol),
                'fr_furan': Descriptors.fr_furan(mol),
                'fr_halogen': Descriptors.fr_halogen(mol),
                'fr_ketone': Descriptors.fr_ketone(mol),
                'fr_lactam': Descriptors.fr_lactam(mol),
                'fr_lactone': Descriptors.fr_lactone(mol),
                'fr_methoxy': Descriptors.fr_methoxy(mol),
                'fr_nitro': Descriptors.fr_nitro(mol),
                'fr_oxazole': Descriptors.fr_oxazole(mol),
                'fr_phenol': Descriptors.fr_phenol(mol),
                'fr_pyridine': Descriptors.fr_pyridine(mol),
                'fr_quatN': Descriptors.fr_quatN(mol),
                'fr_sulfide': Descriptors.fr_sulfide(mol),
                'fr_sulfonamd': Descriptors.fr_sulfonamd(mol),
                'fr_thiazole': Descriptors.fr_thiazole(mol),
                'fr_thiophene': Descriptors.fr_thiophene(mol),
                'fr_unbrch_alkane': Descriptors.fr_unbrch_alkane(mol),
                'fr_urea': Descriptors.fr_urea(mol),
            })

            # Handle NaN/Inf values
            for key, value in descriptors.items():
                if np.isnan(value) or np.isinf(value):
                    descriptors[key] = 0.0

        except Exception as e:
            print(f"Error calculating descriptors: {e}")

        return descriptors

    def predict_single_molecule_enhanced(self, smiles):
        """Enhanced prediction using all models"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, "Invalid SMILES string"

            # Get features
            descriptors = self.get_toxicity_descriptors(mol)
            fingerprints = self.get_molecular_fingerprints(mol)
            features = {}
            features.update(descriptors)
            features.update(fingerprints)

            # Create feature vector
            feature_vector = np.array([features.get(name, 0) for name in self.feature_names])
            feature_vector = feature_vector.reshape(1, -1)

            # Scale and select features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            feature_vector_selected = self.feature_selector.transform(feature_vector_scaled)

            # Get predictions from all models
            predictions = {}

            for name, model in self.models.items():
                if name == 'nn': # NN prediction is handled separately due to Keras model
                    continue
                pred = model.predict(feature_vector_selected)[0]
                proba = model.predict_proba(feature_vector_selected)[0]
                predictions[name] = {'prediction': pred, 'probability': proba[1]}

            # # Neural network prediction
            # nn_proba = self.nn_model.predict(feature_vector_selected)[0][0]
            # nn_pred = 1 if nn_proba > 0.5 else 0
            # predictions['nn'] = {'prediction': nn_pred, 'probability': nn_proba}


            # Ensemble prediction
            ensemble_pred = self.ensemble_model.predict(feature_vector_selected)[0]
            ensemble_proba = self.ensemble_model.predict_proba(feature_vector_selected)[0]
            predictions['ensemble'] = {'prediction': ensemble_pred, 'probability': ensemble_proba[1]}

            return predictions, None

        except Exception as e:
            return None, f"Error processing molecule: {str(e)}"

def enhanced_interactive_prediction(user_input):
    predictor = EnhancedToxicityPredictor()

    model_pkl_path = "Talksick-v2-ensemble_toxicity_model.pkl"

    with open(model_pkl_path, 'rb') as f:
        model_data = pickle.load(f)

    predictor.models = model_data['models']
    predictor.ensemble_model = model_data['ensemble_model']
    predictor.scaler = model_data['scaler']
    predictor.feature_selector = model_data['feature_selector']
    predictor.feature_names = model_data['feature_names']
    predictor.best_params = model_data['best_params']
    predictor.is_trained = True

    print("✅ Enhanced models loaded successfully")
    print("Processing with all models...")
    predictions, error = predictor.predict_single_molecule_enhanced(user_input)

    return predictions
   
def display_fancy_result_table(results_dict):
    if not results_dict:
        st.error("❌ No results to display")
        return
       
    model_name_map = {
        'rf': 'Random Forest',
        'xgb': 'XGBoost',
        'svm': 'SVM',
        'lgb': 'LightGBM',
        'ensemble': 'Ensemble'
    }

    # Create enhanced result display
    st.markdown("### Prediction Results")
   
    # Highlight ensemble result
    if 'ensemble' in results_dict:
        ensemble_result = results_dict['ensemble']
        pred_text = "Toxic" if ensemble_result['prediction'] == 1 else "Non-Toxic"
        prob_percent = ensemble_result['probability'] * 100 if ensemble_result['prediction'] == 1 else (1 - ensemble_result['probability']) * 100
       
        st.markdown(f"""
        <div class="ensemble-prediction">
            <h2>Ensemble Prediction</h2>
            <h1>{pred_text}</h1>
            <p>Confidence: {prob_percent:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Individual model results in a modern card layout
    cols = st.columns(2)
    col_idx = 0
   
    for key, val in results_dict.items():
        if key == 'ensemble':  # Skip ensemble as it's already shown above
            continue
           
        with cols[col_idx % 2]:
            name = model_name_map.get(key, key)
            pred_text = "Toxic" if val['prediction'] == 1 else "Non-Toxic"
            prob_percent = val['probability'] * 100 if val['prediction'] == 1 else (1 - val['probability']) * 100
           
            # Color based on prediction
            bg_color = "#fee2e2" if val['prediction'] == 1 else "#dcfce7"
            border_color = "#ef4444" if val['prediction'] == 1 else "#22c55e"
           
            st.markdown(f"""
            <div style="
                background: {bg_color};
                border: 2px solid {border_color};
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                text-align: center;
                transition: all 0.3s ease;
            ">
                <h4 style="margin-bottom: 0.5rem; color: #1e293b;">{name}</h4>
                <h3 style="margin: 0.5rem 0; color: #1e293b;">{pred_text}</h3>
                <p style="margin: 0; color: #64748b; font-weight: 500;">{prob_percent:.1f}% confidence</p>
            </div>
            """, unsafe_allow_html=True)
       
        col_idx += 1

def get_prediction_table(smile_str):
    try:
        result = enhanced_interactive_prediction(smile_str)
        return {'is_valid': True, "result": result}
    except Exception as e:
        return {'is_valid': False, "error": str(e)}

# Main App Layout
def main():
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-icon">☣️</div>
        <h1>ToxicSmiles</h1>
        <div class="subtitle">Machine Learning to Predict Molecular Toxicity</div>
    </div>
    """, unsafe_allow_html=True)

    # Create enhanced tabs with proper spacing
    tab1, tab2, tab3, tab4 = st.tabs([
        "Test Our Model", 
        "Why This Matters", 
        "How It Works", 
        "About Us"
    ])

    # Tab 1 - Test our model
    with tab1:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
       
        col1, col2 = st.columns([2, 1])
       
        with col1:
            st.markdown("### Enter SMILES String")
            st.markdown("Input your molecular SMILES notation to predict mitochondrial membrane potential toxicity")
           
            # Sample SMILES for testing
            st.markdown("**Try these examples:**")
            example_cols = st.columns(3)
            with example_cols[0]:
                if st.button("Caffeine", help="C1=CN=C2C(=N1)C(=NC=N2)N(C)C(=O)N(C)C"):
                    st.session_state.smiles_input = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
            with example_cols[1]:
                if st.button("Aspirin", help="CC(=O)OC1=CC=CC=C1C(=O)O"):
                    st.session_state.smiles_input = "CC(=O)OC1=CC=CC=C1C(=O)O"
            with example_cols[2]:
                if st.button("CCCP", help="C1=CC(=CC=C1NN=C(C#N)C#N)OC(F)(F)F"):
                    st.session_state.smiles_input = "C1=CC(=CC=C1NN=C(C#N)C#N)OC(F)(F)F"
           
            user_input = st.text_input(
                "SMILES Input:",
                value=st.session_state.get('smiles_input', ''),
                placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                help="Enter a valid SMILES string representing your molecule"
            )
           
            submit = st.button("Predict Toxicity", type="primary")
       
        with col2:
            st.markdown("### What You'll Get")
            st.markdown("""
            <div class="feature-list">
                <div class="feature-item">Toxicity Prediction</div>
                <div class="feature-item">Confidence Score</div>
                <div class="feature-item">Individual Model Results</div>
                <div class="feature-item">Ensemble Consensus</div>
            </div>
            """, unsafe_allow_html=True)

        if submit and user_input:
            with st.spinner("Analyzing molecular structure..."):
                result = get_prediction_table(user_input)
               
            if result['is_valid']:
                display_fancy_result_table(result["result"])
               
                # Additional information
                st.markdown("---")
                st.markdown("### Understanding the Results")
                st.info("""
                **Ensemble Prediction**: Our most reliable result, combining insights from multiple AI models.
               
                **Individual Models**: Each algorithm analyzes your molecule differently:
                - **Random Forest**: Analyzes molecular features through decision trees
                - **XGBoost**: Advanced gradient boosting for pattern recognition  
                - **SVM**: Finds optimal boundaries in chemical space
                - **LightGBM**: Efficient gradient boosting with high accuracy
               
                **Confidence Levels**: Higher percentages indicate stronger model certainty.
                """)
            else:
                st.error(f"❌ Error: {result.get('error', 'Invalid SMILES string')}")
                st.info("💡 **Tips for valid SMILES:**\n- Use standard chemical notation\n- Check for proper bonding\n- Ensure parentheses are balanced")
       
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2 - Why this matters
    with tab2:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown(why_this_matters_md, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 3 - How it works
    with tab3:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown(how_it_works_md, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 4 - About us
    with tab4:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        st.markdown(about_us_md, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()