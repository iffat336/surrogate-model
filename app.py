"""
Streamlit Web App for Surrogate Model Predictions
Hygrothermal-Mechanical Coupled Material Analysis
Deployed on Hugging Face Spaces (Free)
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(
    page_title="Surrogate Model: Hygrothermal-Mechanical Coupling",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main { padding: 2rem; }
        .stMetric { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; }
        h1 { color: #1e3a8a; border-bottom: 3px solid #3b82f6; padding-bottom: 0.5rem; }
        h2 { color: #1e40af; margin-top: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load pre-trained GP and RF models"""
    models_dir = Path(__file__).parent / 'results' / 'saved_models'
    
    try:
        with open(models_dir / 'gp_model.pkl', 'rb') as f:
            gp_model = pickle.load(f)
        with open(models_dir / 'rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return gp_model, rf_model, scaler, True
    except FileNotFoundError:
        return None, None, None, False

gp_model, rf_model, scaler, models_loaded = load_models()

@st.cache_resource
def load_material_props():
    """Load material properties database"""
    data_dir = Path(__file__).parent / 'data'
    with open(data_dir / 'material_properties.json', 'r') as f:
        return json.load(f)

material_db = load_material_props()

def calculate_damage_index(E_current, E_dry, critical_stiffness, time_days):
    """Calculate damage index (0=intact, 1=critical, >1=failed)"""
    if E_dry <= 0:
        return 0.0
    damage = (E_dry - E_current) / (E_dry - critical_stiffness)
    return max(0, min(damage, 1.5))

def predict_failure_time(E_current, E_degradation_rate, critical_stiffness):
    """Estimate days to failure"""
    if E_degradation_rate <= 0 or critical_stiffness >= E_current:
        return float('inf')
    T_fail = (E_current - critical_stiffness) / E_degradation_rate
    return max(0, T_fail)

def get_safety_status(damage_index, E_current, critical_stiffness):
    """Return safety status and color"""
    if E_current >= critical_stiffness * 1.5:
        return "✅ SAFE", "green"
    elif E_current >= critical_stiffness * 1.1:
        return "⚠️ DEGRADING", "orange"
    elif E_current >= critical_stiffness:
        return "🔴 RISKY", "red"
    else:
        return "❌ FAILED", "darkred"

# ============================================================================
# TITLE & HEADER
# ============================================================================
st.title("🔬 Surrogate Model: Hygrothermal-Mechanical Coupling")
st.markdown("""
**Predict moisture-induced mechanical degradation in biological materials**

This tool uses machine learning to rapidly approximate expensive FEM simulations
of moisture-mechanical coupled behavior in AAC, cardboard, and bio-composites.

**Speedup**: 3,600–50,000× faster than FEM | **Accuracy**: R² > 0.96 | **Latest**: Multi-material support & agricultural scenarios
""")

if not models_loaded:
    st.error("❌ Models not found. Please ensure saved_models/ directory exists with pickle files.")
    st.stop()

st.success("✅ Models loaded successfully")

# ============================================================================
# SIDEBAR: INPUT PARAMETERS
# ============================================================================
st.sidebar.header("⚙️ Material & Environmental Parameters")
st.sidebar.markdown("---")

# MATERIAL SELECTION (NEW)
material_choice = st.sidebar.radio(
    "📋 Select Material Type:",
    options=list(material_db['materials'].keys()),
    format_func=lambda x: f"{x} - {material_db['materials'][x]['name']}"
)
current_material = material_db['materials'][material_choice]

# Agricultural scenario quick select (NEW)
st.sidebar.markdown("---")
if st.sidebar.checkbox("🌾 Use Agricultural Scenario"):
    scenario_names = {s['name']: s for s in material_db['agricultural_scenarios'].values()}
    selected_scenario_name = st.sidebar.selectbox("Choose scenario:", list(scenario_names.keys()))
    selected_scenario = scenario_names[selected_scenario_name]
    st.sidebar.info(f"📌 {selected_scenario['description']}\n\n**Use:** {selected_scenario['use_case']}")
else:
    selected_scenario = None

st.sidebar.markdown("---")

# Material Parameters
st.sidebar.subheader("📋 Material Properties")
porosity = st.sidebar.slider("Porosity (%)", 
    int(current_material['porosity_range'][0]*100), 
    int(current_material['porosity_range'][1]*100), 
    int(np.mean(current_material['porosity_range'])*100), 
    1, help="Void fraction of material") / 100.0

density = st.sidebar.slider("Density (kg/m³)", 
    int(current_material['density_range'][0]), 
    int(current_material['density_range'][1]), 
    int(np.mean(current_material['density_range'])), 
    10, help="Material mass per volume")

thermal_cond = st.sidebar.slider("Thermal Conductivity (W/m·K)", 0.05, 0.35, 0.19, 0.01)

diffusivity = st.sidebar.select_slider(
    "Moisture Diffusivity (m²/s)",
    options=[1e-8, 3e-8, 1e-7, 3e-7, 1e-6],
    value=1e-7,
    help="Speed of moisture penetration"
)

E0_dry = st.sidebar.slider("Young's Modulus (dry, GPa)", 
    float(current_material['E0_dry_range'][0]), 
    float(current_material['E0_dry_range'][1]), 
    float(np.mean(current_material['E0_dry_range'])), 
    0.1, help="Baseline stiffness at 0% moisture")

alpha_E = st.sidebar.slider("Moisture Sensitivity (α_E)", 
    float(current_material['moisture_sensitivity_alpha'][0]), 
    float(current_material['moisture_sensitivity_alpha'][1]), 
    np.mean(current_material['moisture_sensitivity_alpha']), 
    0.02,
    help="Stiffness loss coefficient (higher = more moisture-sensitive)")

nu = st.sidebar.slider("Poisson's Ratio", 0.20, 0.35, 0.27, 0.01)

st.sidebar.markdown("---")

# Environmental Parameters
st.sidebar.subheader("🌍 Environmental Conditions")
rh_exposure = st.sidebar.slider("Relative Humidity (%)", 30, 95, 70, 5,
    help="Environmental moisture level")
temperature = st.sidebar.slider("Temperature (°C)", 10, 40, 25, 1)
thickness = st.sidebar.slider("Specimen Thickness (mm)", 5, 30, 15, 1,
    help="Affects moisture penetration depth")

st.sidebar.markdown("---")

# Mechanical Loading
st.sidebar.subheader("⚙️ Mechanical Loading")
load_magnitude = st.sidebar.slider("Load Magnitude (MPa)", 0.5, 5.0, 2.5, 0.1)
exposure_time = st.sidebar.slider("Exposure Time (days)", 1, 60, 14, 1,
    help="Duration of hygrothermal exposure")

st.sidebar.markdown("---")
st.sidebar.info("""
**Parameter Ranges:**
- Based on biological materials (AAC, cardboard, bio-composites)
- Typical applications: structural analysis, durability prediction
- For custom ranges, contact research team
""")

# ============================================================================
# PREPARE PREDICTION INPUT
# ============================================================================
feature_cols = [
    'porosity', 'density', 'thermal_cond', 'moisture_diffusivity',
    'E0_dry', 'E_sensitivity_to_moisture', 'nu',
    'RH_exposure', 'temperature', 'thickness',
    'load_magnitude', 'exposure_time_days'
]

X_input = np.array([[
    porosity, density, thermal_cond, diffusivity,
    E0_dry, alpha_E, nu,
    rh_exposure, temperature, thickness,
    load_magnitude, exposure_time
]])

# Standardize
X_scaled = scaler.transform(X_input)

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Gaussian Process Predictions")
    E_pred_gp, E_std_gp = gp_model.predict(X_scaled, return_std=True)
    E_pred_gp = E_pred_gp[0]
    E_std_gp = E_std_gp[0]
    
    # Metrics
    st.metric(
        "E_effective (wet)",
        f"{E_pred_gp:.3f} GPa",
        delta=f"±{1.96*E_std_gp:.3f} GPa (95% CI)",
        delta_color="off"
    )
    
    # Confidence visualization
    lower_bound = E_pred_gp - 1.96*E_std_gp
    upper_bound = E_pred_gp + 1.96*E_std_gp
    
    st.write("**95% Confidence Interval:**")
    col_l, col_m, col_u = st.columns(3)
    col_l.metric("Lower Bound", f"{lower_bound:.3f} GPa")
    col_m.metric("Central", f"{E_pred_gp:.3f} GPa", delta_color="off")
    col_u.metric("Upper Bound", f"{upper_bound:.3f} GPa")
    
    # Safety assessment
    critical_threshold = 1.2
    safety_margin = lower_bound - critical_threshold
    
    if safety_margin > 0.5:
        st.success(f"✅ **SAFE**: Safety margin = {safety_margin:.2f} GPa")
    elif safety_margin > 0:
        st.warning(f"⚠️ **CAUTION**: Safety margin = {safety_margin:.2f} GPa")
    else:
        st.error(f"❌ **CRITICAL**: Below safety threshold")

with col2:
    st.subheader("🌲 Random Forest Predictions")
    E_pred_rf = rf_model.predict(X_scaled)[0]
    
    # Metrics
    st.metric(
        "E_effective (wet)",
        f"{E_pred_rf:.3f} GPa",
        delta=f"{abs(E_pred_rf - E_pred_gp)/E_pred_gp*100:.1f}% vs GP",
        delta_color="off"
    )
    
    # Model comparison
    col_gp, col_rf = st.columns(2)
    col_gp.metric("GP Model", f"{E_pred_gp:.3f} GPa")
    col_rf.metric("RF Model", f"{E_pred_rf:.3f} GPa")
    
    # Ensemble prediction (average)
    E_ensemble = (E_pred_gp + E_pred_rf) / 2
    st.metric("Ensemble (Avg)", f"{E_ensemble:.3f} GPa", delta_color="off")
    
    st.info("""
    **Model Comparison:**
    - **GP**: Better for uncertainty quantification
    - **RF**: Better for feature interpretability
    - **Ensemble**: Robust consensus prediction
    """)

# ============================================================================
# SAFETY & DAMAGE ASSESSMENT (NEW)
# ============================================================================
st.markdown("---")
st.subheader("🛡️ Safety & Damage Assessment")

critical_stiffness = current_material['critical_stiffness']
damage_idx = calculate_damage_index(E_pred_gp, E0_dry, critical_stiffness, exposure_time)
E_degradation_rate = (E0_dry - E_pred_gp) / max(exposure_time, 1)
T_failure = predict_failure_time(E_pred_gp, E_degradation_rate, critical_stiffness)
safety_status, safety_color = get_safety_status(damage_idx, E_pred_gp, critical_stiffness)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<h3 style='color: {safety_color};'>{safety_status}</h3>", unsafe_allow_html=True)
with col2:
    st.metric("Damage Index", f"{damage_idx:.2f}")
with col3:
    if T_failure < 365:
        st.metric("Days to Failure", f"{T_failure:.1f}")
    else:
        st.metric("Days to Failure", ">1 year")
with col4:
    safety_margin = (E_pred_gp - critical_stiffness) / critical_stiffness * 100
    st.metric("Safety Margin", f"{max(0, safety_margin):.1f}%")

# ============================================================================
# DETAILED ANALYSIS
# ============================================================================
st.markdown("---")
st.subheader("📊 Detailed Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Degradation Analysis", "Feature Sensitivity", "Material Info", "🌾 Agricultural Use Cases", "📤 Export & Integration"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Moisture-Mechanical Coupling Effects:**")
        
        # Calculate degradation
        moisture_content = rh_exposure  # Simplified
        degradation_fraction = alpha_E * (moisture_content / 100)
        E_dry = E0_dry
        E_wet = E_pred_gp
        stiffness_loss = ((E_dry - E_wet) / E_dry) * 100
        
        st.metric("Dry Stiffness (E₀)", f"{E_dry:.3f} GPa")
        st.metric("Wet Stiffness (E_wet)", f"{E_wet:.3f} GPa")
        st.metric("Stiffness Loss", f"{stiffness_loss:.1f}%", delta_color="inverse")
        st.metric("Moisture @ RH", f"{moisture_content:.0f}%")
    
    with col2:
        st.write("**Mechanical Response:**")
        
        # Simplified stress-strain
        sigma = load_magnitude / (1 - porosity + 0.1)
        strain = sigma / E_wet if E_wet > 0 else 0
        
        st.metric("Applied Stress", f"{sigma:.3f} MPa")
        st.metric("Induced Strain", f"{strain:.4f}")
        st.metric("Stress/Stiffness", f"{sigma/E_wet:.4f}")
        
        # Damage estimate
        damage_index = (moisture_content / 100) * (sigma / 10) * alpha_E
        st.metric("Damage Index", f"{damage_index:.3f}", help="0=pristine, 1=failed")

with tab2:
    st.write("**How input parameters affect E_effective (wet):**")
    
    sensitivity_data = {
        'Parameter': ['E₀ (dry)', 'Load', 'Diffusivity', 'RH', 'α_E', 'Thickness'],
        'Importance': [0.854, 0.042, 0.030, 0.025, 0.023, 0.012],
        'Effect': ['🔴 Dominant', '🟠 Strong', '🟡 Moderate', '🟡 Moderate', '🟡 Moderate', '🟢 Weak']
    }
    
    sens_df = pd.DataFrame(sensitivity_data)
    st.dataframe(sens_df, use_container_width=True)
    
    st.info("""
    **Interpretation:**
    - **E₀ (dry)** (85%): Baseline stiffness is most important predictor
    - **Load + Diffusivity + RH + α_E** (12%): Coupled moisture-mechanical effects
    - **Geometry** (1%): Minor effect on stiffness prediction
    """)

with tab3:
    st.write("**Material Characterization:**")
    
    material_props = pd.DataFrame({
        'Property': ['Porosity', 'Density', 'Thermal Conductivity', 'Moisture Diffusivity', 
                     'Young\'s Modulus (dry)', 'Moisture Sensitivity', 'Poisson\'s Ratio'],
        'Value': [f'{porosity:.3f}', f'{density:.0f}', f'{thermal_cond:.3f}', f'{diffusivity:.2e}',
                  f'{E0_dry:.2f} GPa', f'{alpha_E:.3f}', f'{nu:.3f}'],
        'Unit': ['–', 'kg/m³', 'W/m·K', 'm²/s', 'GPa', '–', '–']
    })
    
    st.dataframe(material_props, use_container_width=True)
    
    # Material type inference
    if porosity > 0.5 and density < 600:
        mat_type = "AAC (Autoclaved Aerated Concrete)"
    elif porosity < 0.4 and density > 800:
        mat_type = "Dense Composite"
    else:
        mat_type = "Mixed Composite"
    
    st.success(f"**Inferred Material Type:** {mat_type}")

with tab4:
    st.header("🌾 Agricultural & Biosystems Applications")
    st.write(f"**Relevant use cases for {material_choice}:**")
    
    scenarios = list(material_db['agricultural_scenarios'].values())
    for i, scenario in enumerate(scenarios):
        with st.expander(f"📌 {scenario['name']}", expanded=(i==0)):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Material:** {scenario['material']}\n\n{scenario['description']}\n\n**Use case:** {scenario['use_case']}")
            with col2:
                params = scenario['parameters']
                st.markdown(f"**Scenario Parameters:**\n- RH: {params['RH']}%\n- Temp: {params['temperature']}°C\n- Load: {params['load_magnitude']} MPa\n- Duration: {params['exposure_time']} days")

with tab5:
    st.header("📤 Export & FEM Integration")
    col1, col2, col3 = st.columns(3)
    
    export_data = {
        'Material': material_choice,
        'Timestamp': datetime.now().isoformat(),
        'E_GP_GPa': E_pred_gp,
        'E_RF_GPa': E_pred_rf,
        'Uncertainty_GPa': 1.96*E_std_gp,
        'Damage_Index': damage_idx,
        'Days_to_Failure': T_failure
    }
    
    with col1:
        csv_data = pd.DataFrame([export_data]).to_csv(index=False)
        st.download_button("📊 Download CSV", csv_data, f"prediction_{material_choice}.csv", mime="text/csv")
    
    with col2:
        json_data = json.dumps(export_data, indent=2)
        st.download_button("📋 Download JSON", json_data, f"prediction_{material_choice}.json", mime="application/json")
    
    with col3:
        txt_report = f"PREDICTION REPORT\n{'='*50}\nMaterial: {material_choice}\nE_GP: {E_pred_gp:.4f} ± {E_std_gp:.4f} GPa\nE_RF: {E_pred_rf:.4f} GPa\nStatus: {safety_status}"
        st.download_button("📄 Download TXT", txt_report, f"report_{material_choice}.txt", mime="text/plain")

# ============================================================================
# EXPORT & DOCUMENTATION
# ============================================================================
st.markdown("---")
st.subheader("📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    # Download results as CSV
    results_df = pd.DataFrame({
        'Parameter': feature_cols,
        'Value': X_input[0]
    })
    
    results_df['GP_Prediction_GPa'] = [E_pred_gp] * len(feature_cols)
    results_df['RF_Prediction_GPa'] = [E_pred_rf] * len(feature_cols)
    results_df['Uncertainty_95CI_GPa'] = [1.96*E_std_gp] * len(feature_cols)
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📊 Download Results (CSV)",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

with col2:
    # Summary text
    summary_text = f"""
SURROGATE MODEL PREDICTION SUMMARY
====================================
Date: March 14, 2026
Model: Gaussian Process + Random Forest Ensemble

INPUT PARAMETERS:
- Porosity: {porosity:.3f}
- Density: {density:.0f} kg/m³
- RH Exposure: {rh_exposure}%
- Temperature: {temperature}°C
- Load: {load_magnitude} MPa
- Exposure Time: {exposure_time} days

PREDICTIONS:
- E_effective (GP): {E_pred_gp:.3f} ± {1.96*E_std_gp:.3f} GPa (95% CI)
- E_effective (RF): {E_pred_rf:.3f} GPa
- Stiffness Loss: {stiffness_loss:.1f}%

ASSESSMENT:
- Safety Margin: {safety_margin:.2f} GPa
- Status: {'SAFE' if safety_margin > 0.5 else 'CAUTION' if safety_margin > 0 else 'CRITICAL'}
    """
    
    st.download_button(
        label="📄 Download Summary (TXT)",
        data=summary_text,
        file_name="prediction_summary.txt",
        mime="text/plain"
    )

# ============================================================================
# FOOTER & INFO
# ============================================================================
st.markdown("---")
st.markdown("""
<center>

### 🔗 Project Information

**Surrogate Modeling for Hygrothermal-Mechanical Coupling in Biosystems Materials**

Developed for: **Prof. Tomasz Garbowski** & **Prof. Anna Szymczak-Graczyk**

Poznań University of Life Sciences | Department of Biosystems Engineering

**Models:**
- **Gaussian Process**: R² = 1.00, RMSE = 0.0053 GPa (with UQ)
- **Random Forest**: R² = 0.96, RMSE = 0.1603 GPa (feature importance)

**Speedup:** 3,600–50,000× faster than FEM | **Materials:** AAC, Cardboard, Bio-Composites

**Features:** Multi-material support • Damage prediction • Agricultural scenarios • FEM integration

**Validated for:**  Structural homogenization •  Digital twins •  Agricultural applications

</center>
""", unsafe_allow_html=True)
