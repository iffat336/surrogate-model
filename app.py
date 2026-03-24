"""
Streamlit app for a research-facing surrogate-model demo.

This interface is intentionally framed as a proof of concept. It demonstrates
how surrogate models can support hygrothermal-mechanical digital-twin workflows
for porous and bio-based materials. The current benchmark data are synthetic,
so the app should be presented as a workflow demonstration rather than a
publication-ready validated study.
"""

from datetime import datetime
import json
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Hygrothermal Digital Twin Demo",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main { padding: 2rem; }
        .stMetric {
            background-color: #f7f9fc;
            padding: 0.9rem;
            border-radius: 0.7rem;
            border: 1px solid #dbe4f0;
        }
        h1 {
            color: #183b74;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 0.5rem;
        }
        h2, h3 {
            color: #1d4d8f;
        }
        .research-box {
            background: #eef6ff;
            border: 1px solid #bfdbfe;
            border-radius: 0.8rem;
            padding: 1rem 1.1rem;
            margin: 1rem 0 1.1rem;
        }
        .context-box {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 0.8rem;
            padding: 1rem 1.1rem;
            margin: 0.8rem 0 1.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    """Load pre-trained surrogate models and scaler."""
    models_dir = Path(__file__).parent / "results" / "saved_models"
    try:
        with open(models_dir / "gp_model.pkl", "rb") as handle:
            gp_model = pickle.load(handle)
        with open(models_dir / "rf_model.pkl", "rb") as handle:
            rf_model = pickle.load(handle)
        with open(models_dir / "scaler.pkl", "rb") as handle:
            scaler = pickle.load(handle)
        return gp_model, rf_model, scaler, True
    except FileNotFoundError:
        return None, None, None, False


@st.cache_resource
def load_material_props():
    """Load material metadata."""
    data_path = Path(__file__).parent / "data" / "material_properties.json"
    with open(data_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def calculate_damage_index(e_current, e_dry, critical_stiffness):
    """Return a simple normalized damage measure."""
    if e_dry <= critical_stiffness:
        return 0.0
    damage = (e_dry - e_current) / (e_dry - critical_stiffness)
    return max(0.0, min(damage, 1.5))


def predict_failure_time(e_current, e_degradation_rate, critical_stiffness):
    """Estimate days to failure using a linearized degradation trend."""
    if e_degradation_rate <= 0 or critical_stiffness >= e_current:
        return float("inf")
    return max(0.0, (e_current - critical_stiffness) / e_degradation_rate)


def get_safety_status(e_current, critical_stiffness):
    """Return a simple status label and color."""
    if e_current >= critical_stiffness * 1.5:
        return "SAFE", "green"
    if e_current >= critical_stiffness * 1.1:
        return "DEGRADING", "orange"
    if e_current >= critical_stiffness:
        return "RISKY", "red"
    return "FAILED", "darkred"


def build_feature_frame(
    porosity,
    density,
    thermal_cond,
    diffusivity,
    e0_dry,
    alpha_e,
    nu,
    rh_exposure,
    temperature,
    thickness,
    load_magnitude,
    exposure_time,
):
    """Create a feature frame in the training-column order."""
    return pd.DataFrame(
        {
            "porosity": [porosity],
            "density": [density],
            "thermal_cond": [thermal_cond],
            "moisture_diffusivity": [diffusivity],
            "E0_dry": [e0_dry],
            "E_sensitivity_to_moisture": [alpha_e],
            "nu": [nu],
            "RH_exposure": [rh_exposure],
            "temperature": [temperature],
            "thickness": [thickness],
            "load_magnitude": [load_magnitude],
            "exposure_time_days": [exposure_time],
        }
    )


gp_model, rf_model, scaler, models_loaded = load_models()
material_db = load_material_props()

st.title("Hygrothermal Digital Twin and Surrogate Modeling Demo")
st.markdown(
    """
    **Research-facing proof of concept for moisture-sensitive porous and bio-based materials**

    This app demonstrates how a surrogate model can approximate a slower reference
    model for coupled hygrothermal-mechanical response. It is designed to support
    research discussion around computational mechanics, moisture durability,
    digital-twin workflows, and biosystems or environmental applications.
    """
)

st.markdown(
    """
    <div class="research-box">
      <strong>Interview-safe framing:</strong> this app is built on a synthetic benchmark dataset.
      It is useful for explaining workflow, uncertainty-aware surrogate modeling, and deployment logic.
      A publication-grade study would replace the synthetic benchmark with validated FEM output and/or
      experimental measurements.
    </div>
    """,
    unsafe_allow_html=True,
)

if not models_loaded:
    st.error("Models not found. Please ensure results/saved_models contains the trained artifacts.")
    st.stop()

st.success("Demo models loaded successfully.")

st.sidebar.header("Research Scenario and Inputs")
st.sidebar.markdown("---")

study_mode = st.sidebar.selectbox(
    "Select research framing:",
    [
        "Biosystems durability",
        "Building-material hygrothermal performance",
        "Digital-twin surrogate workflow",
    ],
)

study_descriptions = {
    "Biosystems durability": (
        "Use this mode when discussing bio-based materials, plant-linked systems, "
        "agricultural packaging, or storage durability."
    ),
    "Building-material hygrothermal performance": (
        "Use this mode when discussing AAC, perlite concrete, porous wall elements, "
        "moisture permeability, and durability under environmental exposure."
    ),
    "Digital-twin surrogate workflow": (
        "Use this mode when discussing validated reference models, surrogate comparison, "
        "uncertainty estimates, and deployment for fast scenario testing."
    ),
}
st.sidebar.info(study_descriptions[study_mode])

material_choice = st.sidebar.radio(
    "Select material type:",
    options=list(material_db["materials"].keys()),
    format_func=lambda key: f"{key} - {material_db['materials'][key]['name']}",
)
current_material = material_db["materials"][material_choice]

st.sidebar.markdown("---")
use_scenario = st.sidebar.checkbox("Use preset research scenario")

selected_scenario = None
scenario_defaults = {
    "RH": 70,
    "temperature": 25,
    "load_magnitude": 2.5,
    "exposure_time": 14,
}

if use_scenario:
    scenario_names = {
        item["name"]: item for item in material_db["agricultural_scenarios"].values()
    }
    selected_scenario_name = st.sidebar.selectbox("Choose scenario:", list(scenario_names.keys()))
    selected_scenario = scenario_names[selected_scenario_name]
    scenario_defaults = selected_scenario["parameters"]
    scenario_material = selected_scenario["material"]
    if scenario_material in material_db["materials"] and scenario_material != material_choice:
        material_choice = scenario_material
        current_material = material_db["materials"][material_choice]
        st.sidebar.caption("Preset scenario overrides the selected material metadata.")
    st.sidebar.info(
        f"{selected_scenario['description']}\n\nUse case: {selected_scenario['use_case']}"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Material Properties")

porosity = st.sidebar.slider(
    "Porosity (%)",
    int(current_material["porosity_range"][0] * 100),
    int(current_material["porosity_range"][1] * 100),
    int(np.mean(current_material["porosity_range"]) * 100),
    1,
    help="Void fraction of the material",
) / 100.0

density = st.sidebar.slider(
    "Density (kg/m^3)",
    int(current_material["density_range"][0]),
    int(current_material["density_range"][1]),
    int(np.mean(current_material["density_range"])),
    10,
    help="Material mass per volume",
)

thermal_cond = st.sidebar.slider(
    "Thermal Conductivity (W/m.K)",
    0.05,
    0.35,
    0.19,
    0.01,
)

diffusivity = st.sidebar.select_slider(
    "Moisture Diffusivity (m^2/s)",
    options=[1e-8, 3e-8, 1e-7, 3e-7, 1e-6],
    value=1e-7,
    help="Approximate moisture transport rate",
)

e0_dry = st.sidebar.slider(
    "Young's Modulus (dry, GPa)",
    float(current_material["E0_dry_range"][0]),
    float(current_material["E0_dry_range"][1]),
    float(np.mean(current_material["E0_dry_range"])),
    0.1,
    help="Baseline stiffness at dry state",
)

alpha_e = st.sidebar.slider(
    "Moisture Sensitivity (alpha_E)",
    float(current_material["moisture_sensitivity_alpha"][0]),
    float(current_material["moisture_sensitivity_alpha"][1]),
    float(np.mean(current_material["moisture_sensitivity_alpha"])),
    0.02,
    help="Larger values indicate stronger stiffness loss under moisture exposure",
)

nu = st.sidebar.slider("Poisson's Ratio", 0.20, 0.35, 0.27, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Environmental Conditions")

rh_exposure = st.sidebar.slider(
    "Relative Humidity (%)",
    30,
    95,
    int(scenario_defaults["RH"]),
    5,
    help="Environmental moisture level",
)

temperature = st.sidebar.slider(
    "Temperature (C)",
    10,
    40,
    int(scenario_defaults["temperature"]),
    1,
)

thickness = st.sidebar.slider(
    "Specimen Thickness (mm)",
    5,
    30,
    15,
    1,
    help="Affects moisture penetration depth",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Mechanical Loading")

load_magnitude = st.sidebar.slider(
    "Load Magnitude (MPa)",
    0.5,
    5.0,
    float(scenario_defaults["load_magnitude"]),
    0.1,
)

exposure_time = st.sidebar.slider(
    "Exposure Time (days)",
    1,
    60,
    int(scenario_defaults["exposure_time"]),
    1,
    help="Duration of hygrothermal exposure",
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Parameter ranges**
    - Demonstration ranges based on porous, bio-based, and moisture-sensitive materials
    - Useful for discussing durability, hygrothermal response, and surrogate behavior
    - In a formal study, ranges should be calibrated to validated reference data
    """
)

feature_cols = [
    "porosity",
    "density",
    "thermal_cond",
    "moisture_diffusivity",
    "E0_dry",
    "E_sensitivity_to_moisture",
    "nu",
    "RH_exposure",
    "temperature",
    "thickness",
    "load_magnitude",
    "exposure_time_days",
]

feature_frame = build_feature_frame(
    porosity=porosity,
    density=density,
    thermal_cond=thermal_cond,
    diffusivity=diffusivity,
    e0_dry=e0_dry,
    alpha_e=alpha_e,
    nu=nu,
    rh_exposure=rh_exposure,
    temperature=temperature,
    thickness=thickness,
    load_magnitude=load_magnitude,
    exposure_time=exposure_time,
)
x_input = feature_frame[feature_cols].to_numpy()
x_scaled = scaler.transform(feature_frame[feature_cols])

if selected_scenario is not None:
    scenario_label = (
        f"{selected_scenario['name']} | use case: {selected_scenario['use_case']}"
    )
else:
    scenario_label = f"Custom scenario for {current_material['name']}"

st.markdown(
    f"""
    <div class="context-box">
      <strong>Current discussion mode:</strong> {study_mode}<br>
      <strong>Selected material:</strong> {current_material['name']}<br>
      <strong>Scenario:</strong> {scenario_label}
    </div>
    """,
    unsafe_allow_html=True,
)

e_pred_gp, e_std_gp = gp_model.predict(x_scaled, return_std=True)
e_pred_gp = float(e_pred_gp[0])
e_std_gp = float(e_std_gp[0])
e_pred_rf = float(rf_model.predict(x_scaled)[0])
e_ensemble = (e_pred_gp + e_pred_rf) / 2.0
lower_bound = e_pred_gp - 1.96 * e_std_gp
upper_bound = e_pred_gp + 1.96 * e_std_gp

critical_stiffness = current_material["critical_stiffness"]
damage_idx = calculate_damage_index(e_pred_gp, e0_dry, critical_stiffness)
e_degradation_rate = (e0_dry - e_pred_gp) / max(exposure_time, 1)
t_failure = predict_failure_time(e_pred_gp, e_degradation_rate, critical_stiffness)
safety_status, safety_color = get_safety_status(e_pred_gp, critical_stiffness)
stiffness_loss = ((e0_dry - e_pred_gp) / e0_dry) * 100 if e0_dry else 0.0

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gaussian Process Surrogate")
    st.metric(
        "E_effective (wet)",
        f"{e_pred_gp:.3f} GPa",
        delta=f"+/- {1.96 * e_std_gp:.3f} GPa (95% CI)",
        delta_color="off",
    )

    gp_ci_cols = st.columns(3)
    gp_ci_cols[0].metric("Lower Bound", f"{lower_bound:.3f} GPa")
    gp_ci_cols[1].metric("Central", f"{e_pred_gp:.3f} GPa")
    gp_ci_cols[2].metric("Upper Bound", f"{upper_bound:.3f} GPa")

    threshold_margin = lower_bound - critical_stiffness
    if threshold_margin > 0.5:
        st.success(f"SAFE: confidence margin = {threshold_margin:.2f} GPa")
    elif threshold_margin > 0:
        st.warning(f"CAUTION: margin is only {threshold_margin:.2f} GPa")
    else:
        st.error("CRITICAL: lower confidence bound falls below the chosen threshold")

with col2:
    st.subheader("Random Forest Baseline")
    st.metric(
        "E_effective (wet)",
        f"{e_pred_rf:.3f} GPa",
        delta=f"{abs(e_pred_rf - e_pred_gp) / max(abs(e_pred_gp), 1e-6) * 100:.1f}% vs GP",
        delta_color="off",
    )

    compare_cols = st.columns(3)
    compare_cols[0].metric("GP", f"{e_pred_gp:.3f} GPa")
    compare_cols[1].metric("RF", f"{e_pred_rf:.3f} GPa")
    compare_cols[2].metric("Ensemble", f"{e_ensemble:.3f} GPa")

    st.info(
        """
        **How to explain the models**
        - GP is useful when uncertainty estimates matter.
        - RF is a strong nonlinear baseline with interpretability.
        - The ensemble is only a simple consensus check, not a replacement for validation.
        """
    )

st.markdown("---")
st.subheader("Durability and Safety-Oriented Assessment")

status_cols = st.columns(4)
with status_cols[0]:
    st.markdown(
        f"<h3 style='color: {safety_color};'>{safety_status}</h3>",
        unsafe_allow_html=True,
    )
with status_cols[1]:
    st.metric("Damage Index", f"{damage_idx:.2f}")
with status_cols[2]:
    if t_failure < 365:
        st.metric("Estimated Days to Threshold", f"{t_failure:.1f}")
    else:
        st.metric("Estimated Days to Threshold", "> 365")
with status_cols[3]:
    margin_pct = ((e_pred_gp - critical_stiffness) / critical_stiffness) * 100
    st.metric("Safety Margin", f"{max(0.0, margin_pct):.1f}%")

st.markdown("---")
st.subheader("Detailed Analysis")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Response Analysis",
        "Feature Sensitivity",
        "Material Context",
        "Research Use Cases",
        "Export and Integration",
        "Validation and Next Steps",
    ]
)

with tab1:
    analysis_cols = st.columns(2)

    with analysis_cols[0]:
        st.write("**Moisture-mechanical coupling effects**")
        moisture_content = float(rh_exposure)
        st.metric("Dry Stiffness (E0)", f"{e0_dry:.3f} GPa")
        st.metric("Wet Stiffness (E_wet)", f"{e_pred_gp:.3f} GPa")
        st.metric("Stiffness Loss", f"{stiffness_loss:.1f}%")
        st.metric("Moisture Indicator", f"{moisture_content:.0f}% RH")

    with analysis_cols[1]:
        st.write("**Mechanical response indicators**")
        sigma = load_magnitude / (1 - porosity + 0.1)
        strain = sigma / e_pred_gp if e_pred_gp > 0 else 0.0
        damage_proxy = (moisture_content / 100.0) * (sigma / 10.0) * alpha_e

        st.metric("Applied Stress Proxy", f"{sigma:.3f} MPa")
        st.metric("Induced Strain Proxy", f"{strain:.4f}")
        st.metric("Stress/Stiffness Ratio", f"{sigma / max(e_pred_gp, 1e-6):.4f}")
        st.metric("Damage Proxy", f"{damage_proxy:.3f}")

    st.info(
        "Use this tab to answer in a structured way: inputs -> surrogate response -> "
        "degradation indicators -> engineering relevance."
    )

    st.markdown("**Graph: stiffness comparison for the active scenario**")
    stiffness_chart_df = pd.DataFrame(
        {
            "State": ["Dry stiffness", "GP wet stiffness", "RF wet stiffness", "Critical threshold"],
            "GPa": [e0_dry, e_pred_gp, e_pred_rf, critical_stiffness],
        }
    ).set_index("State")
    st.bar_chart(stiffness_chart_df)
    st.caption(
        "Interpretation: this graph compares the dry baseline, the predicted wet response, and the "
        "critical threshold. It helps show how much stiffness is lost under moisture exposure and "
        "whether the predicted wet state remains safely above the threshold."
    )

    rh_values = np.arange(30, 96, 5)
    rh_sweep_frame = pd.DataFrame(
        {
            "porosity": porosity,
            "density": density,
            "thermal_cond": thermal_cond,
            "moisture_diffusivity": diffusivity,
            "E0_dry": e0_dry,
            "E_sensitivity_to_moisture": alpha_e,
            "nu": nu,
            "RH_exposure": rh_values,
            "temperature": temperature,
            "thickness": thickness,
            "load_magnitude": load_magnitude,
            "exposure_time_days": exposure_time,
        }
    )
    rh_scaled = scaler.transform(rh_sweep_frame[feature_cols])
    gp_rh_mean, _ = gp_model.predict(rh_scaled, return_std=True)
    rf_rh_mean = rf_model.predict(rh_scaled)
    rh_chart_df = pd.DataFrame(
        {
            "Relative Humidity (%)": rh_values,
            "Gaussian Process": gp_rh_mean,
            "Random Forest": rf_rh_mean,
        }
    ).set_index("Relative Humidity (%)")
    st.markdown("**Graph: predicted wet stiffness across humidity levels**")
    st.line_chart(rh_chart_df)
    st.caption(
        "Interpretation: this graph shows how the predicted wet stiffness changes as relative humidity "
        "increases while the other conditions stay fixed. The main message is that environmental "
        "exposure and baseline material properties interact in the prediction workflow."
    )

with tab2:
    st.write("**Illustrative feature ranking from the synthetic benchmark**")
    sensitivity_df = pd.DataFrame(
        {
            "Parameter": ["E0 (dry)", "Load", "Diffusivity", "RH", "alpha_E", "Thickness"],
            "Importance": [0.854, 0.042, 0.030, 0.025, 0.023, 0.012],
            "Interpretation": [
                "Dominant baseline stiffness term",
                "Strong loading contribution",
                "Moisture transport effect",
                "Environmental exposure effect",
                "Material moisture sensitivity",
                "Minor geometric effect",
            ],
        }
    )
    st.dataframe(sensitivity_df, use_container_width=True)
    st.info(
        """
        In the interview, describe this as an illustrative ranking from the benchmark,
        not a universal scientific conclusion. The main point is that baseline stiffness,
        moisture transport, and environmental exposure interact in the prediction workflow.
        """
    )
    st.caption(
        "Table interpretation: the ranking suggests that dry baseline stiffness dominates this benchmark, "
        "while load, diffusivity, relative humidity, and moisture sensitivity provide the coupled "
        "environmental-mechanical contribution to the final prediction."
    )
    feature_chart_df = sensitivity_df.set_index("Parameter")[["Importance"]]
    st.markdown("**Graph: feature-importance profile**")
    st.bar_chart(feature_chart_df)
    st.caption(
        "Interpretation: this graph visualizes the same benchmark ranking in a more intuitive way. "
        "It helps explain that the model is driven first by baseline stiffness, then by moisture- and "
        "load-related factors, rather than by geometry alone."
    )

with tab3:
    st.write("**Material characterization**")
    material_props = pd.DataFrame(
        {
            "Property": [
                "Porosity",
                "Density",
                "Thermal Conductivity",
                "Moisture Diffusivity",
                "Young's Modulus (dry)",
                "Moisture Sensitivity",
                "Poisson's Ratio",
            ],
            "Value": [
                f"{porosity:.3f}",
                f"{density:.0f}",
                f"{thermal_cond:.3f}",
                f"{diffusivity:.2e}",
                f"{e0_dry:.2f} GPa",
                f"{alpha_e:.3f}",
                f"{nu:.3f}",
            ],
            "Unit": ["-", "kg/m^3", "W/m.K", "m^2/s", "GPa", "-", "-"],
        }
    )
    st.dataframe(material_props, use_container_width=True)
    st.caption(
        "Table interpretation: this table summarizes the active material state used in the current scenario. "
        "It is useful for linking the prediction to physically meaningful descriptors such as porosity, "
        "density, diffusivity, and moisture sensitivity."
    )

    if porosity > 0.5 and density < 650:
        inferred_type = "Highly porous wall-material analogue"
    elif porosity < 0.4 and density > 800:
        inferred_type = "Dense composite-like analogue"
    else:
        inferred_type = "Intermediate porous composite"

    st.success(f"Inferred material family: {inferred_type}")
    st.write("**Research areas connected to this material**")
    st.write(", ".join(current_material["research_areas"]))
    st.write("**Representative applications**")
    st.write(", ".join(current_material["applications"]))

    normalized_props_df = pd.DataFrame(
        {
            "Property": ["Porosity", "Density", "Thermal Cond.", "Diffusivity", "Dry Stiffness", "Moisture Sens."],
            "Normalized value": [
                porosity,
                (density - current_material["density_range"][0])
                / max(current_material["density_range"][1] - current_material["density_range"][0], 1),
                (thermal_cond - 0.05) / 0.30,
                (
                    np.log10(diffusivity) - np.log10(1e-8)
                ) / max(np.log10(1e-6) - np.log10(1e-8), 1e-6),
                (e0_dry - current_material["E0_dry_range"][0])
                / max(current_material["E0_dry_range"][1] - current_material["E0_dry_range"][0], 1e-6),
                (alpha_e - current_material["moisture_sensitivity_alpha"][0])
                / max(
                    current_material["moisture_sensitivity_alpha"][1]
                    - current_material["moisture_sensitivity_alpha"][0],
                    1e-6,
                ),
            ],
        }
    ).set_index("Property")
    st.markdown("**Graph: normalized material profile for the active case**")
    st.bar_chart(normalized_props_df)
    st.caption(
        "Interpretation: this graph shows where the current case sits within its material-property ranges. "
        "It helps explain whether the selected case represents a lighter, denser, more moisture-sensitive, "
        "or stiffer part of the chosen material family."
    )

with tab4:
    st.header("Research-facing use cases")
    st.write(f"**Relevant scenarios for {material_choice}**")

    for index, scenario in enumerate(material_db["agricultural_scenarios"].values()):
        with st.expander(scenario["name"], expanded=(index == 0)):
            left, right = st.columns(2)
            with left:
                st.markdown(
                    f"**Material:** {scenario['material']}\n\n"
                    f"{scenario['description']}\n\n"
                    f"**Use case:** {scenario['use_case']}"
                )
            with right:
                params = scenario["parameters"]
                st.markdown(
                    f"**Scenario parameters**\n"
                    f"- RH: {params['RH']}%\n"
                    f"- Temperature: {params['temperature']} C\n"
                    f"- Load: {params['load_magnitude']} MPa\n"
                    f"- Duration: {params['exposure_time']} days"
                )

    st.markdown(
        """
        **How this aligns with likely PhD directions**
        - Hygrothermal performance and moisture permeability of porous materials
        - Surrogate-assisted approximation of a validated reference model
        - Digital-twin style monitoring and fast scenario evaluation
        - Durability and mechanical response under environmental exposure
        """
    )

with tab5:
    st.header("Export and Integration")
    export_data = {
        "Material": material_choice,
        "Timestamp": datetime.now().isoformat(),
        "Study_Mode": study_mode,
        "Scenario": scenario_label,
        "E_GP_GPa": e_pred_gp,
        "E_RF_GPa": e_pred_rf,
        "Uncertainty_GPa": 1.96 * e_std_gp,
        "Damage_Index": damage_idx,
        "Estimated_Days_to_Threshold": t_failure,
    }

    export_cols = st.columns(3)
    with export_cols[0]:
        csv_data = pd.DataFrame([export_data]).to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv_data,
            f"prediction_{material_choice}.csv",
            mime="text/csv",
        )
    with export_cols[1]:
        json_data = json.dumps(export_data, indent=2)
        st.download_button(
            "Download JSON",
            json_data,
            f"prediction_{material_choice}.json",
            mime="application/json",
        )
    with export_cols[2]:
        txt_report = (
            "PREDICTION REPORT\n"
            + "=" * 50
            + f"\nMaterial: {material_choice}"
            + f"\nStudy mode: {study_mode}"
            + f"\nScenario: {scenario_label}"
            + f"\nE_GP: {e_pred_gp:.4f} +/- {e_std_gp:.4f} GPa"
            + f"\nE_RF: {e_pred_rf:.4f} GPa"
            + f"\nStatus: {safety_status}"
        )
        st.download_button(
            "Download TXT",
            txt_report,
            f"report_{material_choice}.txt",
            mime="text/plain",
        )

    st.info(
        """
        A real research workflow would export calibrated material states, surrogate inputs,
        and validation summaries for comparison against FEM or laboratory results.
        """
    )

with tab6:
    st.header("Validation and Next Steps")
    st.markdown(
        """
        **What this demo already shows**
        - A full workflow from input conditions to surrogate prediction and uncertainty-aware reporting
        - Comparison between two surrogate approaches in the same feature space
        - A deployable interface for scenario testing and research discussion

        **What would make it publication-grade**
        - Replace the synthetic benchmark with validated FEM output and/or laboratory measurements
        - Compare ANN, GP, and tree-based models under the same train/validation/test split
        - Report MAE, RMSE, R2, parity plots, residual plots, and uncertainty calibration
        - Test generalization across material classes, moisture regimes, and loading conditions

        **Best way to explain this in the interview**
        - Problem: high-fidelity reference models are informative but expensive
        - Method: train surrogate models on the reference response
        - Result: enable fast scenario evaluation with uncertainty estimates
        - Relevance: supports digital twins, durability studies, and biosystems or environmental applications
        """
    )

st.markdown("---")
st.subheader("Export Results")

result_cols = st.columns(2)

with result_cols[0]:
    results_df = pd.DataFrame({"Parameter": feature_cols, "Value": x_input[0]})
    results_df["GP_Prediction_GPa"] = [e_pred_gp] * len(feature_cols)
    results_df["RF_Prediction_GPa"] = [e_pred_rf] * len(feature_cols)
    results_df["Uncertainty_95CI_GPa"] = [1.96 * e_std_gp] * len(feature_cols)
    csv_blob = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results (CSV)",
        data=csv_blob,
        file_name="prediction_results.csv",
        mime="text/csv",
    )

with result_cols[1]:
    summary_text = f"""
SURROGATE MODEL PREDICTION SUMMARY
====================================
Date: {datetime.now().strftime("%B %d, %Y")}
Model family: Gaussian Process + Random Forest

INPUT PARAMETERS:
- Porosity: {porosity:.3f}
- Density: {density:.0f} kg/m^3
- RH Exposure: {rh_exposure}%
- Temperature: {temperature} C
- Load: {load_magnitude} MPa
- Exposure Time: {exposure_time} days

PREDICTIONS:
- E_effective (GP): {e_pred_gp:.3f} +/- {1.96 * e_std_gp:.3f} GPa (95% CI)
- E_effective (RF): {e_pred_rf:.3f} GPa
- Stiffness Loss: {stiffness_loss:.1f}%

ASSESSMENT:
- Safety Margin: {max(0.0, margin_pct):.2f}%
- Status: {safety_status}
- Scenario: {scenario_label}
"""
    st.download_button(
        label="Download Summary (TXT)",
        data=summary_text,
        file_name="prediction_summary.txt",
        mime="text/plain",
    )

st.markdown("---")
st.markdown(
    """
    <center>

    ### Project Information

    **Surrogate-assisted hygrothermal-mechanical response demo**

    Current framing: proof of concept for digital twins, surrogate validation, and durability studies

    Suggested interview message: use this app to discuss research workflow, not to overclaim final validation

    **Synthetic benchmark models**
    - Gaussian Process: strong interpolation performance with uncertainty estimates
    - Random Forest: nonlinear baseline with feature-importance view

    **Materials**
    AAC, cardboard, bio-composites, and porous wall-material analogues

    **Best discussion themes**
    structural homogenization, hygrothermal performance, moisture durability, and digital twins

    </center>
    """,
    unsafe_allow_html=True,
)
