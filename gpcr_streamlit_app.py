import io
import zipfile
from typing import List

import pandas as pd
import streamlit as st


# ---- Emerald Theme Palette ----
MINT_CREAM = "#E9F6EF"
SOFT_SAGE = "#9CC9AE"
RICH_EMERALD = "#1B7F5D"
DEEP_PINE = "#145943"
ACCENT_FOREST = "#0D3B2D"
WHITE = "#FFFFFF"


st.set_page_config(page_title="GPCR Functional Activity Studio", page_icon="🧬", layout="wide")

THEME_CSS = f"""
<style>
.stApp {{
    background: linear-gradient(135deg, {MINT_CREAM} 0%, #d9f1e6 40%, #f2fbf7 100%);
}}
.block-container {{
    background-color: rgba(255,255,255,0.98);
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(138,43,43,0.08);
}}
.stButton > button, .stDownloadButton > button {{
    background: {RICH_EMERALD} !important;
    color: {WHITE} !important;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-weight: 600;
}}
.stButton > button:hover {{
    background: {DEEP_PINE} !important;
    transform: translateY(-1px);
}}
[data-testid="stSidebar"] {{
    background: {ACCENT_FOREST};
    color: {WHITE};
}}
[data-testid="stSidebar"] * {{ color: {WHITE} !important; }}

h1, h2, h3, h4 {{ color: {RICH_EMERALD}; }}

[data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button * {{
    color: #000 !important;
    background: transparent !important;
}}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def _mock_gpcr_results(smiles: List[str], threshold: float) -> pd.DataFrame:
    # Mock outputs for UI-only preview
    base = pd.DataFrame({"smiles": smiles})
    base["p_agonist"] = 0.55
    base["p_antagonist"] = 0.35
    base["p_inactive"] = 0.10
    base["pred_label"] = base[["p_agonist", "p_antagonist", "p_inactive"]].idxmax(axis=1)
    base["binary_agonist_vs_antagonist"] = (base["p_agonist"] >= threshold).astype(int)
    return base


st.title("GPCR Functional Activity Studio")
st.caption("Upload SMILES or paste one per line to preview the UI for agonist vs antagonist predictions.")

with st.sidebar:
    st.header("Controls")
    st.text_input("Artifacts folder", value="artifacts", help="Disabled in UI-only mode")
    threshold = st.slider("Agonist threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.markdown("---")
    st.markdown("About: UI-only mode (model loading disabled).")


tabs = st.tabs(["Overview", "Predict", "Methods", "Results"])

with tabs[0]:
    st.header("Project Overview")
    st.markdown("**Title**")
    st.write("Functional Activity Prediction for Class A GPCR Ligands")
    st.markdown("**Authors (alphabetical)**")
    st.write("Sivanesan Dakshanamurthy, Sahith Mada, Joshua Mathew")
    st.markdown("**Abstract**")
    st.write(
        "This project builds machine learning models to classify GPCR ligands as agonist, antagonist, "
        "or inactive by integrating ligand descriptors, receptor pocket features, and interaction terms."
    )
    st.markdown("**Keywords**")
    st.write("GPCR, functional activity, ligand descriptors, receptor pocket, stacking ensemble")

with tabs[1]:
    st.header("Predict - submit molecules")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        smiles_text = st.text_area("SMILES (one per line)")
        uploaded = st.file_uploader("Or upload CSV (must contain `smiles` column)", type=["csv"], accept_multiple_files=False)
    with col_b:
        st.markdown("**Options**")
        show_raw = st.checkbox("Show raw probabilities", value=True)
        run_btn = st.button("Predict Activity", type="primary")

    smiles_list: List[str] = []
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if "smiles" not in df_up.columns:
                st.error("Uploaded CSV must contain a 'smiles' column.")
            else:
                smiles_list = df_up["smiles"].astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

    if smiles_text and not smiles_list:
        smiles_list = [s.strip() for s in smiles_text.splitlines() if s.strip()]

    if run_btn:
        if not smiles_list:
            st.error("Provide at least one SMILES string before predicting.")
        else:
            results = _mock_gpcr_results(smiles_list, threshold=threshold)

            if show_raw:
                st.subheader("Results (mock probabilities)")
                st.dataframe(results, use_container_width=True)

            csv_bytes = results.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv_bytes, file_name="gpcr_predictions.csv")

            mem = io.BytesIO()
            with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i, row in results.iterrows():
                    name = f"ligand_{i+1}.txt"
                    txt = (
                        f"smiles: {row['smiles']}\n"
                        f"p_agonist: {row['p_agonist']:.4f}\n"
                        f"p_antagonist: {row['p_antagonist']:.4f}\n"
                        f"p_inactive: {row['p_inactive']:.4f}\n"
                        f"label: {row['pred_label']}\n"
                    )
                    zf.writestr(name, txt)
            mem.seek(0)
            st.download_button("Download per-ligand summary (ZIP)", data=mem.getvalue(), file_name="gpcr_per_ligand.zip")

    if not run_btn and smiles_list:
        st.info(f"Ready to predict {len(smiles_list)} ligand(s). Click 'Predict Activity' to run.")

with tabs[2]:
    st.header("Methods at a Glance")
    st.markdown("- Ligands curated for 69 human Class A GPCRs from ChEMBL")
    st.markdown("- RDKit and Mordred used for 2D/3D descriptors")
    st.markdown("- Receptor pocket features derived from 3D structures")
    st.markdown("- Feature matrix combines ligand, receptor, and interaction terms")
    st.markdown("- Splits: random stratified, Bemis-Murcko scaffold, and LORO")
    st.markdown("- Models: Random Forest, XGBoost, LightGBM, and stacking ensemble")

with tabs[3]:
    st.header("Results Highlights")
    st.markdown("**Random Stratified Split**")
    st.write("Macro F1 around 0.80-0.81 for base learners; AUC near 0.97.")
    st.markdown("**Scaffold Split**")
    st.write("Macro F1 around 0.83-0.84; AUC around 0.97.")
    st.markdown("**LORO Split**")
    st.write("Macro F1 around 0.71-0.72; AUC around 0.92.")
    st.markdown("**Ensemble**")
    st.write("Stacked model improves precision and F1 on independent ligand validation.")
