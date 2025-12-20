"""
Gradio app for warfarin dose prediction
======================================

This script exposes a simple web interface built with Gradio that
wraps a previously trained XGBoost regression pipeline.  The model
expects a fixed set of 29 engineered feature columns.  To make the
service usable without requiring users to construct their own CSVs,
the interface collects a handful of clinically relevant inputs and
derives the remaining engineered features automatically.  Any
attribute that is not explicitly collected from the user is filled
with a sensible default – either zero for numeric fields or
``"Unknown"`` for categorical fields.  These defaults may differ
slightly from the medians and modes used during training, but the
presence of the full feature set ensures the model does not raise
missing‐column errors.

The core logic lives in :func:`build_feature_row`, which turns
user inputs into a single‐row :class:`pandas.DataFrame` matching the
model’s schema.  The :func:`predict_dose` function performs the
prediction using the loaded model.  Finally, the Gradio
``Interface`` ties everything together by defining input widgets
for each parameter and a textbox output for the predicted dose.

To launch the app locally, run this module directly::

    python app.py

When deployed on Hugging Face Spaces, the platform will execute
``app.py`` automatically if present at the repository root.
"""

from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

try:
    # Gradio is only required when launching the web interface.  When
    # importing this module for testing or offline predictions, the
    # gradio import may fail if the dependency is not installed.  To
    # allow non‑interactive use, we wrap the import in a try/except
    # block and only raise the error when actually launching the app.
    import gradio as gr  # type: ignore
except ModuleNotFoundError:
    gr = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
# The trained pipeline must be located in the same directory as this
# script.  When running on Hugging Face, the `.joblib` file should
# be uploaded alongside `app.py` in the repository.  If the file
# cannot be found, an informative error message will be raised.
MODEL_FILENAME = "warfarin_xgb_pipeline.joblib"
MODEL_PATH = Path(__file__).with_name(MODEL_FILENAME)
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Required model file '{MODEL_FILENAME}' not found in {MODEL_PATH.parent}. "
        "Upload the trained joblib along with this script."
    )

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Expected feature columns
# ---------------------------------------------------------------------------
# This list enumerates all features the pipeline was trained on.  It was
# extracted by inspecting the `ColumnTransformer` inside the persisted
# pipeline.  Do not reorder the entries – the model relies on this
# ordering.
EXPECTED_COLS: list[str] = [
    # numeric columns
    "Age",
    "Weight_kg",
    "Height_cm",
    "Hypertension",
    "Diabetes",
    "Chronic_Kidney_Disease",
    "Heart_Failure",
    "Amiodarone",
    "Antibiotics",
    "Aspirin",
    "Statins",
    "INR_Stabilization_Days",
    "Time_in_Therapeutic_Range_Pct",
    "Adverse_Event_Flag",
    "CYP2C9_risk",
    "VKORC1_sensitivity",
    "CYP4F2_effect",
    "BMI",
    "Amiodarone_CYP2C9_interaction",
    "Comorbidity_Score",
    # categorical columns
    "CYP2C9",
    "VKORC1",
    "CYP4F2",
    "Sex",
    "Ethnicity",
    "Alcohol_Intake",
    "Smoking_Status",
    "Diet_VitK_Intake",
    "Adverse_Event",
]


def genotype_to_risk(genotype: str) -> float:
    """Convert a CYP2C9 genotype (e.g. ``"*1/*2"``) into a numeric risk
    score.  The heuristic counts the number of variant alleles (any
    allele other than ``*1``) and returns that count as a float.
    """
    g = str(genotype or "").strip()
    if "*" not in g:
        return 0.0
    # split on slash and count non‐*1 entries
    alleles = [a.strip() for a in g.split("/") if a.strip()]
    return float(sum(1 for a in alleles if a != "*1"))


def genotype_to_sensitivity(genotype: str) -> float:
    """Convert a VKORC1 genotype (``"G/G"``, ``"A/G"`` or ``"A/A"``) to
    a numeric sensitivity score between 0 and 1.
    """
    g = str(genotype or "").replace("/", "").upper()
    if g in {"AA", "A A"}:
        return 1.0
    if g in {"AG", "GA"}:
        return 0.5
    return 0.0


def genotype_to_effect(genotype: str) -> float:
    """Convert a CYP4F2 genotype (``"Normal"`` or ``"Increased"``) to a
    numeric effect.  ``"Increased"`` returns 1.0, anything else returns 0.
    """
    g = str(genotype or "").strip().lower()
    return 1.0 if g in {"increased", "variant", "mutant", "incr"} else 0.0


def build_feature_row(
    *,
    age: float,
    weight: float,
    height: float,
    sex: str,
    ethnicity: str,
    cyp2c9: str,
    vkorc1: str,
    cyp4f2: str,
    on_amiodarone: bool,
    on_antibiotics: bool,
    on_aspirin: bool,
    on_statins: bool,
    has_hypertension: bool,
    has_diabetes: bool,
    has_ckd: bool,
    has_heart_failure: bool,
    alcohol_intake: str = "Unknown",
    smoking_status: str = "Unknown",
    diet_vitk: str = "Unknown",
    inr_days: float = 0.0,
    ttr_pct: float = 0.0,
    adverse_event_flag: bool,
    adverse_event: str = "None",
    comorbidity_score: float | None = None,
) -> pd.DataFrame:
    """Construct a 1‑row DataFrame matching the expected schema.

    Parameters
    ----------
    age, weight, height : numeric
        Basic demographic inputs.

    sex : {"Male", "Female"}
        Patient's sex; used as a categorical feature.

    ethnicity : str
        Patient's ethnicity.  Unrecognised values are passed through
        unchanged; the one‑hot encoder in the pipeline will handle
        unseen categories safely.

    cyp2c9, vkorc1, cyp4f2 : str
        Genotype strings for each gene.  They are used both as
        categorical features and to derive numeric risk/sensitivity
        scores.

    on_amiodarone, on_antibiotics, on_aspirin, on_statins : bool
        Indicators of concomitant medications.

    has_hypertension, has_diabetes, has_ckd, has_heart_failure : bool
        Indicators of comorbidities.

    alcohol_intake, smoking_status, diet_vitk : str
        Lifestyle/dietary categorical inputs.

    inr_days : float
        Number of days taken to stabilise the patient’s INR.

    ttr_pct : float
        Percentage of time in therapeutic range.

    adverse_event_flag : bool
        Indicator of whether the patient experienced an adverse event.

    adverse_event : str
        Descriptor of the adverse event (e.g. ``"None"``, ``"Minor"``,
        ``"Major"``).  Unknown values are passed through unchanged.

    comorbidity_score : float, optional
        Precomputed comorbidity score.  If ``None`` it will be computed
        automatically as the sum of chronic conditions (hypertension,
        diabetes, CKD and heart failure).  You can override this
        behaviour by supplying your own numeric value.

    Returns
    -------
    pandas.DataFrame
        A 1‑row DataFrame containing all features in the correct
        order.
    """
    # Convert booleans to 0/1 integers for numeric features.
    def _bool_to_int(b: bool) -> int:
        return 1 if bool(b) else 0

    # Compute BMI if height and weight are provided; height is in cm.
    bmi = np.nan
    if height and height > 0 and weight and weight > 0:
        h_m = float(height) / 100.0
        if h_m > 0:
            bmi = float(weight) / (h_m * h_m)

    # Convert genotypes to numeric encodings.
    cyp2c9_risk = genotype_to_risk(cyp2c9)
    vkorc1_sens = genotype_to_sensitivity(vkorc1)
    cyp4f2_eff = genotype_to_effect(cyp4f2)

    # Fill comorbidity score if not provided: count positive chronic
    # conditions (hypertension, diabetes, CKD, heart failure).
    if comorbidity_score is None:
        comorbidity_score = (
            _bool_to_int(has_hypertension)
            + _bool_to_int(has_diabetes)
            + _bool_to_int(has_ckd)
            + _bool_to_int(has_heart_failure)
        )

    # Derive the amiodarone–CYP2C9 interaction: simply the product of
    # the binary amiodarone flag and the CYP2C9 risk score.
    amio_cyp2c9_inter = _bool_to_int(on_amiodarone) * cyp2c9_risk

    # Construct dictionary for all features.  Defaults are applied to
    # unspecified numeric fields (0.0) and categorical fields
    # ("Unknown").
    row: dict[str, object] = {col: None for col in EXPECTED_COLS}

    # Numeric fields
    row["Age"] = float(age)
    row["Weight_kg"] = float(weight)
    row["Height_cm"] = float(height)
    row["Hypertension"] = _bool_to_int(has_hypertension)
    row["Diabetes"] = _bool_to_int(has_diabetes)
    row["Chronic_Kidney_Disease"] = _bool_to_int(has_ckd)
    row["Heart_Failure"] = _bool_to_int(has_heart_failure)
    row["Amiodarone"] = _bool_to_int(on_amiodarone)
    row["Antibiotics"] = _bool_to_int(on_antibiotics)
    row["Aspirin"] = _bool_to_int(on_aspirin)
    row["Statins"] = _bool_to_int(on_statins)
    row["INR_Stabilization_Days"] = float(inr_days)
    row["Time_in_Therapeutic_Range_Pct"] = float(ttr_pct)
    row["Adverse_Event_Flag"] = _bool_to_int(adverse_event_flag)
    row["CYP2C9_risk"] = float(cyp2c9_risk)
    row["VKORC1_sensitivity"] = float(vkorc1_sens)
    row["CYP4F2_effect"] = float(cyp4f2_eff)
    row["BMI"] = float(bmi) if not np.isnan(bmi) else 0.0
    row["Amiodarone_CYP2C9_interaction"] = float(amio_cyp2c9_inter)
    row["Comorbidity_Score"] = float(comorbidity_score)

    # Categorical fields (convert to strings; unknown values allowed)
    row["CYP2C9"] = str(cyp2c9)
    row["VKORC1"] = str(vkorc1)
    row["CYP4F2"] = str(cyp4f2)
    row["Sex"] = str(sex)
    row["Ethnicity"] = str(ethnicity)
    row["Alcohol_Intake"] = str(alcohol_intake)
    row["Smoking_Status"] = str(smoking_status)
    row["Diet_VitK_Intake"] = str(diet_vitk)
    row["Adverse_Event"] = str(adverse_event)

    # Fill any remaining missing numeric fields with 0 and categorical
    # with "Unknown".  Although the above assignment covers all
    # expected columns, this guard ensures future schema changes
    # don’t produce NaNs.
    for col in EXPECTED_COLS:
        if row[col] is None or (isinstance(row[col], float) and np.isnan(row[col])):
            if col in {
                "Age",
                "Weight_kg",
                "Height_cm",
                "Hypertension",
                "Diabetes",
                "Chronic_Kidney_Disease",
                "Heart_Failure",
                "Amiodarone",
                "Antibiotics",
                "Aspirin",
                "Statins",
                "INR_Stabilization_Days",
                "Time_in_Therapeutic_Range_Pct",
                "Adverse_Event_Flag",
                "CYP2C9_risk",
                "VKORC1_sensitivity",
                "CYP4F2_effect",
                "BMI",
                "Amiodarone_CYP2C9_interaction",
                "Comorbidity_Score",
            }:
                row[col] = 0.0
            else:
                row[col] = "Unknown"

    # Create DataFrame in correct column order
    df = pd.DataFrame([row], columns=EXPECTED_COLS)
    return df


def predict_dose(**inputs) -> str:
    """Wrapper used by Gradio to build the feature row and return
    the model’s prediction as a human readable string.

    Parameters
    ----------
    **inputs : dict
        Named parameters corresponding to the UI components defined in
        the Gradio interface.

    Returns
    -------
    str
        Formatted recommended dose in milligrams per day.
    """
    df = build_feature_row(**inputs)
    # The model returns a 1‑element numpy array.  Cast to float for
    # formatting.
    pred = float(model.predict(df)[0])
    return f"Recommended warfarin dose: {pred:.2f} mg/day"


# ---------------------------------------------------------------------------
# Gradio interface definition
# ---------------------------------------------------------------------------
def create_demo() -> "gr.Blocks":  # type: ignore[return-type]
    """Create the Gradio Blocks demo.  This function defers the import
    of Gradio until it is actually needed, avoiding import errors when
    gradio is not installed (e.g. during offline testing).
    """
    if gr is None:
        raise RuntimeError(
            "gradio is not available.  Install gradio to launch the web interface."
        )
    # Define dropdown options.  These lists are a minimal set of possible
    # values; the user may choose "Other" to input unseen categories.
    sex_options = ["Male", "Female", "Other"]
    ethnicity_options = [
        "Caucasian",
        "Asian",
        "African",
        "Hispanic",
        "Other",
    ]
    cyp2c9_options = ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3", "Other"]
    vkorc1_options = ["G/G", "A/G", "A/A", "Other"]
    cyp4f2_options = ["Normal", "Increased", "Other"]
    alcohol_options = ["None", "Light", "Moderate", "Heavy"]
    smoking_options = ["Never", "Former", "Current"]
    vitk_options = ["Low", "Moderate", "High"]
    adverse_event_options = ["None", "Minor", "Major", "Unknown"]

    with gr.Blocks() as demo:
        gr.Markdown("# Warfarin Dose Recommendation")
        gr.Markdown(
            "Provide the following patient characteristics to estimate an "
            "appropriate warfarin dose.  Fields not supplied will be "
            "substituted with default values."
        )
        with gr.Column():
            # Demographics
            # Use numeric inputs rather than sliders for age, weight and height
            age = gr.Number(value=60, label="Age (years)")
            weight = gr.Number(value=75, label="Weight (kg)")
            height = gr.Number(value=170, label="Height (cm)")
            sex = gr.Dropdown(sex_options, value="Male", label="Sex")
            ethnicity = gr.Dropdown(ethnicity_options, value="Caucasian", label="Ethnicity")
            # Genotype information
            cyp2c9 = gr.Dropdown(cyp2c9_options, value="*1/*1", label="CYP2C9 genotype")
            vkorc1 = gr.Dropdown(vkorc1_options, value="G/G", label="VKORC1 genotype")
            cyp4f2 = gr.Dropdown(cyp4f2_options, value="Normal", label="CYP4F2 genotype")
            # Comorbidities (checkboxes)
            has_hypertension = gr.Checkbox(label="Hypertension", value=False)
            has_diabetes = gr.Checkbox(label="Diabetes", value=False)
            has_ckd = gr.Checkbox(label="Chronic kidney disease", value=False)
            has_heart_failure = gr.Checkbox(label="Heart failure", value=False)
            # Medications
            on_amiodarone = gr.Checkbox(label="On Amiodarone", value=False)
            on_antibiotics = gr.Checkbox(label="On interacting antibiotics", value=False)
            on_aspirin = gr.Checkbox(label="On Aspirin", value=False)
            on_statins = gr.Checkbox(label="On Statins", value=False)
            # Lifestyle
            alcohol_intake = gr.Dropdown(alcohol_options, value="None", label="Alcohol intake")
            smoking_status = gr.Dropdown(smoking_options, value="Never", label="Smoking status")
            # Dietary vitamin K intake, INR stabilisation days, time in therapeutic range,
            # detailed adverse event information and explicit comorbidity score are removed
            # at the user's request; sensible defaults will be used internally.
            adverse_event_flag = gr.Checkbox(label="Experienced adverse event", value=False)

            submit = gr.Button("Predict dose")
            output = gr.Textbox(label="Recommended dose (mg/day)")

        def on_submit(
            age,
            weight,
            height,
            sex,
            ethnicity,
            cyp2c9,
            vkorc1,
            cyp4f2,
            has_hypertension,
            has_diabetes,
            has_ckd,
            has_heart_failure,
            on_amiodarone,
            on_antibiotics,
            on_aspirin,
            on_statins,
            alcohol_intake,
            smoking_status,
            adverse_event_flag,
        ):
            # Pass only the required arguments to predict_dose.  Removed fields
            # (e.g. INR days, vitamin K intake, time in range, adverse event type
            # and comorbidity score) will take default values within the
            # prediction pipeline.
            return predict_dose(
                age=age,
                weight=weight,
                height=height,
                sex=sex,
                ethnicity=ethnicity,
                cyp2c9=cyp2c9,
                vkorc1=vkorc1,
                cyp4f2=cyp4f2,
                on_amiodarone=on_amiodarone,
                on_antibiotics=on_antibiotics,
                on_aspirin=on_aspirin,
                on_statins=on_statins,
                has_hypertension=has_hypertension,
                has_diabetes=has_diabetes,
                has_ckd=has_ckd,
                has_heart_failure=has_heart_failure,
                alcohol_intake=alcohol_intake,
                smoking_status=smoking_status,
                adverse_event_flag=adverse_event_flag,
            )

        submit.click(
            on_submit,
            inputs=[
                age,
                weight,
                height,
                sex,
                ethnicity,
                cyp2c9,
                vkorc1,
                cyp4f2,
                has_hypertension,
                has_diabetes,
                has_ckd,
                has_heart_failure,
                on_amiodarone,
                on_antibiotics,
                on_aspirin,
                on_statins,
                alcohol_intake,
                smoking_status,
                adverse_event_flag,
            ],
            outputs=output,
        )

    return demo


if __name__ == "__main__":
    # For local debugging.  Only attempt to launch the UI if gradio
    # imported successfully.  If gradio is missing, print a helpful
    # message instead of raising an exception at import time.
    if gr is None:
        print(
            "gradio is not installed; unable to launch the web interface. "
            "Install gradio and run this script again to test the app locally."
        )
    else:
        create_demo().launch()
