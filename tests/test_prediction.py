"""
test_prediction.py
------------------
Unit tests for the patient risk prediction pipeline, clinical input validation,
key order invariance, batch inference, and medical safety disclaimers.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.prediction import (
    FEATURE_ORDER,
    MEDICAL_DISCLAIMER,
    get_default_model_path,
    load_model,
    predict_batch,
    predict_risk,
    validate_patient_input,
)


@pytest.fixture
def low_risk_patient():
    """A realistic clinical profile typically corresponding to low cardiac risk."""
    return {
        "age": 42,
        "sex": 0,
        "cp": 1,  # Atypical angina
        "trestbps": 115,
        "chol": 190,
        "fbs": 0,
        "restecg": 0,
        "thalach": 175,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 0,
        "ca": 0,
        "thal": 1,  # Normal
    }


@pytest.fixture
def high_risk_patient():
    """A realistic clinical profile typically corresponding to elevated cardiac risk."""
    return {
        "age": 62,
        "sex": 1,
        "cp": 3,  # Asymptomatic
        "trestbps": 155,
        "chol": 290,
        "fbs": 1,
        "restecg": 2,
        "thalach": 115,
        "exang": 1,  # Exercise angina
        "oldpeak": 3.0,
        "slope": 1,
        "ca": 2,  # 2 vessels
        "thal": 3,  # Reversible defect
    }


def test_model_file_exists_and_loads():
    """Test that the serialized final pipeline file exists and loads."""
    model_path = get_default_model_path()
    assert model_path.exists(), f"Final model file not found at {model_path}"
    model = load_model(model_path)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_predict_risk_structure_and_disclaimer(low_risk_patient):
    """Test output schema and presence of required educational disclaimer."""
    result = predict_risk(low_risk_patient)

    expected_keys = [
        "prediction",
        "predicted_category",
        "risk_level",
        "heart_disease_probability",
        "heart_disease_percentage",
        "no_disease_probability",
        "confidence_score",
        "contributing_clinical_factors",
        "input_features",
        "medical_disclaimer",
    ]
    for key in expected_keys:
        assert key in result

    assert result["prediction"] in [0, 1]
    assert 0.0 <= result["heart_disease_probability"] <= 1.0
    assert result["medical_disclaimer"] == MEDICAL_DISCLAIMER


def test_predict_risk_classification(low_risk_patient, high_risk_patient):
    """Test that low and high risk profiles yield expected comparative probabilities."""
    low_res = predict_risk(low_risk_patient)
    high_res = predict_risk(high_risk_patient)

    # High risk profile probability should exceed low risk profile probability
    assert high_res["heart_disease_probability"] > low_res["heart_disease_probability"]
    assert high_res["prediction"] == 1
    assert "Higher Predicted Risk" in high_res["predicted_category"]


def test_validate_patient_input_missing_field(low_risk_patient):
    """Test that missing required clinical fields raise descriptive ValueError."""
    incomplete_input = low_risk_patient.copy()
    del incomplete_input["age"]

    with pytest.raises(ValueError, match="Missing required clinical fields"):
        validate_patient_input(incomplete_input)


def test_validate_patient_input_out_of_range(low_risk_patient):
    """Test that continuous features outside supported bounds raise ValueError."""
    invalid_input = low_risk_patient.copy()
    invalid_input["trestbps"] = 350.0  # Above max 220 limit

    with pytest.raises(ValueError, match="exceeds the supported model validation limit"):
        validate_patient_input(invalid_input)


def test_validate_patient_input_invalid_category(low_risk_patient):
    """Test that invalid categorical codes raise descriptive ValueError."""
    invalid_input = low_risk_patient.copy()
    invalid_input["cp"] = 9  # Invalid chest pain category

    with pytest.raises(ValueError, match="is invalid. Supported values"):
        validate_patient_input(invalid_input)


def test_key_order_invariance(low_risk_patient):
    """Test that arbitrary dictionary key order produces identical prediction."""
    # Reverse key order
    reversed_input = {k: low_risk_patient[k] for k in reversed(list(low_risk_patient.keys()))}

    res_standard = predict_risk(low_risk_patient)
    res_reversed = predict_risk(reversed_input)

    assert res_standard["prediction"] == res_reversed["prediction"]
    assert res_standard["heart_disease_probability"] == res_reversed["heart_disease_probability"]


def test_predict_batch(low_risk_patient, high_risk_patient):
    """Test batch prediction across multiple patient rows."""
    batch_df = pd.DataFrame([low_risk_patient, high_risk_patient])
    output_df = predict_batch(batch_df)

    assert len(output_df) == 2
    assert "predicted_risk_class" in output_df.columns
    assert "heart_disease_probability" in output_df.columns
    assert output_df.loc[0, "predicted_risk_class"] in [0, 1]
    assert output_df.loc[1, "predicted_risk_class"] in [0, 1]
