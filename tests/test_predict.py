import pytest
from pydantic import ValidationError
from src.predict import predict_loan


class DummyModel:
    def predict(self, X):
        return ["Low Risk"]


def test_prediction_fails_on_invalid_schema():
    bad_input = {
        "age": 10,                 # ❌ invalid
        "income": 50000,
        "loan_amount": 100000,
        "credit_score": 700,
        "employment_years": 5
    }

    model = DummyModel()

    with pytest.raises(ValidationError):
        predict_loan(bad_input, model)
