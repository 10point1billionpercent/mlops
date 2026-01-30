from pydantic import ValidationError
from src.schemas import LoanInput


def test_valid_input():
    data = {
        "age": 30,
        "income": 50000,
        "loan_amount": 200000,
        "credit_score": 750,
        "employment_years": 5
    }
    loan = LoanInput(**data)
    assert loan.age == 30


def test_invalid_age():
    data = {
        "age": 15,
        "income": 50000,
        "loan_amount": 200000,
        "credit_score": 750,
        "employment_years": 5
    }
    try:
        LoanInput(**data)
        assert False
    except ValidationError:
        assert True


def test_negative_income():
    data = {
        "age": 30,
        "income": -1000,
        "loan_amount": 200000,
        "credit_score": 750,
        "employment_years": 5
    }
    try:
        LoanInput(**data)
        assert False
    except ValidationError:
        assert True


def test_missing_field():
    data = {
        "age": 30,
        "income": 50000,
        "credit_score": 700,
        "employment_years": 5
    }
    try:
        LoanInput(**data)
        assert False
    except ValidationError:
        assert True


def test_wrong_type():
    data = {
        "age": "thirty",
        "income": 50000,
        "loan_amount": 100000,
        "credit_score": 700,
        "employment_years": 5
    }
    try:
        LoanInput(**data)
        assert False
    except ValidationError:
        assert True


def test_boundary_values():
    data = {
        "age": 18,
        "income": 1,
        "loan_amount": 1,
        "credit_score": 300,
        "employment_years": 0
    }
    loan = LoanInput(**data)
    assert loan.credit_score == 300
