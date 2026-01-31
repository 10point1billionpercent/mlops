from backup.schemas import LoanInput

def predict_loan(input_data, model):
    # Validate input
    validated = LoanInput(**input_data)

    # If validation passes, prediction happens
    X = [[
        validated.age,
        validated.income,
        validated.loan_amount,
        validated.credit_score,
        validated.employment_years
    ]]

    return model.predict(X)
