from pydantic import BaseModel, Field

class LoanInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=900)
    employment_years: int = Field(..., ge=0, le=50)
