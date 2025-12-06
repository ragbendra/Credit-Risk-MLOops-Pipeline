"""
Pydantic schemas for request/response validation.

All input validation is handled here using Pydantic Field constraints.
This is the ONLY validation layer - no Great Expectations needed.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CreditRequest(BaseModel):
    """
    Credit application request schema.
    
    All fields are validated with appropriate constraints.
    The dataset has 20 features, all encoded as integers after preprocessing.
    """
    
    # Account status (checking account status)
    status: int = Field(
        ge=0, le=4,
        description="Checking account status (0-4 encoded)"
    )
    
    # Loan details
    duration: int = Field(
        ge=1, le=72,
        description="Loan duration in months (1-72)"
    )
    amount: float = Field(
        gt=0, le=100000,
        description="Credit amount (must be positive)"
    )
    
    # Credit history
    credit_history: int = Field(
        ge=0, le=4,
        description="Credit history category (0-4 encoded)"
    )
    
    # Purpose of loan
    purpose: int = Field(
        ge=0, le=10,
        description="Loan purpose category (0-10 encoded)"
    )
    
    # Financial status
    savings: int = Field(
        ge=0, le=4,
        description="Savings account category (0-4 encoded)"
    )
    
    # Employment
    employment: int = Field(
        ge=0, le=4,
        description="Employment duration category (0-4 encoded)"
    )
    
    # Installment rate
    installment_rate: int = Field(
        ge=1, le=4,
        description="Installment rate as % of income (1-4)"
    )
    
    # Personal status
    personal_status: int = Field(
        ge=0, le=4,
        description="Personal status and sex (0-4 encoded)"
    )
    
    # Other debtors/guarantors
    other_debtors: int = Field(
        ge=0, le=2,
        description="Other debtors/guarantors (0-2 encoded)"
    )
    
    # Residence duration
    residence: int = Field(
        ge=1, le=4,
        description="Present residence duration (1-4 years)"
    )
    
    # Property
    property: int = Field(
        ge=0, le=3,
        description="Property type (0-3 encoded)"
    )
    
    # Age
    age: int = Field(
        ge=18, le=100,
        description="Applicant age (18-100 years)"
    )
    
    # Other installment plans
    other_plans: int = Field(
        ge=0, le=2,
        description="Other installment plans (0-2 encoded)"
    )
    
    # Housing
    housing: int = Field(
        ge=0, le=2,
        description="Housing type (0-2 encoded)"
    )
    
    # Existing credits
    existing_credits: int = Field(
        ge=1, le=4,
        description="Number of existing credits (1-4)"
    )
    
    # Job
    job: int = Field(
        ge=0, le=3,
        description="Job category (0-3 encoded)"
    )
    
    # Dependents
    dependents: int = Field(
        ge=1, le=2,
        description="Number of dependents (1-2)"
    )
    
    # Telephone
    telephone: int = Field(
        ge=0, le=1,
        description="Has telephone (0=no, 1=yes)"
    )
    
    # Foreign worker
    foreign_worker: int = Field(
        ge=0, le=1,
        description="Is foreign worker (0=no, 1=yes)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": 1,
                    "duration": 24,
                    "amount": 5000,
                    "credit_history": 2,
                    "purpose": 3,
                    "savings": 1,
                    "employment": 2,
                    "installment_rate": 2,
                    "personal_status": 2,
                    "other_debtors": 0,
                    "residence": 3,
                    "property": 1,
                    "age": 35,
                    "other_plans": 0,
                    "housing": 1,
                    "existing_credits": 1,
                    "job": 2,
                    "dependents": 1,
                    "telephone": 1,
                    "foreign_worker": 0,
                }
            ]
        }
    }


class CreditResponse(BaseModel):
    """Credit risk prediction response."""
    
    prediction: str = Field(
        description="Risk prediction: 'good' or 'bad'"
    )
    probability: float = Field(
        ge=0, le=1,
        description="Probability of bad credit risk (0.0-1.0)"
    )
    risk_score: int = Field(
        ge=0, le=100,
        description="Risk score (0-100, higher = riskier)"
    )
    model_version: str = Field(
        description="Version of the model used"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether model is loaded")
    version: str = Field(description="API version")
