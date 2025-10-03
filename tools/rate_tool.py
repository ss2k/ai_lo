from langchain.tools import Tool
from pathlib import Path
from services.rate_calculator import RateCalculator
from typing import Dict, Any

def calculate_mortgage_rate(input_data: str) -> str:
    try:
        # Parse input
        parts = input_data.strip().split(',')
        if len(parts) != 4:
            return "Error: Invalid input format. Expected: credit_score,ltv,dti,loan_term"

        credit_score = int(parts[0])
        ltv = float(parts[1])
        dti = float(parts[2])
        loan_term = int(parts[3])

        # Calculate rate
        rate_matrix_path = Path(__file__).parent.parent / "docs" / "rate_matrix.csv"
        calculator = RateCalculator(matrix_path=rate_matrix_path)
        rate = calculator.calculate(
            credit_score=credit_score,
            ltv=ltv,
            dti=dti,
            loan_term=loan_term
        )

        if rate:
            return f"The estimated interest rate is {rate:.3f}%"
        else:
            return "Unfortunately, no matching rate was found for the provided criteria. The LTV or DTI may be outside our lending guidelines."

    except Exception as e:
        return f"Error calculating rate: {str(e)}"

rate_calculation_tool = Tool(
    name="CalculateMortgageRate",
    func=calculate_mortgage_rate,
    description="""Calculate mortgage interest rate based on applicant's financial data.
    Input format: 'credit_score,ltv,dti,loan_term'
    - credit_score: applicant's credit score (e.g., 760)
    - ltv: loan-to-value ratio as percentage (e.g., 80 for 80%)
    - dti: debt-to-income ratio as percentage (e.g., 43 for 43%)
    - loan_term: loan term in years, either 15 or 30

    Example: '760,80,43,30' for credit score 760, 80% LTV, 43% DTI, 30-year term"""
)
