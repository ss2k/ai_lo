import csv
from pathlib import Path
from typing import Optional

class RateCalculator:
    """
    A service to calculate loan interest rates based on a rate matrix CSV.
    """
    def __init__(self, matrix_path: Path):
        """
        Initializes the RateCalculator with the path to the rate matrix.

        Args:
            matrix_path: The path to the rate_matrix.csv file.
        """
        self.matrix_path = matrix_path
        self.rate_matrix = self._load_matrix()

    def _load_matrix(self) -> list[dict]:
        """Loads the rate matrix from the CSV file into memory."""
        matrix = []
        if not self.matrix_path.exists():
            raise FileNotFoundError(f"Rate matrix not found at: {self.matrix_path}")

        with open(self.matrix_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    matrix.append({
                        "min_credit": int(row["min_credit"]),
                        "max_ltv": float(row["max_ltv"]),
                        "max_dti": float(row["max_dti"]),
                        "loan_term": int(row["loan_term"]),
                        "rate": float(row["rate"]),
                    })
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping invalid row in rate matrix: {row}. Error: {e}")
        return matrix

    def calculate(self, credit_score: int, ltv: float, dti: float, loan_term: int) -> Optional[float]:
        """
        Calculates the interest rate based on user's financial data.

        Args:
            credit_score: The user's credit score.
            ltv: The loan-to-value ratio.
            dti: The debt-to-income ratio.
            loan_term: The loan term in years (15 or 30).

        Returns:
            The calculated interest rate as a float, or None if no matching rate is found.
        """
        if not self.rate_matrix:
            return None

        # Find matching rate based on criteria
        for row in self.rate_matrix:
            if (credit_score >= row["min_credit"] and
                ltv <= row["max_ltv"] and
                dti <= row["max_dti"] and
                loan_term == row["loan_term"]):
                return row["rate"]

        return None
