import pytest
from pathlib import Path
from services.rate_calculator import RateCalculator

# Path to the actual rate matrix
RATE_MATRIX_PATH = Path(__file__).parent.parent / "docs" / "rate_matrix.csv"


class TestRateCalculator:
    """Test suite for RateCalculator class."""

    def test_load_matrix_success(self):
        """Test that the rate matrix loads successfully."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)
        assert calculator.rate_matrix is not None
        assert len(calculator.rate_matrix) > 0

    def test_load_matrix_file_not_found(self):
        """Test that FileNotFoundError is raised when matrix file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            RateCalculator(matrix_path=Path("nonexistent.csv"))

    def test_calculate_exact_match(self):
        """Test rate calculation with exact match from matrix."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # From CSV: 760,60,36,30,6.500
        rate = calculator.calculate(
            credit_score=760,
            ltv=60,
            dti=36,
            loan_term=30
        )
        assert rate == 6.500

    def test_calculate_credit_score_above_minimum(self):
        """Test that credit score >= min_credit works."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # 780 is >= 760, should match 760,60,36,30,6.500
        rate = calculator.calculate(
            credit_score=780,
            ltv=60,
            dti=36,
            loan_term=30
        )
        assert rate == 6.500

    def test_calculate_ltv_at_boundary(self):
        """Test LTV at exact boundary."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # LTV at max_ltv (60) should match
        rate = calculator.calculate(
            credit_score=760,
            ltv=60,
            dti=36,
            loan_term=30
        )
        assert rate == 6.500

    def test_calculate_ltv_above_boundary_no_match(self):
        """Test that LTV > max_ltv returns None."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # LTV 61 > 60, should not match
        rate = calculator.calculate(
            credit_score=760,
            ltv=61,
            dti=36,
            loan_term=30
        )
        # Should either return None or find a higher LTV bracket
        assert rate is None or rate != 6.500

    def test_calculate_dti_at_boundary(self):
        """Test DTI at exact boundary."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # DTI at max_dti (36) should match
        rate = calculator.calculate(
            credit_score=760,
            ltv=60,
            dti=36,
            loan_term=30
        )
        assert rate == 6.500

    def test_calculate_15_year_term(self):
        """Test calculation for 15-year loan term."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # From CSV: 760,60,36,15,5.750
        rate = calculator.calculate(
            credit_score=760,
            ltv=60,
            dti=36,
            loan_term=15
        )
        assert rate == 5.750

    def test_calculate_30_year_term(self):
        """Test calculation for 30-year loan term."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # From CSV: 760,60,36,30,6.500
        rate = calculator.calculate(
            credit_score=760,
            ltv=60,
            dti=36,
            loan_term=30
        )
        assert rate == 6.500

    def test_calculate_low_credit_score(self):
        """Test calculation with low credit score."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # From CSV: 600,60,36,30,7.750
        rate = calculator.calculate(
            credit_score=600,
            ltv=60,
            dti=36,
            loan_term=30
        )
        assert rate == 7.750

    def test_calculate_no_match_returns_none(self):
        """Test that no match returns None."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # Credit score below minimum (below 600)
        rate = calculator.calculate(
            credit_score=500,
            ltv=60,
            dti=36,
            loan_term=30
        )
        assert rate is None

    def test_calculate_invalid_loan_term(self):
        """Test calculation with invalid loan term (not 15 or 30)."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # 20-year term doesn't exist in matrix
        rate = calculator.calculate(
            credit_score=760,
            ltv=60,
            dti=36,
            loan_term=20
        )
        assert rate is None

    def test_calculate_high_ltv(self):
        """Test calculation with high LTV ratio."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # From CSV: 760,95,36,30,7.250
        rate = calculator.calculate(
            credit_score=760,
            ltv=95,
            dti=36,
            loan_term=30
        )
        assert rate == 7.250

    def test_calculate_high_dti(self):
        """Test calculation with high DTI ratio."""
        calculator = RateCalculator(matrix_path=RATE_MATRIX_PATH)

        # From CSV: 760,60,50,30,6.875
        rate = calculator.calculate(
            credit_score=760,
            ltv=60,
            dti=50,
            loan_term=30
        )
        assert rate == 6.875
