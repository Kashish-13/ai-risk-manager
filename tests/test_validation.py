import numpy as np
import pandas as pd

from src.validation import validate_dataframe


def test_validation_reports_quality_without_modifying_data() -> None:
    source = pd.DataFrame({"Amount": [10.0, np.inf, 10.0], "Class": [0, 1, 0]})
    original = source.copy(deep=True)
    report = validate_dataframe(source, "Class")
    assert not report.is_valid
    assert report.invalid_numeric_values == {"Amount": 1}
    assert report.duplicate_rows == 1
    pd.testing.assert_frame_equal(source, original)


def test_validation_requires_target() -> None:
    report = validate_dataframe(pd.DataFrame({"Amount": [1.0, 2.0]}), "Class")
    assert not report.is_valid
    assert "missing" in report.errors[0].lower()
