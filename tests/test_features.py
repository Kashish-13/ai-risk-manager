import pandas as pd

from src.features import engineer_features


def test_ulb_features_are_added_without_removing_source_columns() -> None:
    source = pd.DataFrame({"Time": [0.0, 3_600.0], "Amount": [0.0, 99.0], "Class": [0, 1]})
    result = engineer_features(source)
    assert {"Time", "Amount", "Class"}.issubset(result.columns)
    assert {"Time_hour", "Time_hour_sin", "Time_hour_cos", "Time_day_index"}.issubset(result.columns)
    assert {"Amount_log1p", "Amount_sqrt", "Amount_is_zero", "Amount_squared"}.issubset(result.columns)
    assert result.loc[0, "Amount_is_zero"] == 1
    assert source.columns.tolist() == ["Time", "Amount", "Class"]


def test_unsupported_risk_fields_are_not_invented() -> None:
    result = engineer_features(pd.DataFrame({"Amount": [10.0]}))
    assert "device_risk" not in result.columns
    assert "location_risk" not in result.columns
    assert "merchant_risk" not in result.columns

