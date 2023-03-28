from feature_plot import feature_plot
import altair as alt
import pandas as pd


def test_feature_plot():
    # Helper data used to test the function
    data = {
        "name": ["Winston", "Mark", "Sarah", "Jack", "Mohammed", "Susan"],
        "bday": [1988, 1990, 1967, 1973, 1980, 1990],
        "business_owner": [True, False, True, True, True, True],
        "annual_income": [74000, 49000, 82000, 91000, 97000, 69000],
    }
    # Transform the helper data to a pandas dataframe

    helper_data = pd.DataFrame.from_dict(data)

    # Apply the function
    results = feature_plot(helper_data, "name", "annual_income", "show")

    assert (results.shape) == (6, 2)
    assert (list(results["name"])) == [
        "Mohammed",
        "Jack",
        "Sarah",
        "Winston",
        "Susan",
        "Mark",
    ]
    assert (list(results["annual_income"])) == [
        97000,
        91000,
        82000,
        74000,
        69000,
        49000,
    ]
    assert (list(results.iloc[0])) == ["Mohammed", 97000]
    assert (list(results.iloc[-1])) == ["Mark", 49000]
    assert (type(results.loc[:, "name"])) == pd.core.series.Series
    assert (type(results)) == pd.core.frame.DataFrame

    return
