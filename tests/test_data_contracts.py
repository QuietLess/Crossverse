import pandas as pd
import pytest

from crossverse.data.schema import DataContractError, validate_interactions, validate_items


def test_fixture_satisfies_contracts(fixture_data):
    interactions, items = fixture_data
    validate_items(items)
    validate_interactions(interactions, items)


def _base():
    return pd.DataFrame(
        {"user_id": ["u1", "u1"], "item_id": ["m_1", "g_1"], "domain": ["movie", "game"],
         "rating": [5.0, 3.0], "timestamp": [1, 2]}
    )


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda d: d.drop(columns="rating"), "missing"),
        (lambda d: d.assign(rating=[6.0, 3.0]), "ratings"),
        (lambda d: d.assign(domain=["movie", "book"]), "domain"),
        (lambda d: d.assign(timestamp=[0, 2]), "timestamps"),
        (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
        (lambda d: d.assign(item_id=[None, "g_1"]), "nulls"),
    ],
)
def test_interaction_contract_violations(mutate, msg):
    with pytest.raises(DataContractError, match=msg):
        validate_interactions(mutate(_base()))


def test_unknown_item_reference(fixture_data):
    _, items = fixture_data
    df = _base().assign(item_id=["nope", "nope2"])
    with pytest.raises(DataContractError, match="unknown items"):
        validate_interactions(df, items)


def test_item_contract(fixture_data):
    _, items = fixture_data
    with pytest.raises(DataContractError, match="duplicate"):
        validate_items(pd.concat([items, items.iloc[[0]]]))
