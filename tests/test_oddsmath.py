import math

import pytest

from proplab.odds import oddsmath as om


@pytest.mark.parametrize("a,d", [(150, 2.5), (-120, 1 + 100 / 120), (100, 2.0), (-100, 2.0), (-250, 1.4), (300, 4.0)])
def test_american_decimal_roundtrip(a, d):
    assert om.american_to_decimal(a) == pytest.approx(d)
    back = om.decimal_to_american(d)
    assert back == pytest.approx(a if a != -100 else 100)


def test_invalid_american():
    with pytest.raises(ValueError):
        om.american_to_decimal(50)


def test_standard_minus110_market():
    d = om.american_to_decimal(-110)
    assert om.implied(d) == pytest.approx(110 / 210)
    assert om.overround(d, d) == pytest.approx(220 / 210 - 1)  # 4.76%
    for f in om.NOVIG.values():
        assert f(d, d) == pytest.approx(0.5, abs=1e-9)


def test_novig_methods_sum_to_one_and_order():
    do, du = om.american_to_decimal(-150), om.american_to_decimal(120)
    m = om.novig_multiplicative(do, du)
    p = om.novig_power(do, du)
    s = om.novig_shin(do, du)
    # each is a proper probability, and the under side mirrors it
    for f, v in (("m", m), ("p", p), ("s", s)):
        assert 0 < v < 1
    assert om.novig_multiplicative(du, do) == pytest.approx(1 - m)
    assert om.novig_power(du, do) == pytest.approx(1 - p, abs=1e-9)
    assert om.novig_shin(du, do) == pytest.approx(1 - s, abs=1e-9)
    # power and Shin move margin off the longshot, so the favorite's fair prob is higher
    assert p > m and s > m


def test_expected_value_and_fair_odds():
    assert om.expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert om.expected_value(0.55, om.american_to_decimal(-110)) == pytest.approx(0.55 * 100 / 110 - 0.45)
    assert om.fair_decimal(0.25) == pytest.approx(4.0)


def test_clv_and_log_loss():
    assert om.clv_prob(0.50, 0.55, "over") == pytest.approx(0.05)
    assert om.clv_prob(0.50, 0.55, "under") == pytest.approx(-0.05)
    assert om.log_loss(0.8, 1) == pytest.approx(-math.log(0.8))
    assert om.log_loss(0.8, 0) == pytest.approx(-math.log(0.2))
