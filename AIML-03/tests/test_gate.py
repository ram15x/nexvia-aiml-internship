from rec.gate import passes


def test_gate_promotes_equal_or_better_and_blocks_regressions():
    prod = {"recall@100": 0.2323}
    assert passes({"recall@100": 0.2400}, prod)[0] and passes({"recall@100": 0.2323}, prod)[0]
    assert passes({"recall@100": 0.2290}, prod)[0]                  # within the 2% tolerance
    ok, msg = passes({"recall@100": 0.2000}, prod); assert not ok and "BLOCK" in msg
