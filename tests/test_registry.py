import pytest
from shiftprofile.registry import Registry


def test_register_and_get():
    reg = Registry("metric")

    @reg.register("brier")
    def brier_fn(x):
        return x * 2

    assert reg.get("brier")(3) == 6


def test_duplicate_name_raises():
    reg = Registry("metric")

    @reg.register("ece")
    def first(x):
        return x

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("ece")
        def second(x):
            return x


def test_unknown_name_lists_available():
    reg = Registry("metric")

    @reg.register("brier")
    def brier_fn(x):
        return x

    with pytest.raises(KeyError, match="brier"):
        reg.get("nope")


def test_names_are_sorted():
    reg = Registry("metric")
    for name in ["zeta", "alpha", "mu"]:
        reg.register(name)(lambda x: x)
    assert reg.names() == ["alpha", "mu", "zeta"]
