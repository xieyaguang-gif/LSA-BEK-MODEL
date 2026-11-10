"""Check the physical inlet constraints for increasing/decreasing continuation."""
import numpy as np
import pytest

from llsa.secondary.inlet_amplitude import InletAmplitudePath


@pytest.mark.parametrize('ratio', [9.828212863049744, .1])
def test_exact_endpoints_and_velocity_scaling_ignore_pressure(ratio):
    rng = np.random.default_rng(341)
    source = rng.normal(size=(5, 16))+1j*rng.normal(size=(5, 16))
    source = .5*(source+source[::-1].conj())
    rows = np.array([1, 2, 5, 6, 9, 10])
    target = source.copy()
    target[3, rows] *= ratio
    target[1] = target[3].conj()
    target[:, 12:] *= 17  # Pressure is not a prescribed inlet velocity.
    path = InletAmplitudePath.fit(source, target, 3, rows)
    assert path.ratio == pytest.approx(ratio)
    np.testing.assert_array_equal(path.at(0), source)
    np.testing.assert_array_equal(path.at(path.end), target)
    coordinate = 0.
    for _ in range(50):
        field = path.at(coordinate)
        np.testing.assert_allclose(field[3, rows], np.exp(coordinate)*source[3, rows])
        np.testing.assert_allclose(field, field[::-1].conj())
        if coordinate == path.end:
            break
        coordinate = path.advance(coordinate, .3)
    assert coordinate == path.end
    assert path.nominal_af(path.end, .1, 1.) == pytest.approx(1.)


def test_nominal_tenfold_ratio_cannot_replace_calibrated_inlet():
    source = np.ones((3, 8), complex)
    target = source*9.828212863049744
    with pytest.raises(ValueError, match='mismatch'):
        InletAmplitudePath.fit(source, target, 2, [1, 2, 3], expected_ratio=10.)


@pytest.mark.parametrize('change', ['phase', 'profile'])
def test_refuse_different_inlet_shape_or_phase(change):
    source = np.ones((3, 8), complex)
    target = 2*source
    if change == 'phase':
        target *= np.exp(.1j)
    else:
        target[2, 1] *= 1.1
    with pytest.raises(ValueError, match='mismatch'):
        InletAmplitudePath.fit(source, target, 2, [1, 2, 3])
