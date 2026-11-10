import numpy as np
import pytest

from llsa.secondary import PierOperators, WaveSolver
from llsa.secondary.time_domain import LinearTimeProblem, time_grid, advance, fit_growth


@pytest.fixture(scope='module')
def wave():
    solver = WaveSolver(PierOperators(n=24, mean_mode='parallel'), 1)
    return solver.solve(solver.bootstrap(), frequency=50.5)


def test_mapped_higher_derivatives():
    grid = time_grid(80, 160, 5)
    f = np.exp(-grid.z)
    assert np.max(abs(grid.D@f+f)) < 1e-7
    assert np.max(abs(grid.D2@f-f)) < 1e-5
    assert np.max(abs(grid.D4@f-f)) < 2e-3
    assert abs(f@grid.mass@f-.5) < 1e-9


def test_initial_conditions_are_incompressible(wave):
    problem = LinearTimeProblem(wave, alpha_hat=-.002-.037j, n=24, zmax=120, harmonics=1)
    initial = problem.random_state()
    defects = problem.constraints_defect(initial)
    assert defects['continuity'] < 1e-12
    assert defects['boundary_velocity'] < 1e-12


def test_phase_padding_does_not_change_rhs(wave):
    small = LinearTimeProblem(wave, n=24, zmax=120, harmonics=1, phase_points=5)
    padded = LinearTimeProblem(wave, n=24, zmax=120, harmonics=1, phase_points=15)
    initial = small.random_state(amplitude=1)
    np.testing.assert_allclose(small.explicit_rhs(initial), padded.explicit_rhs(initial),
                               rtol=1e-10, atol=1e-9)


def test_viscosity_damps_instead_of_creating_spurious_growth(wave):
    problem = LinearTimeProblem(wave, n=32, zmax=160, harmonics=1)
    initial = problem.random_state()
    problem.explicit_rhs = lambda q: np.zeros((len(problem.ns), 2*problem.m), complex)
    trace = advance(problem, initial, dt=.001, end_time=.2, sample_every=10)
    assert np.max(trace['energy']) <= trace['energy'][0]*(1+1e-8)
    assert trace['energy'][-1] < .95*trace['energy'][0]
    assert trace['constraints']['boundary_velocity'] < 1e-10


def test_time_order_and_energy_growth_factor(wave):
    problem = LinearTimeProblem(wave, n=16, zmax=80, harmonics=1)
    initial = problem.random_state()
    problem.diffusion[:] = 0
    lam = .3-2j
    problem.explicit_rhs = lambda q: lam*np.einsum(
        'hij,hj->hi', problem.mass, q.reshape(len(problem.ns), -1))
    errors = []
    for dt in [.02, .01, .005]:
        trace = advance(problem, initial, dt=dt, end_time=1, sample_every=1)
        errors.append(np.linalg.norm(trace['final_state']-initial*np.exp(lam))/np.linalg.norm(initial))
    assert errors[0]/errors[1] > 3.7
    assert errors[1]/errors[2] > 3.7
    fit = fit_growth(trace, .5)
    assert abs(fit['growth']-.3) < 1e-4
    assert abs(fit['omega_real']-2) < 1e-4


def test_linear_amplitude_scaling(wave):
    problem = LinearTimeProblem(wave, n=24, zmax=120, harmonics=1)
    initial = problem.random_state()
    low = advance(problem, initial, dt=.001, end_time=.1)
    high = advance(problem, initial*100, dt=.001, end_time=.1)
    np.testing.assert_allclose(high['energy']/low['energy'], 1e4, rtol=1e-8)
