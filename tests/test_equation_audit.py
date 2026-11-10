import pytest

from llsa.secondary import PierOperators, WaveSolver
from llsa.secondary.equation_audit import (independent_laminar_check, isolated_coefficient_checks,
                                          floquet_full_ns_check, primary_equation_defects)


@pytest.fixture(scope='module')
def wave():
    solver = WaveSolver(PierOperators(n=24, mean_mode='parallel'), 1)
    return solver.solve(solver.bootstrap(), frequency=50.5)


def test_direct_von_karman_bvp_agrees_with_bek_conversion():
    assert independent_laminar_check()['passed']


def test_isolated_ns_coefficients_and_metric_terms():
    result = isolated_coefficient_checks()
    assert result['cases'] == 124
    assert result['passed']


def test_differentiating_full_ns_recovers_floquet_pencil(wave):
    assert floquet_full_ns_check(wave)['passed']


def test_parallel_mean_is_not_reported_as_full_paper_continuity(wave):
    from llsa.secondary.validation import compare_tables
    defects = primary_equation_defects(wave)
    assert max(defects['interior_momentum_by_component']) < 1e-5
    assert defects['nonzero_interior_continuity'] < 1e-8
    assert defects['mean_continuity'] == pytest.approx(defects['mean_u_over_R'], abs=1e-10)
    assert defects['mean_continuity'] > 1e-3
    assert not defects['frozen_paper_mean_continuity_passed']
    assert compare_tables(wave)['paper_mean_constraints']['known_mismatch']


def test_validation_rejects_table_agreement_when_mean_equations_fail(wave, tmp_path, monkeypatch):
    from llsa.secondary import cli
    original = cli.compare_tables
    def hypothetical_table_agreement(*args, **kwargs):
        result = original(*args, **kwargs)
        result['table2']['passed'] = True
        if 'table3' in result:
            result['table3']['passed'] = True
        return result
    monkeypatch.setattr(cli, 'compare_tables', hypothetical_table_agreement)
    source = tmp_path/'wave.npz'
    wave.save(source)
    code = cli.main(['pier-validate', '--wave', str(source), '--secondary-wall', 'continuity',
                     '--secondary-harmonics', '1', '--output', str(tmp_path/'validation')])
    assert code == 2
