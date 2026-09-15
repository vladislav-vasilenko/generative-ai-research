"""Проверки численных методов по независимым аналитическим формулам."""

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent / '.cache/matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import nbformat
import pytest
import torch


@pytest.fixture(scope='module')
def lab():
    notebook = nbformat.read(ROOT / 'lab_one_ru.ipynb', as_version=4)
    namespace = {}
    for cell in notebook.cells:
        if cell.cell_type == 'code' and 'definitions' in cell.metadata.get('tags', []):
            exec(compile(cell.source, cell.id, 'exec'), namespace)
    return namespace


def test_translation_covers_every_original_cell():
    original_path = ROOT / 'references/lab_one_original.ipynb'
    original = json.loads(original_path.read_text())
    for name in ['lab_one_ru.ipynb', 'lab_one_ru_exercises.ipynb']:
        nb = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(nb)
        mapped = [c for c in nb.cells if 'original_cell_index' in c.metadata]
        assert [c.metadata.original_cell_index for c in mapped] == list(range(49))
        assert [c.cell_type for c in mapped] == [c['cell_type'] for c in original['cells']]
        assert nb.metadata.lab_translation.original_sha256 == hashlib.sha256(original_path.read_bytes()).hexdigest()
        assert len({c.id for c in nb.cells}) == len(nb.cells)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                compile(cell.source, cell.id, 'exec')
    exercise = nbformat.read(ROOT / 'lab_one_ru_exercises.ipynb', as_version=4)
    assert sum(c.source.count('raise NotImplementedError') for c in exercise.cells) == 8
    assert not any(c.metadata.get('role') == 'solution' for c in exercise.cells)
    solved = nbformat.read(ROOT / 'lab_one_ru.ipynb', as_version=4)
    assert not any('raise NotImplementedError' in c.source for c in solved.cells)


def test_euler_converges_to_exact_exponential(lab):
    class Decay(lab['ODE']):
        def drift_coefficient(self, x, t):
            return -x
    errors = []
    for n in [11, 21, 41]:
        final = lab['EulerSimulator'](Decay()).simulate(torch.tensor([[2.0]]), torch.linspace(0, 1, n))
        errors.append(abs(final.item() - 2 * math.exp(-1)))
    assert errors[0] > errors[1] > errors[2]
    assert 1.8 < errors[0] / errors[1] < 2.2


def test_zero_diffusion_matches_euler_on_nonuniform_grid(lab):
    model = lab['OUProcess'](theta=0.7, sigma=0)
    x0 = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float64)
    original = x0.clone()
    ts = torch.tensor([0., 0.01, 0.15, 0.4], dtype=x0.dtype)
    ode_path = lab['EulerSimulator'](model).simulate_with_trajectory(x0, ts)
    sde_path = lab['EulerMaruyamaSimulator'](model).simulate_with_trajectory(x0, ts)
    assert ode_path.shape == (2, 4, 2)
    assert torch.equal(ode_path, sde_path)
    assert torch.equal(x0, original)
    expected = x0 * torch.prod(1 - 0.7 * (ts[1:] - ts[:-1]))
    torch.testing.assert_close(ode_path[:, -1], expected)


def test_brownian_mean_and_variance(lab):
    torch.manual_seed(7)
    sigma, duration, n = 1.7, 0.8, 40000
    x0 = torch.zeros(n, 2)
    final = lab['EulerMaruyamaSimulator'](lab['BrownianMotion'](sigma)).simulate(
        x0, torch.linspace(0, duration, 21))
    variance = sigma**2 * duration
    assert torch.all(final.mean(0).abs() < 5 * math.sqrt(variance / n))
    assert torch.all((final.var(0) - variance).abs() < 5 * variance * math.sqrt(2 / (n - 1)))
    assert abs(torch.corrcoef(final.T)[0, 1]) < 0.03


def test_ou_finite_time_moments_and_discretization(lab):
    torch.manual_seed(17)
    theta, sigma, duration, steps, n = 0.8, 1.1, 2.0, 400, 25000
    h = duration / steps
    final = lab['EulerMaruyamaSimulator'](lab['OUProcess'](theta, sigma)).simulate(
        torch.full((n, 1), 2.), torch.linspace(0, duration, steps + 1))[:, 0]
    exact_mean = 2 * math.exp(-theta * duration)
    exact_var = sigma**2 / (2 * theta) * (1 - math.exp(-2 * theta * duration))
    # Точная дисперсия именно рекуррентной схемы, полученная из геометрической суммы.
    a = 1 - theta * h
    discrete_var = sigma**2 * h * (1 - a**(2 * steps)) / (1 - a*a)
    assert abs(final.mean().item() - exact_mean) < 0.035
    assert abs(final.var().item() - exact_var) < 0.035
    assert abs(discrete_var - exact_var) < 0.003


def test_gaussian_score_full_covariance_and_no_grad(lab):
    mean = torch.tensor([1., -2.], dtype=torch.float64)
    cov = torch.tensor([[2., 0.6], [0.6, 1.]], dtype=torch.float64)
    density = lab['Gaussian'](mean, cov)
    x = torch.tensor([[3., -1.], [0., 2.]], dtype=torch.float64)
    expected = -torch.linalg.solve(cov, (x - mean).T).T
    with torch.no_grad():
        actual = density.score(x)
    torch.testing.assert_close(actual, expected)


def test_mixture_score_against_responsibility_formula(lab):
    density = lab['GaussianMixture'](
        torch.tensor([[-1., 0.], [2., 1.]], dtype=torch.float64),
        torch.stack([torch.eye(2), 2 * torch.eye(2)]).double(),
        torch.tensor([0.3, 0.7], dtype=torch.float64))
    x = torch.tensor([[0.1, 0.2], [3., -2.]], dtype=torch.float64)
    components = torch.distributions.MultivariateNormal(density.means, density.covs)
    log_probs = components.log_prob(x[:, None]) + density.weights.log()
    responsibilities = log_probs.softmax(dim=1)
    delta = x[:, None] - density.means
    component_scores = -torch.linalg.solve(density.covs, delta.unsqueeze(-1)).squeeze(-1)
    expected = (responsibilities.unsqueeze(-1) * component_scores).sum(dim=1)
    torch.testing.assert_close(density.score(x), expected)


def test_langevin_and_ou_have_identical_paths(lab):
    theta, sigma = 0.7, 1.2
    density = lab['Gaussian'](torch.zeros(2), torch.eye(2) * sigma**2 / (2 * theta))
    x0 = torch.randn(20, 2)
    ts = torch.linspace(0, 1, 31)
    torch.manual_seed(8)
    first = lab['EulerMaruyamaSimulator'](lab['LangevinSDE'](sigma, density)).simulate_with_trajectory(x0, ts)
    torch.manual_seed(8)
    second = lab['EulerMaruyamaSimulator'](lab['OUProcess'](theta, sigma)).simulate_with_trajectory(x0, ts)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('values', [[0., 0.], [1., 0.], [0., float('nan')]])
def test_invalid_time_grid_is_rejected(lab, values):
    with pytest.raises(ValueError):
        lab['EulerMaruyamaSimulator'](lab['BrownianMotion'](1.)).simulate(torch.zeros(2, 1), torch.tensor(values))


def test_single_time_and_recording_endpoints(lab):
    x = torch.ones(3, 1)
    path = lab['EulerMaruyamaSimulator'](lab['BrownianMotion'](1.)).simulate_with_trajectory(x, torch.tensor([0.]))
    assert path.shape == (3, 1, 1)
    assert lab['every_nth_index'](7, 3).tolist() == [0, 3, 6]
    assert lab['every_nth_index'](8, 3).tolist() == [0, 3, 6, 7]
    assert lab['every_nth_index'](1, 4).tolist() == [0]


def test_random_mixture_does_not_reset_global_rng(lab):
    torch.manual_seed(123)
    expected = torch.rand(5)
    torch.manual_seed(123)
    lab['GaussianMixture'].random_2D(3, std=1., seed=4)
    assert torch.equal(torch.rand(5), expected)


def test_optional_animation_without_ffmpeg(lab):
    torch.manual_seed(12)
    density = lab['Gaussian'](torch.zeros(2), torch.eye(2))
    result = lab['animate_dynamics'](40, density,
        lab['EulerMaruyamaSimulator'](lab['LangevinSDE'](1., density)),
        density, torch.linspace(0, 0.1, 4), animate_every=2, bins=15, scale=5)
    assert 'data:image/png;base64' in result.data
