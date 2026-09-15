"""DDPM: проверяем формулы и обучение на синтетических данных без скачивания CIFAR-10."""

import ast
import json
import math
from pathlib import Path

import nbformat
import pytest
import torch
from torch import nn
from torch.nn import functional as F

NOTEBOOK = Path(__file__).resolve().parents[1] / 'Diffusion_DDPM_Research.ipynb'


@pytest.fixture(params=['cpu', 'cuda'])
def ddpm(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA недоступна в проверяемом окружении.')
    device = torch.device(request.param)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    scope = {'torch':torch, 'nn':nn, 'F':F, 'device':device}
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if cell['metadata']['original_cell_index'] == 2:
            exec(source, scope)
        # Загружаем функции и классы, пропуская обучение, чтение весов и датасет.
        tree = ast.parse(source)
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        if definitions:
            exec(compile(ast.Module(body=definitions, type_ignores=[]), '<ddpm>', 'exec'), scope)
    return scope


def test_notebook_schema_and_python_syntax():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            compile(cell.source, cell.id, 'exec')
    assert len([c for c in nb.cells if c.cell_type == 'markdown']) >= 15


def test_forward_uses_correct_per_image_coefficients(ddpm):
    device = ddpm['device']
    x0 = torch.randn(3, 3, 4, 4, device=device)
    t = torch.tensor([0, 100, 999], device=device)
    fixed_noise = torch.randn_like(x0)
    actual = ddpm['q_sample'](x0, t, fixed_noise)
    expected = torch.stack([
        math.sqrt(ddpm['alphas_cumprod'][idx].item()) * x0[i]
        + math.sqrt(1 - ddpm['alphas_cumprod'][idx].item()) * fixed_noise[i]
        for i,idx in enumerate(t)
    ])
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert actual.shape == x0.shape
    assert actual.device == device


def test_forward_gaussian_moments(ddpm):
    device = ddpm['device']
    n, level = 40000, 400
    x0 = torch.full((n, 1), 0.8, device=device)
    result = ddpm['q_sample'](x0, torch.full((n,), level, device=device, dtype=torch.long))
    alpha_bar = ddpm['alphas_cumprod'][level].item()
    assert abs(result.mean().item() - 0.8 * math.sqrt(alpha_bar)) < 0.025
    assert abs(result.var().item() - (1 - alpha_bar)) < 0.03


@pytest.mark.parametrize('kind', ['MLPDiffusion', 'ConvDiffusion'])
def test_model_can_train_and_sample_images(ddpm, kind):
    device = ddpm['device']
    model = ddpm[kind](3 * 32 * 32) if kind == 'MLPDiffusion' else ddpm[kind]()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x0 = torch.randn(2, 3, 32, 32, device=device).clamp(-1, 1)
    t = torch.tensor([0, 500], device=device)
    noise = torch.randn_like(x0)
    xt = ddpm['q_sample'](x0, t, noise)
    before = next(model.parameters()).detach().clone()
    prediction = model(xt, t).reshape_as(noise)
    loss = F.mse_loss(prediction, noise)
    optimizer.zero_grad()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, next(model.parameters()))
    model.eval()
    result = ddpm['p_sample'](model, xt, t)
    assert result.shape == x0.shape
    assert torch.isfinite(result).all()


def test_reverse_final_step_matches_known_posterior_mean(ddpm):
    device = ddpm['device']
    x0 = torch.randn(3, 3, 4, 4, device=device)
    noise = torch.randn_like(x0)
    t = torch.zeros(3, dtype=torch.long, device=device)
    xt = ddpm['q_sample'](x0, t, noise)

    class KnownNoise(nn.Module):
        def forward(self, x, t):
            return noise

    first = ddpm['p_sample'](KnownNoise(), xt, t)
    second = ddpm['p_sample'](KnownNoise(), xt, t)
    # На последнем переходе истинный шум позволяет восстановить x0, шум не добавляется.
    torch.testing.assert_close(first, x0, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_reverse_uses_beta_variance(ddpm):
    device = ddpm['device']
    class ZeroNoise(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)
    n, level = 40000, 500
    result = ddpm['p_sample'](ZeroNoise(), torch.zeros(n, 1, device=device),
                             torch.full((n,), level, dtype=torch.long, device=device))
    beta = ddpm['betas'][level].item()
    assert abs(result.mean().item()) < 0.004
    assert abs(result.var().item() - beta) < beta * 0.035


def test_sampling_loop_visits_all_levels_in_reverse(ddpm):
    visited = []
    class Observer(nn.Module):
        def forward(self, x, t):
            visited.append(t[0].item())
            assert torch.all(t == t[0])
            return torch.zeros_like(x)
    result = ddpm['p_sample_loop'](Observer(), (2, 1), ddpm['device'])
    assert visited == list(reversed(range(ddpm['T'])))
    assert result.shape == (2, 1)
    assert torch.isfinite(result).all()
