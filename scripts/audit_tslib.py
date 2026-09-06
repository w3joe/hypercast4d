"""Compare vendored cores against a local checkout of the pinned upstream source.

Run: python scripts/audit_tslib.py /path/to/Time-Series-Library
No downloads or writes. Only run against the trusted, unmodified MIT checkout.
Dependencies are the vendored, namespace-relocated layers. Model files themselves
are independently loaded from the verified git checkout. CPU float32 only.
"""
import argparse
import re
import subprocess
from pathlib import Path
from types import ModuleType

import torch

from hypercast4d.upstream_models import UPSTREAM_MODELS, UpstreamSequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.checkout), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == '4e938a1767106324dd753b2a44832bf870a0252e', revision
    assert not subprocess.check_output(['git', '-C', str(args.checkout), 'status', '--porcelain'], text=True).strip()
    torch.set_num_threads(1)
    for key, name in UPSTREAM_MODELS.items():
        source = (args.checkout / 'models' / f'{name}.py').read_text()
        source = re.sub(r'from (layers|utils|models)\.', r'from hypercast4d._vendor.tslib.\1.', source)
        original_module = ModuleType(f'audit_original_{name}')
        exec(compile(source, f'upstream/models/{name}.py', 'exec'), original_module.__dict__)
        torch.manual_seed(23)
        adapter = UpstreamSequence(key, 10, 4, dropout=0).eval()
        torch.manual_seed(23)
        original = original_module.Model(adapter.config).cpu().eval()
        for parameter, value in original.state_dict().items():
            torch.testing.assert_close(value, adapter.model.state_dict()[parameter], rtol=0, atol=0)
        history = adapter.prepare_input(torch.randn(2, 10, 4))
        actual = adapter.model(history.clone(), None, torch.zeros_like(history), None)
        expected = original(history.clone(), None, torch.zeros_like(history), None)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for (n, p), (m, q) in zip(adapter.model.named_parameters(), original.named_parameters()):
            assert n == m and (p.grad is None) == (q.grad is None), (key, n, m)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)
        print(f'{name}: initialization, output and parameter gradients match pinned source', flush=True)


if __name__ == '__main__':
    main()
