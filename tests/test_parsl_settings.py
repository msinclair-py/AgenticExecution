"""Parsl scaling config: the one piece migrated for HPC deployment."""
from pathlib import Path

from agentic_execution.utils.parsl_settings import HeterogeneousSettings


def _by_label(cfg):
    return {e.label: e for e in cfg.executors}


def test_gpu_executor_pins_one_worker_per_accelerator(tmp_path):
    ex = _by_label(HeterogeneousSettings(
        available_accelerators=4, nodes=8, worker_init='x',
    ).config_factory(tmp_path))
    gpu = ex['gpu']
    assert gpu.max_workers_per_node == 4
    assert list(gpu.available_accelerators) == ['0', '1', '2', '3']
    assert gpu.provider.nodes_per_block == 8


def test_worker_init_propagates_to_both_executors(tmp_path):
    ex = _by_label(HeterogeneousSettings(
        available_accelerators=2, nodes=1, worker_init='export TMPDIR=/tmp',
    ).config_factory(tmp_path))
    for label in ('gpu', 'cpu'):
        assert ex[label].provider.worker_init == 'export TMPDIR=/tmp'


def test_yaml_round_trip(tmp_path):
    s = HeterogeneousSettings(available_accelerators=4, nodes=2, worker_init='y')
    path = tmp_path / 'settings.yaml'
    s.dump_yaml(path)
    assert HeterogeneousSettings.from_yaml(path) == s
