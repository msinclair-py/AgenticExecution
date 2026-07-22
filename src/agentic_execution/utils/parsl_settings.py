from abc import ABC, abstractmethod
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from pathlib import Path
from pydantic import BaseModel
from typing import Sequence, Type, TypeVar
import yaml

_T = TypeVar('_T')

class BaseSettings(BaseModel):
    def dump_yaml(self,
                  filename: Path) -> None:
        with open(filename, 'w') as f:
            yaml.dump(self.model_dump(mode='json'), f, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T],
                  filename: Path) -> _T:
        with open(filename) as f:
            raw_data = yaml.safe_load(f)
        return cls(**raw_data) # type: ignore

class BaseComputeSettings(ABC, BaseSettings):
    """Compute settings (HPC platform, number of GPUs, etc)."""

    @abstractmethod
    def config_factory(self,
                       run_dir: Path) -> Config:
        """Create new parsl config"""

class HeterogeneousSettings(BaseComputeSettings):
    available_accelerators: int | Sequence[str] = 12
    worker_init: str = ''
    nodes: int = 1
    retries: int = 1
    cores_per_worker: float = 1.0
    max_workers_per_node: int = 1
    worker_port_range: tuple[int, int] = (10000, 20000)

    def config_factory(self,
                       run_dir: Path) -> Config:
        if isinstance(self.available_accelerators, int):
            n_gpu_workers = self.available_accelerators
        else:
            n_gpu_workers = len(self.available_accelerators)

        executors = [
            HighThroughputExecutor(
                provider=LocalProvider(
                    nodes_per_block=self.nodes,
                    init_blocks=1,
                    max_blocks=1,
                    launcher=MpiExecLauncher(
                        bind_cmd='--cpu-bind',
                        overrides='--depth=1 --ppn 1',
                    ),
                    worker_init=self.worker_init,
                ),
                label='gpu',
                cpu_affinity='block',
                max_workers_per_node=n_gpu_workers,
                worker_debug=True,
                available_accelerators=self.available_accelerators,
                worker_port_range=self.worker_port_range,
            ),
            HighThroughputExecutor(
                provider=LocalProvider(
                    nodes_per_block=self.nodes,
                    init_blocks=1,
                    max_blocks=1,
                    launcher=MpiExecLauncher(
                        bind_cmd='--cpu-bind',
                        overrides='--depth=1 --ppn 1',
                    ),
                    worker_init=self.worker_init,
                ),
                label='cpu',
                max_workers_per_node=self.max_workers_per_node,
                cores_per_worker=self.cores_per_worker,
                worker_debug=True,
                worker_port_range=self.worker_port_range,
            ),
        ]

        return Config(
            run_dir=str(run_dir / 'runinfo'),
            retries=self.retries,
            executors=executors
        )
