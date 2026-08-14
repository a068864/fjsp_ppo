"""Gymnasium-native Flexible Job Shop Scheduling environment.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch_geometric.data import HeteroData

from config import EnvConfig
from utils import get_device, unflatten_action

# Operation feature columns in state["operation"].x
OP_DURATION = 0
OP_PROGRESS = 1
OP_SEQ_DEPS = 2
OP_PAR_DEPS = 3
OP_CROSS_DEPS = 4
OP_SCHEDULED = 5
OP_PROCESSING = 6
OP_FINISHED = 7
OP_REMAINING = 8
OP_ELIGIBLE_COUNT = 9


class HeteroGraphSpace(spaces.Space):
    """Truthful subspace for live ``HeteroData`` graph observations."""

    def __init__(self) -> None:
        super().__init__(shape=None, dtype=None)

    def sample(self, mask: Any = None) -> HeteroData:
        return HeteroData()

    def contains(self, x: Any) -> bool:
        return isinstance(x, HeteroData)


class GraphObsSpace(spaces.Space):
    """Gymnasium space for opaque graph observations.

    SB3 / Gymnasium require an ``observation_space``. The policy consumes the
    ``HeteroData`` graph directly from the observation dict; this space does
    not flatten graph features into a vector.
    """

    def __init__(self, n_actions: int) -> None:
        super().__init__(shape=None, dtype=None)
        self.n_actions = int(n_actions)
        self.dummy_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.mask_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
        )
        self.graph_space = HeteroGraphSpace()

    def sample(self, mask: Any = None) -> Dict[str, Any]:
        return {
            "dummy": self.dummy_space.sample(),
            "action_mask": np.ones((self.n_actions,), dtype=np.float32),
            "graph": self.graph_space.sample(),
        }

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        if "dummy" not in x or "action_mask" not in x or "graph" not in x:
            return False
        if not self.dummy_space.contains(np.asarray(x["dummy"], dtype=np.float32)):
            return False
        mask_arr = np.asarray(x["action_mask"], dtype=np.float32)
        if not self.mask_space.contains(mask_arr):
            return False
        return self.graph_space.contains(x["graph"])


def make_sb3_graph_observation_space(n_actions: int) -> spaces.Dict:
    """SB3-friendly Dict space for VecEnv / rollout buffers.

    Runtime ``graph`` values remain object arrays of ``HeteroData``. The graph
    subspace is an empty-shaped Box (not a fake length-1 feature vector) so
    SB3 ``get_obs_shape`` accepts it while documenting that no flat graph
    vector is provided. Env-level ``GraphObsSpace`` still uses
    ``HeteroGraphSpace`` for truthful ``contains`` checks.
    """
    n_actions = int(n_actions)
    return spaces.Dict(
        {
            "dummy": spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            ),
            "action_mask": spaces.Box(
                low=0.0, high=1.0, shape=(n_actions,), dtype=np.float32
            ),
            # Opaque carrier: shape (0,) — not a flattened graph embedding.
            "graph": spaces.Box(
                low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
            ),
        }
    )


class FJSPEnv(gym.Env):
    """Flexible Job Shop Scheduling environment (Gymnasium + PyG HeteroData).

    Observations are opaque dicts containing the live ``HeteroData`` graph and
    an action mask. Actions are flat discrete indices::

        action = machine_id * n_operations + operation_id
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_machines: int = 5,
        n_jobs: int = 3,
        avg_operations_per_job: int = 4,
        time_penalty: float = -0.1,
        max_operation_duration: int = 20,
        connection_drop_prob: float = 0.6,
        compatible_efficiency_std: float = 0.2,
        time_step: float = 1.0,
        min_eligible_machines: int = 2,
        cross_job_dep_prob: float = 0.6,
        shared_dep_prob: float = 0.4,
        seed: Optional[int] = None,
        device: Optional[str] = "cpu",
    ) -> None:
        """Initialize the Gymnasium FJSP environment.

        Args:
            n_machines: Number of available machines.
            n_jobs: Number of jobs in the scheduling problem.
            avg_operations_per_job: Average operations per job.
            time_penalty: Penalty for time consumption.
            max_operation_duration: Maximum duration of an operation.
            connection_drop_prob: Probability of dropping machine-operation connections.
            compatible_efficiency_std: Standard deviation for machine efficiency.
            time_step: Granularity of time progression.
            min_eligible_machines: Minimum machines eligible for an operation.
            cross_job_dep_prob: Probability of cross-job dependencies.
            shared_dep_prob: Probability of shared dependencies.
            seed: Random seed for reproducibility.
            device: Torch device for graph tensors. Defaults to CPU for safe
                use with ``SubprocVecEnv``; pass ``"cuda"``, ``"mps"``, or
                ``"auto"`` if needed.
        """
        super().__init__()

        # Reuse centralized EnvConfig checks for direct constructors.
        EnvConfig(
            n_machines=n_machines,
            n_jobs=n_jobs,
            avg_operations_per_job=avg_operations_per_job,
            time_penalty=time_penalty,
            max_operation_duration=max_operation_duration,
            connection_drop_prob=connection_drop_prob,
            compatible_efficiency_std=compatible_efficiency_std,
            time_step=time_step,
            min_eligible_machines=min_eligible_machines,
            cross_job_dep_prob=cross_job_dep_prob,
            shared_dep_prob=shared_dep_prob,
        )

        self.device = get_device("auto" if device is None else device)

        # Scheduling parameters
        self.n_machines = n_machines
        self.n_jobs = n_jobs
        self.avg_operations_per_job = avg_operations_per_job
        self.n_operations = int(n_jobs * avg_operations_per_job)

        # Scheduling constraints and rewards
        self.time_penalty = time_penalty
        self.max_operation_duration = max_operation_duration
        self.connection_drop_prob = connection_drop_prob
        self.cross_job_dep_prob = cross_job_dep_prob
        self.shared_dep_prob = shared_dep_prob

        # Machine and operation configuration
        self.compatible_efficiency_std = compatible_efficiency_std
        self.time_step = time_step
        self.min_eligible_machines = min_eligible_machines

        # Create environment-specific random number generators.
        # Always seed Torch from NumPy so seed=None does not share Torch's fixed default.
        self._seed = seed
        self.np_rng = np.random.RandomState(seed)
        self.torch_gen = torch.Generator(device=self.device)
        self.torch_gen.manual_seed(int(self.np_rng.randint(0, 2**31 - 1)))

        # Environment state tracking
        self.state = None
        self.job_sequences = None
        self.initial_state = None

        # Initialize efficiency and eligibility matrices
        self.efficiency_modifiers = self._generate_efficiency_modifiers()
        self.eligibility_matrix = self._generate_eligibility_matrix()

        # Add time tracking
        self.current_time = 0.0
        self.makespan = float('inf')

        self.last_success = False
        self._episode_steps = 0
        self._cached_action_mask: Optional[np.ndarray] = None
        self._zero = torch.tensor(0.0, device=self.device)
        self._one = torch.tensor(1.0, device=self.device)
        self._done_status = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._front_machines = torch.empty(0, dtype=torch.long, device=self.device)
        self._front_ops = torch.empty(0, dtype=torch.long, device=self.device)
        self._front_can_process = torch.empty(0, dtype=torch.bool, device=self.device)

        self.action_space = spaces.Discrete(self.n_machines * self.n_operations)
        self.observation_space = GraphObsSpace(self.n_machines * self.n_operations)

    def _create_job_sequences(self) -> List[List[int]]:
        """Partition all operations into ``n_jobs`` non-empty job sequences.

        Every operation index in ``[0, n_operations)`` is assigned to exactly one
        job. Sizes are sampled around ``avg_operations_per_job`` then adjusted so
        the counts sum to ``n_operations``.
        """
        operations = list(range(self.n_operations))
        self.np_rng.shuffle(operations)

        if self.n_jobs <= 0:
            raise ValueError(f"n_jobs must be positive, got {self.n_jobs}")
        if self.n_operations < self.n_jobs:
            raise ValueError(
                f"n_operations ({self.n_operations}) must be >= n_jobs ({self.n_jobs})"
            )

        # Initial sizes ~ Poisson, then clamp and repair to exact sum.
        raw = [
            max(1, int(self.np_rng.poisson(self.avg_operations_per_job)))
            for _ in range(self.n_jobs)
        ]
        # Ensure we can fit: cap total at n_operations while keeping >=1 each.
        while sum(raw) > self.n_operations:
            idx = int(np.argmax(raw))
            if raw[idx] > 1:
                raw[idx] -= 1
            else:
                break
        while sum(raw) < self.n_operations:
            idx = int(self.np_rng.randint(0, self.n_jobs))
            raw[idx] += 1

        job_sequences: List[List[int]] = []
        cursor = 0
        for size in raw:
            job_ops = operations[cursor : cursor + size]
            cursor += size
            self.np_rng.shuffle(job_ops)
            job_sequences.append(job_ops)

        assert cursor == self.n_operations
        assert sum(len(j) for j in job_sequences) == self.n_operations
        return job_sequences

    def _generate_efficiency_modifiers(self) -> torch.Tensor:
        """
        Generate machine efficiency modifiers for operation-machine pairs.

        Returns:
            Tensor of efficiency modifiers (as speedup / slowdown factors)
        """
        modifiers = torch.normal(
            mean=torch.ones((self.n_operations, self.n_machines), dtype=torch.float32, device=self.device),
            std=self.compatible_efficiency_std * torch.ones((self.n_operations, self.n_machines), dtype=torch.float32, device=self.device),
            generator=self.torch_gen
        ).clamp(min=0.5, max=1.5)
        return modifiers

    def _generate_eligibility_matrix(self) -> torch.Tensor:
        """
        Generate machine eligibility matrix with connection dropping.

        Returns:
            Binary tensor indicating machine-operation eligibility
        """
        n_ops, n_mach = self.n_operations, self.n_machines
        scores = torch.rand(
            (n_ops, n_mach), generator=self.torch_gen, device=self.device
        )
        perm = torch.argsort(scores, dim=1)
        eligibility = torch.zeros((n_ops, n_mach), dtype=torch.bool, device=self.device)
        eligibility.scatter_(1, perm[:, : self.min_eligible_machines], True)
        extra = n_mach - self.min_eligible_machines
        if extra > 0:
            keep = (
                torch.rand(
                    (n_ops, extra), generator=self.torch_gen, device=self.device
                )
                > self.connection_drop_prob
            )
            eligibility.scatter_(1, perm[:, self.min_eligible_machines :], keep)
        return eligibility

    def _calculate_eligible_machine_counts(self) -> torch.Tensor:
        """
        Calculate the number of eligible machines for each operation.

        Returns:
            Tensor of eligible machine counts for each operation
        """
        return torch.sum(self.eligibility_matrix, dim=1).to(torch.float32)


    @staticmethod
    def _build_adj(
        deps: List[Tuple[int, int]],
        all_ops: Set[int],
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        adj: Dict[int, List[int]] = {op: [] for op in all_ops}
        indeg: Dict[int, int] = {op: 0 for op in all_ops}
        for from_op, to_op in deps:
            adj[from_op].append(to_op)
            indeg[to_op] += 1
        return adj, indeg

    def _has_cycle(self, deps: List[Tuple[int, int]], all_ops: Set[int]) -> bool:
        """Return True if ``deps`` contains a cycle (Kahn / BFS)."""
        adj, indeg = self._build_adj(deps, all_ops)
        queue: deque[int] = deque(op for op, degree in indeg.items() if degree == 0)
        seen = 0
        while queue:
            current = queue.popleft()
            seen += 1
            for neighbor in adj[current]:
                indeg[neighbor] -= 1
                if indeg[neighbor] == 0:
                    queue.append(neighbor)
        return seen != len(all_ops)

    def _get_topological_order(
        self, deps: List[Tuple[int, int]], all_ops: Set[int]
    ) -> Dict[int, int]:
        """Return operation ID -> topo rank using Kahn / BFS."""
        adj, indeg = self._build_adj(deps, all_ops)
        queue: deque[int] = deque(op for op, degree in indeg.items() if degree == 0)
        order: Dict[int, int] = {}
        rank = 0
        while queue:
            current = queue.popleft()
            order[current] = rank
            rank += 1
            for neighbor in adj[current]:
                indeg[neighbor] -= 1
                if indeg[neighbor] == 0:
                    queue.append(neighbor)
        return order

    @staticmethod
    def _can_reach(adj: Dict[int, List[int]], src: int, dst: int) -> bool:
        """BFS reachability: True if ``dst`` is reachable from ``src``."""
        if src == dst:
            return True
        seen = {src}
        queue: deque[int] = deque([src])
        while queue:
            node = queue.popleft()
            for nxt in adj[node]:
                if nxt == dst:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    def _create_enhanced_dependencies(self) -> List[Tuple[int, int, str]]:
        """Build sequential / parallel / cross-job dependencies without cycles."""
        all_ops: Set[int] = set()
        op_to_job: Dict[int, int] = {}
        for job_idx, job_sequence in enumerate(self.job_sequences):
            for op in job_sequence:
                all_ops.add(op)
                op_to_job[op] = job_idx

        deps: List[Tuple[int, int, str]] = []
        adj: Dict[int, List[int]] = {op: [] for op in all_ops}
        seen_pairs: Set[Tuple[int, int]] = set()
        # Transitive closure bitsets: reach[u] bit v set iff u can reach v (incl. self).
        reach: Dict[int, int] = {op: 1 << op for op in all_ops}

        def _reaches(src: int, dst: int) -> bool:
            return ((reach[src] >> dst) & 1) != 0

        def _try_add(u: int, v: int, dep_type: str) -> bool:
            pair = (u, v)
            if pair in seen_pairs:
                return False
            if _reaches(v, u) or _reaches(u, v):
                return False
            deps.append((u, v, dep_type))
            adj[u].append(v)
            seen_pairs.add(pair)
            rv = reach[v]
            for src, bits in reach.items():
                if (bits >> u) & 1:
                    reach[src] = bits | rv
            return True

        for job_sequence in self.job_sequences:
            for i in range(len(job_sequence) - 1):
                u, v = job_sequence[i], job_sequence[i + 1]
                _try_add(u, v, "sequential")

        # Within-job forward skips cannot cycle on a single chain; still guard
        # with reachability so future structural changes stay safe.
        for job_sequence in self.job_sequences:
            job_length = len(job_sequence)
            for i in range(job_length - 2):
                for j in range(i + 2, job_length):
                    if self.np_rng.rand() >= self.shared_dep_prob:
                        continue
                    u, v = job_sequence[i], job_sequence[j]
                    _try_add(u, v, "parallel")

        pair_deps = [(f, t) for f, t, _ in deps]
        op_order = self._get_topological_order(pair_deps, all_ops)

        ops_list = list(all_ops)
        n_cross_deps = int(len(ops_list) * self.cross_job_dep_prob)
        attempts = 0
        max_attempts = max(n_cross_deps * 3, 1)
        added_cross_deps = 0

        while added_cross_deps < n_cross_deps and attempts < max_attempts:
            attempts += 1
            from_op = int(self.np_rng.choice(ops_list))
            from_job = op_to_job[from_op]

            eligible_targets = []
            for to_op in ops_list:
                if (from_op, to_op) in seen_pairs:
                    continue
                if op_to_job[to_op] == from_job:
                    continue
                if op_order[to_op] <= op_order[from_op]:
                    continue
                # Rank filter already forbids a cycle; skip transitive duplicates.
                if _reaches(from_op, to_op):
                    continue
                eligible_targets.append(to_op)

            if not eligible_targets:
                continue

            to_op = int(self.np_rng.choice(eligible_targets))
            if _try_add(from_op, to_op, "cross_job"):
                added_cross_deps += 1
                pair_deps.append((from_op, to_op))
                op_order = self._get_topological_order(pair_deps, all_ops)

        return deps

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and return ``(observation, info)``.

        Options:
            regenerate: Force a new random instance even if the seed matches.
            reuse_instance: Reuse the cached initial graph when possible (fast path).
                Default behavior without this flag regenerates on seedless resets so
                vectorized training sees diverse instances.
        """
        super().reset(seed=seed)
        options = options or {}
        reuse_instance = bool(options.get("reuse_instance", False))
        force_regenerate = bool(options.get("regenerate", False))

        seed_changed = seed is not None and seed != self._seed
        first_reset = not hasattr(self, "initial_state_seed") or not hasattr(self, "initial_state")

        # Seedless resets regenerate by default (unless reuse_instance=True).
        if seed is None and not reuse_instance and not first_reset:
            force_regenerate = True
            seed = int(self.np_rng.randint(0, 2**31 - 1))
            seed_changed = seed != self._seed

        should_regenerate = first_reset or seed_changed or force_regenerate

        # If seed changed or this is the first reset, regenerate everything
        if should_regenerate:
            if seed is not None:
                # New seed takes precedence
                self._seed = seed  # Update the stored seed
                # Update local RNGs
                self.np_rng = np.random.RandomState(seed)
                self.torch_gen = torch.Generator(device=self.device)
                self.torch_gen.manual_seed(int(self.np_rng.randint(0, 2**31 - 1)))
            elif self._seed is not None:
                # If no new seed but a stored seed
                self.np_rng = np.random.RandomState(self._seed)
                self.torch_gen = torch.Generator(device=self.device)
                self.torch_gen.manual_seed(int(self.np_rng.randint(0, 2**31 - 1)))
            else:
                self.np_rng = np.random.RandomState(None)
                self.torch_gen = torch.Generator(device=self.device)
                self.torch_gen.manual_seed(int(self.np_rng.randint(0, 2**31 - 1)))

            # Generate job sequences
            self.job_sequences = self._create_job_sequences()

            # Generate enhanced dependencies with types
            typed_deps = self._create_enhanced_dependencies()

            # Extract just the (from, to) pairs for the graph edges
            deps = [(from_op, to_op) for from_op, to_op, _ in typed_deps]

            # Convert dependencies to tensor format
            from_ops, to_ops = zip(*deps) if deps else ([], [])
            edge_index = torch.tensor([list(from_ops), list(to_ops)], dtype=torch.long, device=self.device)

            # Store dependency types
            self.dependency_types = {}
            for from_op, to_op, dep_type in typed_deps:
                self.dependency_types[(from_op, to_op)] = dep_type

            # Regenerate matrices
            self.eligibility_matrix = self._generate_eligibility_matrix()
            self.efficiency_modifiers = self._generate_efficiency_modifiers()

            # Calculate eligible machine counts
            eligible_machine_counts = self._calculate_eligible_machine_counts()

            # Initialize state graph
            self.state = HeteroData()

            mean = torch.tensor(self.max_operation_duration / 2, device=self.device)
            std = torch.tensor(self.max_operation_duration / 4, device=self.device)

            # Use torch generator for the log normal distribution
            loc = torch.log(mean / torch.sqrt(1 + (std/mean)**2))
            scale = torch.sqrt(torch.log(1 + (std/mean)**2))
            normal_sample = loc + scale * torch.randn((self.n_operations, 1), generator=self.torch_gen, device=self.device)
            log_normal_sample = torch.exp(normal_sample)

            # Scale to desired range and ensure minimum duration
            base_times = self.time_step * torch.clamp(
                log_normal_sample,
                min=1.0,
                max=float(self.max_operation_duration)
            )

            # Calculate dependency counts by type
            sequential_deps = torch.zeros(self.n_operations, device=self.device)
            parallel_deps = torch.zeros(self.n_operations, device=self.device)
            cross_job_deps = torch.zeros(self.n_operations, device=self.device)

            if typed_deps:
                for from_op, to_op, dep_type in typed_deps:
                    # Count incoming dependencies on the successor operation.
                    if dep_type == "sequential":
                        sequential_deps[to_op] += 1
                    elif dep_type == "parallel":
                        parallel_deps[to_op] += 1
                    elif dep_type == "cross_job":
                        cross_job_deps[to_op] += 1

            compatible_pairs = self.eligibility_matrix.nonzero(as_tuple=False)
            if compatible_pairs.numel() > 0:
                self.state["operation", "compatible", "machine"].edge_index = (
                    compatible_pairs.t().contiguous()
                )
                self.state["operation", "compatible", "machine"].edge_attr = (
                    self.efficiency_modifiers[
                        compatible_pairs[:, 0], compatible_pairs[:, 1]
                    ]
                    .unsqueeze(-1)
                    .to(torch.float32)
                )

            # Initialize operation features
            self.state['operation'].x = torch.cat([
                base_times,
                torch.zeros(self.n_operations, 1, dtype=torch.float32, device=self.device),
                sequential_deps.unsqueeze(-1).to(torch.float32),
                parallel_deps.unsqueeze(-1).to(torch.float32),
                cross_job_deps.unsqueeze(-1).to(torch.float32),
                torch.zeros(self.n_operations, 3, dtype=torch.float32, device=self.device),
                base_times,
                eligible_machine_counts.unsqueeze(-1)
            ], dim=1)

            # Initialize machine features
            self.state['machine'].x = torch.zeros(self.n_machines, 3, device=self.device)

            # Initialize edges
            self.state['operation', 'precede', 'operation'].edge_index = edge_index
            self.state['operation', 'next', 'operation'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self.state['machine', 'processing', 'operation'].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

            # Store the initial state and the seed used to generate it
            self.initial_state = self.state.clone()
            self.initial_state_seed = self._seed

            # Also cache these values that we'll need to restore on fast reset
            self.cached_eligibility_matrix = self.eligibility_matrix.clone()
            self.cached_efficiency_modifiers = self.efficiency_modifiers.clone()
            self.cached_dependency_types = dict(self.dependency_types)
            self.cached_job_sequences = [list(seq) for seq in self.job_sequences]
        else:
            # Fast reset path - use cached initial state
            self.state = self.initial_state.clone()

            # Restore cached values directly
            self.eligibility_matrix = self.cached_eligibility_matrix.clone()
            self.efficiency_modifiers = self.cached_efficiency_modifiers.clone()
            self.dependency_types = dict(self.cached_dependency_types)
            self.job_sequences = [list(seq) for seq in self.cached_job_sequences]

        self.current_time = 0.0
        self.makespan = float("inf")
        self.last_success = False
        self._episode_steps = 0
        self._cached_action_mask = None

        obs = self._get_obs()
        info = self._get_info(success=False, action_mask=obs["action_mask"])
        return obs, info

    _MUTABLE_EDGE_TYPES = (
        ("machine", "processing", "operation"),
        ("operation", "processed_by", "machine"),
        ("operation", "next", "operation"),
        ("operation", "precede", "operation"),
    )

    def _remove_operation_edges(self, operation: int):
        """Remove all edges connected to a completed operation."""
        self._remove_operations_edges([int(operation)])

    def _remove_operations_edges(self, operations: List[int]) -> None:
        """Batch-remove edges for completed operations; decrement successor dep counts."""
        if not operations:
            return
        dead = torch.zeros(self.n_operations, dtype=torch.bool, device=self.device)
        dead[torch.tensor(operations, dtype=torch.long, device=self.device)] = True

        precede_key = ("operation", "precede", "operation")
        if precede_key in self.state.edge_index_dict:
            edge_index = self.state[precede_key].edge_index
            if edge_index.numel() > 0:
                dying = dead[edge_index[0]]
                if bool(dying.any().item()):
                    op_x = self.state["operation"].x
                    for src, succ in zip(
                        edge_index[0][dying].tolist(),
                        edge_index[1][dying].tolist(),
                    ):
                        dep_type = self.dependency_types.get((int(src), int(succ)))
                        col = (
                            OP_SEQ_DEPS
                            if dep_type == "sequential"
                            else OP_PAR_DEPS
                            if dep_type == "parallel"
                            else OP_CROSS_DEPS
                            if dep_type == "cross_job"
                            else None
                        )
                        if col is None:
                            continue
                        op_x[succ, col] = torch.maximum(op_x[succ, col] - 1, self._zero)

        for key in self._MUTABLE_EDGE_TYPES:
            if key not in self.state.edge_index_dict:
                continue
            edge_index = self.state[key].edge_index
            if edge_index.size(1) == 0:
                continue
            src_is_op = key[0] == "operation"
            dst_is_op = key[2] == "operation"
            mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=self.device)
            if src_is_op:
                mask &= ~dead[edge_index[0]]
            if dst_is_op:
                mask &= ~dead[edge_index[1]]
            self._apply_edge_keep_mask(key, mask)

    def _apply_edge_keep_mask(self, key, mask: torch.Tensor) -> None:
        edge_index = self.state[key].edge_index
        attr = getattr(self.state[key], "edge_attr", None)
        if not bool(mask.any().item()):
            self.state[key].edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )
            if attr is not None:
                self.state[key].edge_attr = torch.empty(
                    (0, attr.size(1)), dtype=attr.dtype, device=self.device
                )
            return
        self.state[key].edge_index = edge_index[:, mask]
        if attr is None:
            return
        if len(attr) == len(mask):
            self.state[key].edge_attr = attr[mask]
        else:
            self.state[key].edge_attr = torch.zeros(
                (int(mask.sum().item()), attr.size(1)),
                dtype=attr.dtype,
                device=self.device,
            )

    def _compute_action_mask(self, state: Optional[HeteroData] = None) -> np.ndarray:
        """Vectorized valid-action mask: unscheduled ∧ prereqs started ∧ eligible."""
        if state is None:
            state = self.state
        n_actions = self.n_machines * self.n_operations
        if state is None or "operation" not in state.node_types:
            return np.zeros((n_actions,), dtype=np.float32)

        op_x = state["operation"].x
        unscheduled = (op_x[:, OP_SCHEDULED] == 0) & (op_x[:, OP_FINISHED] == 0)
        started = (
            op_x[:, OP_SCHEDULED] + op_x[:, OP_PROCESSING] + op_x[:, OP_FINISHED]
        ) > 0

        prereq_ok = torch.ones(self.n_operations, dtype=torch.bool, device=op_x.device)
        dep = state["operation", "precede", "operation"].edge_index
        if dep.numel() > 0:
            src, dst = dep[0], dep[1]
            bad_dst = dst[~started[src]]
            if bad_dst.numel() > 0:
                prereq_ok[bad_dst.unique()] = False

        ready_ops = unscheduled & prereq_ok  # (n_ops,)
        # eligibility: (n_ops, n_machines); action layout is machine-major.
        elig = self.eligibility_matrix & ready_ops.unsqueeze(1)
        return elig.T.reshape(-1).to(dtype=torch.float32).detach().cpu().numpy()

    def get_action_mask(self) -> np.ndarray:
        """Return a float32 mask of shape ``(n_actions,)`` with 1 = valid."""
        if self._cached_action_mask is None:
            self._cached_action_mask = self._compute_action_mask()
        return np.array(self._cached_action_mask, copy=True)

    def get_eligible_machines(self, operation: int) -> List[int]:
        """Get list of eligible machines for a given operation."""
        if not (0 <= int(operation) < self.n_operations):
            return []
        return torch.where(self.eligibility_matrix[operation])[0].tolist()

    def _is_gridlock(
        self,
        processing: torch.Tensor,
        blocked_machines: int,
    ) -> bool:
        """True when queued fronts exist but none can process."""
        proc_edges = self.state["machine", "processing", "operation"].edge_index
        if proc_edges.numel() == 0:
            return False
        if bool(processing.any().item()):
            return False
        return int(blocked_machines) > 0

    def _policy_graph_snapshot(self) -> HeteroData:
        """Clone only policy-consumed tensors (features, edges, efficiency attrs)."""
        from models.graph_encoder import EDGE_TYPES

        if self.state is None:
            return HeteroData()
        out = HeteroData()
        out["operation"].x = self.state["operation"].x.clone()
        out["machine"].x = self.state["machine"].x.clone()
        for edge_type in EDGE_TYPES:
            if edge_type not in self.state.edge_types:
                continue
            edge_index = self.state[edge_type].edge_index
            if edge_index is None or edge_index.numel() == 0:
                continue
            edge_attr = getattr(self.state[edge_type], "edge_attr", None)
            if edge_type == ("operation", "compatible", "machine"):
                alive = out["operation"].x[:, OP_FINISHED] < 0.5
                keep = alive[edge_index[0]]
                if not bool(keep.any().item()):
                    continue
                out[edge_type].edge_index = edge_index[:, keep].contiguous()
                if edge_attr is not None:
                    out[edge_type].edge_attr = edge_attr[keep].contiguous()
                continue
            out[edge_type].edge_index = edge_index.clone()
            if edge_attr is not None:
                out[edge_type].edge_attr = edge_attr.clone()
        return out

    def _get_obs(self) -> Dict[str, Any]:
        """Build the opaque graph observation dict."""
        graph = self._policy_graph_snapshot()
        return {
            "dummy": np.zeros((1,), dtype=np.float32),
            "action_mask": self.get_action_mask(),
            "graph": graph,
        }

    def _get_info(
        self,
        success: bool = False,
        action_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Build the info dict for reset/step."""
        mask = action_mask if action_mask is not None else self.get_action_mask()
        return {
            "makespan": float(self.makespan),
            "success": bool(success),
            "current_time": float(self.current_time),
            "n_valid_actions": int(np.sum(mask)),
            "episode_steps": int(self._episode_steps),
        }

    def get_valid_actions(self, state=None) -> List[int]:
        """Return valid flat action indices for ``state`` (defaults to current)."""
        if state is None:
            mask = self.get_action_mask()
        else:
            mask = self._compute_action_mask(state)
        return np.flatnonzero(mask).astype(np.int64).tolist()

    def _is_valid_action(self, action: Tuple[int, int], *, check_prereqs: bool = True) -> bool:
        machine, operation = action
        if not (0 <= machine < self.n_machines and 0 <= operation < self.n_operations):
            return False
        op_x = self.state["operation"].x
        if float(op_x[operation, OP_SCHEDULED]) != 0.0:
            return False
        if float(op_x[operation, OP_FINISHED]) != 0.0:
            return False
        if not bool(self.eligibility_matrix[operation, machine].item()):
            return False
        if not check_prereqs:
            return True

        dep_edge_index = self.state["operation", "precede", "operation"].edge_index
        if dep_edge_index.numel() == 0:
            return True
        prereqs_mask = dep_edge_index[1] == operation
        if not prereqs_mask.any():
            return True
        prerequisites = dep_edge_index[0][prereqs_mask]
        prereq_statuses = op_x[prerequisites, OP_SCHEDULED : OP_FINISHED + 1]
        return bool(torch.all(torch.sum(prereq_statuses, dim=1) > 0))

    def _get_processing_operations(self) -> Tuple[torch.Tensor, int]:
        """
        Identify processing operations and count blocked machines.

        Returns:
            Tuple of:
            - processing: Boolean tensor indicating which operations can process
            - blocked_machines: Number of machines with front operations that cannot be processed
        """
        processing = torch.zeros(self.n_operations, dtype=torch.bool, device=self.device)
        self._front_machines = torch.empty(0, dtype=torch.long, device=self.device)
        self._front_ops = torch.empty(0, dtype=torch.long, device=self.device)
        self._front_can_process = torch.empty(0, dtype=torch.bool, device=self.device)
        edge_index = self.state["machine", "processing", "operation"].edge_index
        if edge_index.numel() == 0:
            return processing, 0

        machines = edge_index[0]
        ops = edge_index[1]
        n_edges = int(machines.numel())
        order = torch.arange(n_edges, device=self.device)
        first = torch.full(
            (self.n_machines,), n_edges, dtype=torch.long, device=self.device
        )
        first.scatter_reduce_(0, machines.long(), order, reduce="amin")
        busy = first < n_edges
        if not bool(busy.any().item()):
            return processing, 0
        front_machines = torch.nonzero(busy, as_tuple=False).view(-1)
        front_ops = ops[first[front_machines]]

        finished = self.state["operation"].x[:, OP_FINISHED] == 1
        prereq_ok = torch.ones(self.n_operations, dtype=torch.bool, device=self.device)
        dep = self.state["operation", "precede", "operation"].edge_index
        if dep.numel() > 0:
            unfinished_pred = ~finished[dep[0]]
            if bool(unfinished_pred.any().item()):
                prereq_ok[dep[1][unfinished_pred]] = False

        can_process = prereq_ok[front_ops]
        processing[front_ops] = can_process
        self._front_machines = front_machines
        self._front_ops = front_ops
        self._front_can_process = can_process
        blocked_machines = int((~can_process).sum().item())
        return processing, blocked_machines

    def _ticks_to_next_completion(self, processing_ops: torch.Tensor) -> int:
        remaining = self.state["operation"].x[processing_ops, OP_REMAINING]
        dt = float(self.time_step)
        ticks = torch.ceil((remaining - 1e-6).clamp(min=0.0) / dt)
        return max(1, int(ticks.min().item()))

    def _stamp_idle_fronts(self) -> None:
        if self._front_machines.numel() == 0:
            return
        idle = self._front_machines[~self._front_can_process]
        if idle.numel() > 0:
            self.state["machine"].x[idle, 2] = self.current_time

    def _apply_processing_work(
        self, n_ticks: int, processing_ops: torch.Tensor
    ) -> List[int]:
        """Apply ``n_ticks`` of work to current fronts. Completions only at the end."""
        dt = float(self.time_step) * int(n_ticks)
        dt_t = torch.tensor(dt, device=self.device)
        self.current_time += dt
        self._stamp_idle_fronts()
        if not bool(processing_ops.any().item()):
            return []

        op_features = self.state["operation"].x
        can_m = self._front_machines[self._front_can_process]
        can_ops = self._front_ops[self._front_can_process]
        rem_front = op_features[can_ops, OP_REMAINING]
        work_front = torch.minimum(rem_front, dt_t)
        if can_m.numel() > 0:
            self.state["machine"].x[can_m, 1] = torch.maximum(
                self.state["machine"].x[can_m, 1] - work_front, self._zero
            )

        op_features[:, OP_PROCESSING] = self._zero
        op_features[can_ops, OP_PROCESSING] = self._one
        op_features[can_ops, OP_REMAINING] = torch.maximum(
            rem_front - work_front, self._zero
        )
        op_features[can_ops, OP_PROGRESS] = torch.clamp(
            1
            - torch.div(
                op_features[can_ops, OP_REMAINING],
                op_features[can_ops, OP_DURATION],
            ),
            min=self._zero,
            max=self._one,
        )

        completed = processing_ops & (
            (op_features[:, OP_REMAINING] <= 1e-6)
            | (op_features[:, OP_PROGRESS] >= 1 - 1e-6)
        )
        if not bool(completed.any().item()):
            return []

        op_features[completed, OP_PROGRESS] = self._one
        op_features[completed, OP_REMAINING] = self._zero
        op_features[completed, OP_SCHEDULED : OP_FINISHED + 1] = self._done_status
        op_features[completed, OP_SEQ_DEPS : OP_CROSS_DEPS + 1] = self._zero

        done_on_front = completed[can_ops]
        done_machines = can_m[done_on_front]
        if done_machines.numel() > 0:
            self.state["machine"].x[done_machines, 0] = torch.maximum(
                self.state["machine"].x[done_machines, 0] - 1, self._zero
            )
            newly_idle = done_machines[self.state["machine"].x[done_machines, 0] == 0]
            if newly_idle.numel() > 0:
                self.state["machine"].x[newly_idle, 2] = self.current_time

        completed_ids = [int(i) for i in torch.where(completed)[0].tolist()]
        self._remove_operations_edges(completed_ids)
        op_features[:, OP_SEQ_DEPS : OP_CROSS_DEPS + 1] = torch.maximum(
            op_features[:, OP_SEQ_DEPS : OP_CROSS_DEPS + 1], self._zero
        )
        return completed_ids

    def _advance_time_tick(self) -> List[int]:
        """Advance simulation by one fixed ``time_step`` and complete finished ops."""
        return self._advance_time_ticks(1)

    def _advance_time_ticks(self, n_ticks: int) -> List[int]:
        """Advance ``n_ticks`` discrete steps, jumping between completion events.

        A jump never crosses a completion, so a queued successor cannot start in
        the same interval that its predecessor finishes (no unused-tick transfer).
        """
        left = int(n_ticks)
        if left <= 0:
            return []
        completed_all: List[int] = []
        dt = float(self.time_step)
        while left > 0:
            processing_ops, _blocked = self._get_processing_operations()
            if not bool(processing_ops.any().item()):
                self.current_time += left * dt
                self._stamp_idle_fronts()
                break
            k = min(left, self._ticks_to_next_completion(processing_ops))
            completed_all.extend(self._apply_processing_work(k, processing_ops))
            left -= k
        return completed_all

    def _add_edge(self, edge_type: str, src: int, dst: int, edge_attr=None):
        """
        Add edge to graph for processing, precede, or next relationships.

        Args:
            edge_type (str): Type of edge ('processing', 'precede', 'next')
            src (int): Source node index
            dst (int): Destination node index
            edge_attr (torch.Tensor, optional): Edge attribute tensor
        """
        # Determine the edge key based on type
        if edge_type == 'processing':
            key = ('machine', 'processing', 'operation')
        elif edge_type == 'processed_by':
            key = ('operation', 'processed_by', 'machine')
        elif edge_type in ['precede', 'next']:
            key = ('operation', edge_type, 'operation')
        else:
            raise ValueError(
                f"Invalid edge type: {edge_type}. Expected 'processing', "
                "'processed_by', 'precede', or 'next'"
            )

        edge = torch.tensor([[src], [dst]], dtype=torch.long, device=self.device)

        if key not in self.state.edge_index_dict or self.state[key].edge_index.numel() == 0:
            self.state[key].edge_index = edge
            if edge_attr is not None:
                self.state[key].edge_attr = edge_attr
            return

        edge_index = self.state[key].edge_index
        self.state[key].edge_index = torch.cat([edge_index, edge], dim=1)
        if edge_attr is not None:
            if getattr(self.state[key], "edge_attr", None) is not None:
                self.state[key].edge_attr = torch.cat(
                    [self.state[key].edge_attr, edge_attr], dim=0
                )
            else:
                self.state[key].edge_attr = edge_attr

    def schedule_operation(self, machine: int, operation: int):
        """
        Schedule operation on a machine with efficiency modifier consideration.

        Args:
            machine (int): Machine index to schedule operation on
            operation (int): Operation index to be scheduled
        """
        # Verify machine eligibility
        assert self.eligibility_matrix[operation, machine], f"Machine {machine} is not eligible for operation {operation}"

        proc_edges = self.state['machine', 'processing', 'operation'].edge_index

        # If machine has existing operations, add 'next' edge from last operation to this one
        if proc_edges.numel() > 0:
            machine_mask = proc_edges[0] == machine
            if machine_mask.any():
                queue_ops = proc_edges[1][machine_mask]
                if len(queue_ops) > 0:
                    last_op = queue_ops[-1].item()
                    self._add_edge('next', last_op, operation)

        # Get efficiency modifier for this operation-machine pair
        efficiency_modifier = self.efficiency_modifiers[operation, machine]

        # Create edge attribute tensor
        edge_attr = torch.tensor([[efficiency_modifier.item()]], dtype=torch.float32, device=self.device)

        self._add_edge('processing', machine, operation, edge_attr)
        self._add_edge('processed_by', operation, machine, edge_attr)

        base_time = self.state['operation'].x[operation, 0]
        adjusted_time = base_time * efficiency_modifier

        self.state['machine'].x[machine, 0] += 1
        self.state['machine'].x[machine, 1] += adjusted_time

        if self.state['machine'].x[machine, 0] == 1:
            self.state['machine'].x[machine, 2] = self.current_time

        # Effective duration becomes the efficiency-adjusted time so completion %
        # (remaining / duration) stays consistent after scheduling.
        self.state["operation"].x[operation, OP_DURATION] = adjusted_time
        self.state["operation"].x[operation, OP_SCHEDULED] = 1
        self.state["operation"].x[operation, OP_PROCESSING] = 0
        self.state["operation"].x[operation, OP_REMAINING] = adjusted_time

    def get_efficiency_modifier(self, operation: int, machine: int) -> float:
        """Get the efficiency modifier for an operation-machine pair."""
        return self.efficiency_modifiers[operation, machine].item()

    def estimated_completion(self) -> float:
        """Lower bound on makespan: clock plus remaining queued machine work."""
        if self.state is None:
            return float(self.current_time)
        workload = self.state["machine"].x[:, 1]
        max_wl = float(workload.max().item()) if workload.numel() else 0.0
        return float(self.current_time) + max_wl

    def failure_penalty(self) -> float:
        """Return worse than serial processing at max duration and slowdown."""
        return (
            float(self.time_penalty)
            * float(self.n_operations)
            * float(self.max_operation_duration)
            * 4.0
        )

    def _completion_delta_reward(self, before: float, after: float) -> float:
        return float(self.time_penalty) * (float(after) - float(before))

    def step(
        self,
        action: int,
    ) -> Tuple[Dict[str, Any], SupportsFloat, bool, bool, Dict[str, Any]]:
        """Execute one scheduling step.

        Args:
            action: Flat discrete action ``machine_id * n_operations + operation_id``.

        Returns:
            Gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``.
        """
        if isinstance(action, (tuple, list)) and len(action) == 2:
            machine, operation = int(action[0]), int(action[1])
        else:
            machine, operation = unflatten_action(int(np.asarray(action).item()), self.n_operations)

        in_range = 0 <= machine < self.n_machines and 0 <= operation < self.n_operations
        mask = self._cached_action_mask
        if mask is None:
            mask = self._compute_action_mask()
        flat = machine * self.n_operations + operation
        if not in_range or flat >= mask.size or float(mask[flat]) <= 0.0:
            raise ValueError(
                f"Invalid action: (m:{machine}, op:{operation}). "
                f"Eligible machines for operation {operation}: "
                f"{self.get_eligible_machines(operation) if in_range else []}; "
                f"valid_actions={self.get_valid_actions()[:20]}"
            )
        self._cached_action_mask = None
        self._episode_steps += 1

        before = self.estimated_completion()
        self.schedule_operation(machine, operation)

        processing_ops, blocked_machines = self._get_processing_operations()
        if self._is_gridlock(processing_ops, blocked_machines):
            # Still advance the clock so makespan/time stay consistent, then terminate.
            self.current_time += float(self.time_step)
            reward = self._completion_delta_reward(before, self.estimated_completion())
            reward += self.failure_penalty()
            self.last_success = False
            self.makespan = float("inf")
            obs = self._get_obs()
            info = self._get_info(success=False)
            info["is_gridlock"] = True
            return obs, float(reward), True, False, info

        self._apply_processing_work(1, processing_ops)
        reward = self._completion_delta_reward(before, self.estimated_completion())

        # Detect gridlock after the tick (e.g. newly blocked fronts).
        processing_ops, blocked_machines = self._get_processing_operations()
        if self._is_gridlock(processing_ops, blocked_machines):
            reward += self.failure_penalty()
            self.last_success = False
            self.makespan = float("inf")
            obs = self._get_obs()
            info = self._get_info(success=False, action_mask=obs["action_mask"])
            info["is_gridlock"] = True
            return obs, float(reward), True, False, info

        if self.terminal():
            r, success = self.rollout()
            reward += r
            if success:
                self.makespan = self.current_time
            else:
                reward += self.failure_penalty()
            self.last_success = bool(success)
            obs = self._get_obs()
            info = self._get_info(success=bool(success), action_mask=obs["action_mask"])
            return obs, float(reward), True, False, info

        self.last_success = False
        obs = self._get_obs()
        info = self._get_info(success=False, action_mask=obs["action_mask"])
        if float(np.sum(obs["action_mask"])) <= 0.0:
            info["no_valid_actions"] = True
            if self._is_gridlock(processing_ops, blocked_machines):
                info["is_gridlock"] = True
            return obs, float(reward + self.failure_penalty()), True, False, info
        return obs, float(reward), False, False, info

    def rollout(self) -> Tuple[float, bool]:
        """Simulate remaining execution under FIFO queue assumption.

        Jumps to the next completion event; each jump equals the same number of
        fixed ``time_step`` ticks ``step`` would have applied.
        """
        reward = 0.0
        guard = 0
        max_ticks = max(10_000, int(self.n_operations * self.max_operation_duration * 4) + 1)

        while True:
            op_features = self.state["operation"].x
            if bool(torch.all(op_features[:, OP_FINISHED] > 0.5).item()):
                return reward, True

            proc_edges = self.state["machine", "processing", "operation"].edge_index
            if proc_edges.numel() == 0:
                return reward, False

            processing_ops, _blocked = self._get_processing_operations()
            if not bool(processing_ops.any().item()):
                return reward, False
            if guard >= max_ticks:
                return reward, False

            k = min(self._ticks_to_next_completion(processing_ops), max_ticks - guard)
            k = max(1, k)
            before = self.estimated_completion()
            self._apply_processing_work(k, processing_ops)
            reward += self._completion_delta_reward(before, self.estimated_completion())
            guard += k
            if guard > max_ticks:
                return reward, False

    def terminal(self) -> bool:
        """Return True when every operation has left the initial state."""
        op_x = self.state["operation"].x
        return bool(
            torch.all(
                (op_x[:, OP_SCHEDULED] == self._one)
                | (op_x[:, OP_PROCESSING] == self._one)
                | (op_x[:, OP_FINISHED] == self._one)
            ).item()
        )

