"""Eager baseline: execute independent samples with one shared step lifecycle."""

from copy import copy

import torch

from memt_sglang.protocol import SessionError
from memt_sglang.runtime.plan import StepPlan
from memt_sglang.runtime.state import StatePool
from memt_sglang.runtime.worker_sessions import WorkerSessions


class MemoryRuntime:
    def __init__(self, pool: StatePool):
        self.pool = pool
        self.plan: StepPlan | None = None
        self._layers: set[int] = set()
        self.worker_sessions = WorkerSessions(pool)
        self.kv_sessions = set()

    def begin(self, forward_batch) -> StepPlan:
        if self.plan is not None:
            raise SessionError("Previous step has not finished")
        batch = copy(forward_batch)
        batch.rids = [
            rid
            if rid in self.worker_sessions.requests
            and not self.worker_sessions.requests[rid].complete
            else "retired"
            for rid in (forward_batch.rids or [])
        ]
        plan = StepPlan.from_batch(batch)
        self.plan = plan
        return plan

    @torch.no_grad()
    def run_layer(
        self, layer: int, hidden: torch.Tensor, output: torch.Tensor, compute
    ) -> torch.Tensor:
        """Execute prepared samples and fill caller-owned residual rows."""
        if self.plan is None or layer in self._layers or layer not in self.pool.storage:
            raise SessionError("Each registered memory layer must run once per step")
        if (
            hidden.ndim != 2
            or output.ndim != 2
            or len(hidden) != len(output)
            or len(hidden) < self.plan.token_count
        ):
            raise ValueError("Expected token-row hidden and residual buffers")
        if hidden.device != output.device or hidden.dtype != output.dtype:
            raise ValueError("Residual buffer must match hidden dtype and device")
        if (
            hidden.numel()
            and output.numel()
            and hidden.untyped_storage().data_ptr()
            == output.untyped_storage().data_ptr()
        ):
            raise ValueError("Residual buffer must not alias hidden states")
        output.zero_()
        self._run_prepared(layer, hidden, output, compute)
        self._layers.add(layer)
        return output

    def _run_prepared(self, layer, hidden, output, compute):
        for commit in (False, True):
            samples = [
                sample
                for sample in self.plan.samples
                if (sample.request.operation == "commit") == commit
                and (commit or self.worker_sessions.memory_read[sample.request.session])
            ]
            if not samples:
                continue
            records = [
                self.worker_sessions.requests[sample.request.rid] for sample in samples
            ]
            indices = torch.cat(
                [
                    torch.arange(sample.start, sample.stop, device=hidden.device)
                    for sample in samples
                ]
            )
            lengths = [sample.stop - sample.start for sample in samples]
            # The concatenated copy is the commit's initial state, so its final
            # state may overwrite the transaction slot in place afterwards.
            states = torch.cat(
                [
                    self.pool.storage[layer][request_record.slot]
                    for request_record in records
                ]
            )
            residual, final = compute.packed(
                hidden.index_select(0, indices),
                states,
                lengths=lengths,
                commit=commit,
                write_states=states,
            )
            if (
                residual.shape != output[indices].shape
                or residual.dtype != output.dtype
            ):
                raise RuntimeError("Packed memory output invariant violated")
            output.index_copy_(0, indices, residual)
            if commit:
                slots = torch.tensor(
                    [request_record.slot for request_record in records],
                    device=hidden.device,
                )
                self.pool.storage[layer].index_copy_(0, slots, final.unsqueeze(1))

    def finish(self, success: bool) -> None:
        if self.plan is None:
            raise SessionError("No active memory step")
        try:
            if not success:
                return
            if self.plan.samples and self._layers != self.pool.storage.keys():
                raise SessionError("Not all memory layers executed")
            for sample in self.plan.samples:
                if sample.request.operation == "commit":
                    self.worker_sessions.advance(
                        sample.request.rid, sample.stop - sample.start
                    )
        finally:
            self.plan = None
            self._layers.clear()
