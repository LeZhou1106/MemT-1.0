"""Shared memory forward lifecycle."""


class MemoryHooks:
    def forward(
        self, native_forward, input_ids, positions, forward_batch, *args, **kwargs
    ):
        runtime = getattr(self.model, "memt_runtime", None)
        if runtime is None:
            raise RuntimeError("MemT pool must be allocated before model forward")
        runtime.begin(forward_batch)
        success = False
        try:
            result = native_forward(
                input_ids, positions, forward_batch, *args, **kwargs
            )
            if self.pending:
                raise RuntimeError(
                    "Memory residual was not consumed by the native projection"
                )
            success = True
            return result
        finally:
            self.pending.clear()
            runtime.finish(success=success)
