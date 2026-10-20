"""Projection hooks preserve native attention, GDN pools and TP collectives."""

from memt_sglang.models.common.hooks import MemoryHooks


class QwenMemoryHooks(MemoryHooks):
    def __init__(self, model, blocks):
        self.model = model
        self.pending = {}
        self.handles = []
        layers = model.model.layers
        for index, block in blocks.items():
            layer = layers[index]
            if (
                not 0 <= layer.attn_tp_rank < layer.attn_tp_size
                or layer.num_heads * layer.attn_tp_size != block.num_heads
                or layer.head_dim != block.head_dim
                or hasattr(layer, "memt")
            ):
                raise ValueError(
                    "Qwen memory requires matching full attention head partition"
                )
            weight = layer.o_proj.weight
            parameter = next(block.parameters())
            if parameter.dtype != weight.dtype or parameter.device != weight.device:
                raise ValueError(
                    "Qwen native projection and memory must share dtype/device"
                )
        for index, block in blocks.items():
            layer = layers[index]
            layer.add_module("memt", block)
            # At v0.5.16 both native classes own qkv_proj/o_proj directly in
            # the full-attention decoder. qkv input is normalized hidden;
            # o_proj input already includes the native sigmoid attention gate.
            self.handles.append(
                layer.qkv_proj.register_forward_pre_hook(
                    self._prepare(index, layer), with_kwargs=True
                )
            )
            self.handles.append(
                layer.o_proj.register_forward_pre_hook(self._inject(index))
            )

    def _prepare(self, index, layer):
        def hook(module, args, kwargs):
            runtime = getattr(self.model, "memt_runtime", None)
            if runtime is None or runtime.plan is None or not runtime.plan.samples:
                return
            hidden = args[0] if args else kwargs["input_"]
            output = hidden.new_empty((len(hidden), layer.memt.value_dim))
            residual = runtime.run_layer(index, hidden, output, layer.memt)
            # Each rank computes with its own full-head state, then slices.
            heads = residual.view(len(hidden), layer.memt.num_heads, layer.head_dim)
            start = layer.attn_tp_rank * layer.num_heads
            self.pending[index] = heads[:, start : start + layer.num_heads].reshape(
                len(hidden), -1
            )

        return hook

    def _inject(self, index):
        def hook(module, args):
            residual = self.pending.pop(index, None)
            return args if residual is None else (args[0] + residual, *args[1:])

        return hook
