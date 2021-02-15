import torch
from torch.utils.checkpoint import (
    check_backward_validity, get_device_states, set_device_states, detach_variable
)


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(*args)
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        # return outputs

        #
        # Lie to torch we have no None items, to avoid the assert
        #
        result = []
        for o in outputs:
            if o is None:
                o = torch.zeros(0)
            result.append(o)

        return tuple(result)

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        #
        # Skip None items and tensors which requires_grad are False when doing backward pass
        #
        backward_outputs = []
        backward_args = []
        for o, a in zip(outputs, args):
            if o is not None and o.requires_grad:
                backward_outputs.append(o)
                backward_args.append(a)
        torch.autograd.backward(backward_outputs, backward_args)

        # torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                      for inp in detached_inputs)
        return (None, None) + grads


def checkpoint(function, *args, **kwargs):
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    outputs = CheckpointFunction.apply(function, preserve, *args)

    #
    # Resotre None items to result
    #
    result = []
    for o in outputs:
        if len(o) == 0:
            o = None
        result.append(o)

    return tuple(result)
