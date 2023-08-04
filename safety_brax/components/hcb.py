import jax
import jax.numpy as jnp
from safety_brax import jumpy as jp
import optax
import jax.experimental.host_callback as hcb





def display(x):
    def host_fn(target,t):
        print(target)
    hcb.id_tap(host_fn, x)


def debug_print(x):
    jax.debug.print(x)


def proj(x, a, b):
    x_proj = jp.where(x >= a, x, a)
    x_proj = jp.where(x_proj <= b, x_proj, b)
    return x_proj

def get_flat_gradients_from(grad_dict):
    grades = []
    for k_params, v_params in grad_dict.items():  # 'params',{...}
        for k_layer, v_layer in v_params.items():  # 'hidden_0', {...}
            for name, item in v_layer.items():  # 'bias', {...}  and 'kernel', {...}
                grades.append(
                    item.reshape(-1, 1)
                )  # params list: bias-kernel-bias-kernel...
    return jnp.concatenate(
        grades
    )  # torch.cat  to get axis=0  return one array to compute
