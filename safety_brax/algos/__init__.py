from safety_brax.algos.base import BaseAlgorithm
from safety_brax.algos.ppo import PPO
from safety_brax.algos.pdo import PDO
from safety_brax.algos.bptt import BPTT
from safety_brax.algos.shac import SHAC
from safety_brax.algos.bptt_lag import BPTT_Lag
from safety_brax.algos.shac_lag import SHAC_Lag
from safety_brax.algos.diff_cpo import DiffCPO
from safety_brax.components import types


def create(
    algorithm: str,
    env: types.Env,
    config: dict,
    prng_key: types.PRNGKey,
):
    """Creates an Agent with a specified algorithm."""
    _str2class = {
        "ppo": PPO,
        "pdo": PDO,
        "bptt": BPTT,
        "shac": SHAC,
        "bptt_lag": BPTT_Lag,
        "shac_lag": SHAC_Lag,
        "diff_cpo": DiffCPO
    }
    return _str2class[algorithm](env, config, prng_key)
