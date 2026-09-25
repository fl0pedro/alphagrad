from alphagrad.approx import ppo
from alphagrad.approx.env import REWARD_INDEX


# dsnn-dfw.112: each value head must read the reward slot that its name says.
def test_the_first_three_value_heads_read_latency_memory_and_the_gradient_cosine():
    assert tuple(ppo.HEAD_REWARD_INDICES[:3]) == (
        REWARD_INDEX["latency_ns"],
        REWARD_INDEX["peak_memory"],
        REWARD_INDEX["cosine_sim"],
    )


def test_each_head_name_names_the_slot_it_reads():
    assert tuple(ppo.HEAD_NAMES[:3]) == ("latency", "mem", "quality")
    reads = dict(zip(ppo.HEAD_NAMES, ppo.HEAD_REWARD_INDICES))
    assert reads["latency"] == REWARD_INDEX["latency_ns"]
    assert reads["mem"] == REWARD_INDEX["peak_memory"]
    assert reads["quality"] == REWARD_INDEX["cosine_sim"]


def test_no_value_head_reads_flops_or_the_frobenius_slot():
    assert REWARD_INDEX["flops"] not in ppo.HEAD_REWARD_INDICES
    assert REWARD_INDEX["frob_residual"] not in ppo.HEAD_REWARD_INDICES
