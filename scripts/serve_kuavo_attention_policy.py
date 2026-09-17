"""Serve a DemoVLA checkpoint to Kuavo and save one attention panel per replan."""

from __future__ import annotations

import dataclasses
import logging

import tyro

from openpi.policies import policy_config
from openpi.serving import kuavo_attention_policy_server
from openpi.training import config as train_config


@dataclasses.dataclass
class Args:
    checkpoint_dir: str = "/home/dongxiaokun/checkpoints/xiaojianshangliao_dynamic_gate_v3_retry1/29999"
    config_name: str = "demovla_kuavo_right_dynamic_gate"
    output_dir: str = "outputs/kuavo_attention"
    default_prompt: str = "Pick and Place"
    host: str = "0.0.0.0"
    port: int = 5555
    top_k: int = 4
    api_token: str | None = None


def main(args: Args) -> None:
    policy = policy_config.create_trained_policy(
        train_config.get_config(args.config_name),
        args.checkpoint_dir,
        default_prompt=args.default_prompt,
        sample_kwargs={"interaction_diagnostics": True},
    )
    kuavo_policy = kuavo_attention_policy_server.KuavoAttentionPolicy(
        policy,
        args.output_dir,
        top_k=args.top_k,
    )
    server = kuavo_attention_policy_server.KuavoZmqPolicyServer(
        kuavo_policy,
        host=args.host,
        port=args.port,
        api_token=args.api_token,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
