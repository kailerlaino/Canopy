"""
Sanity check for DataDumper dataset files.
Loads the same agent graph and checkpoint, re-runs dumped inputs,
and verifies outputs match the dumped values within numerical tolerance.

Requires: tensorflow, numpy, interval_bound_propagation, sonnet (same as agent_v2).
Run from rl-module or set PYTHONPATH so agent_v2 is importable.
"""

import argparse
import json
import os
import sys

import numpy as np

# Run from rl-module so agent_v2 is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)


def _parse_args():
    """Parse CLI args so --help works without importing TensorFlow or agent_v2."""
    parser = argparse.ArgumentParser(
        description="Sanity check DataDumper dataset: re-run inputs and compare outputs."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./dataset",
        help="Directory containing canopy_input.jsonl, canopy_output.jsonl, canopy_experience.jsonl",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory with agent checkpoint (train_dir / ckptdir used when dumping)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Path to params.json (default: rl-module/params.json)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap number of lines checked per file",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for np.allclose",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for np.allclose",
    )
    parser.add_argument(
        "--original_model",
        type=int,
        default=1,
        help="Use original actor: 1=yes, 0=no (default 1)",
    )
    parser.add_argument(
        "--snt_model_wo_ibp",
        type=int,
        default=0,
        help="Use snt model without IBP: 1=yes, 0=no (default 0)",
    )
    return parser.parse_args()


def load_params(params_path):
    """Load params from JSON; return dict-like with .dict property for compatibility."""
    try:
        with open(params_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        if os.path.getsize(params_path) == 0:
            raise ValueError(
                f"Params file is empty: {params_path}\n"
                "Generate it by running eval once (eval.sh creates it from params_base_eval.json), "
                "or pass a valid JSON file with --params (e.g. --params ../params_base_eval.json)."
            ) from e
        raise ValueError(f"Invalid JSON in params file {params_path}: {e}") from e
    if not data:
        raise ValueError(f"Params file is empty (no keys): {params_path}")
    return type("Params", (), {"dict": data})()


def infer_dims_from_data(dataset_dir):
    """
    Infer s_dim and a_dim from first records in JSONL files.
    Returns (s_dim, a_dim) or (None, None) if no data.
    """
    s_dim, a_dim = None, None
    for fname in ("canopy_input.jsonl", "canopy_output.jsonl"):
        path = os.path.join(dataset_dir, fname)
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            line = f.readline()
        if not line.strip():
            continue
        rec = json.loads(line)
        if "state" in rec:
            s_dim = len(rec["state"]) if isinstance(rec["state"], list) else None
        if "s0" in rec:
            s_dim = len(rec["s0"]) if isinstance(rec["s0"], list) else s_dim
        if "action" in rec:
            a_dim = len(rec["action"]) if isinstance(rec["action"], list) else None
        if s_dim is not None and a_dim is not None:
            break
    return s_dim, a_dim


def get_s_dim_a_dim(params, dataset_dir):
    """Get s_dim and a_dim: from params with recurrent support, or infer from data."""
    p = params.dict
    state_dim = p.get("state_dim")
    action_dim = p.get("action_dim")
    if state_dim is not None and action_dim is not None:
        s_dim = state_dim
        if p.get("recurrent"):
            s_dim = state_dim * p.get("rec_dim", 10)
        return s_dim, action_dim
    s_dim, a_dim = infer_dims_from_data(dataset_dir)
    if s_dim is None or a_dim is None:
        raise ValueError(
            "Could not infer s_dim/a_dim from params or dataset. "
            "Ensure params.json has state_dim/action_dim or dataset has at least one record."
        )
    return s_dim, a_dim


def check_actor(agent, input_path, max_samples, rtol, atol):
    """
    For each record in canopy_input.jsonl, run get_concrete_action(state, use_noise=False)
    and compare action_no_noise to dumped value.
    Returns (total_checked, num_failures, list of (index, max_diff) for failures).
    """
    if not os.path.isfile(input_path):
        return None, "file missing"
    failures = []
    total = 0
    with open(input_path) as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = np.array(rec["state"], dtype=np.float32)
            expected = np.array(rec["action_no_noise"], dtype=np.float32)
            # Agent expects batch: (1, s_dim); create_input_op_shape will reshape
            if state.ndim == 1:
                state = state[np.newaxis, :]
            try:
                _, action_no_noise = agent.get_concrete_action(state, use_noise=False)
            except Exception as e:
                failures.append((idx, None, str(e)))
                total += 1
                continue
            pred = np.array(action_no_noise).reshape(-1)
            if not np.allclose(pred, expected, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(pred - expected))
                failures.append((idx, max_diff, None))
            total += 1
    return (total, len(failures), failures)


def check_critic(agent, output_path, max_samples, rtol, atol):
    """
    For each record in canopy_output.jsonl, run get_q(state, action)
    and compare q_value to dumped value.
    Returns (total_checked, num_failures, list of failure info).
    """
    if not os.path.isfile(output_path):
        return None, "file missing"
    failures = []
    total = 0
    with open(output_path) as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = np.array(rec["state"], dtype=np.float32)
            action = np.array(rec["action"], dtype=np.float32)
            expected_q = float(rec["q_value"])
            if state.ndim == 1:
                state = state[np.newaxis, :]
            if action.ndim == 1:
                action = action[np.newaxis, :]
            try:
                q_out = agent.get_q(state, action)
            except Exception as e:
                failures.append((idx, None, str(e)))
                total += 1
                continue
            pred_q = np.squeeze(q_out[0])
            pred_q = float(pred_q)
            if not np.allclose(pred_q, expected_q, rtol=rtol, atol=atol):
                max_diff = abs(pred_q - expected_q)
                failures.append((idx, max_diff, None))
            total += 1
    return (total, len(failures), failures)


def check_experience_schema(experience_path, max_samples, s_dim, a_dim):
    """
    Validate canopy_experience.jsonl: required keys and shapes.
    Returns (total_checked, num_failures, list of (index, message)).
    """
    if not os.path.isfile(experience_path):
        return None, "file missing"
    required = {"s0", "action", "reward", "s1", "terminal"}
    failures = []
    total = 0
    with open(experience_path) as f:
        for idx, line in enumerate(f):
            if max_samples is not None and idx >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                failures.append((idx, f"invalid JSON: {e}"))
                total += 1
                continue
            missing = required - set(rec.keys())
            if missing:
                failures.append((idx, f"missing keys: {missing}"))
                total += 1
                continue
            if s_dim is not None:
                if len(rec["s0"]) != s_dim:
                    failures.append((idx, f"s0 length {len(rec['s0'])} != s_dim {s_dim}"))
                if len(rec["s1"]) != s_dim:
                    failures.append((idx, f"s1 length {len(rec['s1'])} != s_dim {s_dim}"))
            if a_dim is not None and len(rec["action"]) != a_dim:
                failures.append((idx, f"action length {len(rec['action'])} != a_dim {a_dim}"))
            total += 1
    return (total, len(failures), failures)


def main():
    args = _parse_args()

    # Defer TensorFlow and agent_v2 imports so --help works without them.
    # agent_v2 pulls in interval_bound_propagation and sonnet.
    try:
        import tensorflow as tf
        from agent_v2 import Agent
    except ImportError as e:
        print(
            "Missing dependency for sanity check. agent_v2 requires:\n"
            "  pip install tensorflow numpy interval_bound_propagation dm-sonnet\n"
            "Original error:",
            e,
            file=sys.stderr,
        )
        sys.exit(1)

    if args.checkpoint_dir is None:
        print("Error: --checkpoint_dir is required.", file=sys.stderr)
        sys.exit(1)

    params_path = args.params
    if not params_path:
        default_path = os.path.join(SCRIPT_DIR, "params.json")
        fallback_path = os.path.join(SCRIPT_DIR, "..", "params_base_eval.json")
        if os.path.isfile(default_path) and os.path.getsize(default_path) > 0:
            params_path = default_path
        elif os.path.isfile(fallback_path):
            params_path = os.path.normpath(fallback_path)
            print(f"Using fallback params: {params_path}", file=sys.stderr)
        else:
            params_path = default_path
    if not os.path.isfile(params_path):
        print(
            f"Error: params file not found: {params_path}\n"
            "Create rl-module/params.json (e.g. by running eval once) or pass --params path/to/params.json",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        params = load_params(params_path)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    p = params.dict

    dataset_dir = os.path.abspath(args.dataset_dir)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.isdir(dataset_dir):
        print(f"Error: dataset_dir is not a directory: {dataset_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(os.path.join(checkpoint_dir, "checkpoint")):
        print(
            f"Error: checkpoint not found in {checkpoint_dir} (no 'checkpoint' file)",
            file=sys.stderr,
        )
        sys.exit(1)

    s_dim, a_dim = get_s_dim_a_dim(params, dataset_dir)
    use_original = args.original_model == 1
    use_snt_model_wo_ibp = args.snt_model_wo_ibp == 1

    print(f"Using s_dim={s_dim}, a_dim={a_dim}, checkpoint_dir={checkpoint_dir}")
    print(f"use_original={use_original}, use_snt_model_wo_ibp={use_snt_model_wo_ibp}")

    with tf.Graph().as_default():
        agent = Agent(
            s_dim,
            a_dim,
            batch_size=p.get("batch_size", 512),
            summary=None,
            h1_shape=p.get("h1_shape", 256),
            h2_shape=p.get("h2_shape", 256),
            stddev=p.get("stddev", 0.2),
            mem_size=p.get("memsize", 55000),
            gamma=p.get("gamma", 0.995),
            lr_c=p.get("lr_c", 0.001),
            lr_a=p.get("lr_a", 0.0001),
            tau=p.get("tau", 0.001),
            PER=p.get("PER", False),
            CDQ=p.get("CDQ", True),
            LOSS_TYPE=p.get("LOSS_TYPE", "MSE"),
            noise_type=p.get("noise_type", 3),
            noise_exp=p.get("noise_exp", 50000),
            use_original=use_original,
            use_snt_model_wo_ibp=use_snt_model_wo_ibp,
        )
        agent.build_learn()
        agent.create_tf_summary()

        with tf.train.SingularMonitoredSession(checkpoint_dir=checkpoint_dir) as sess:
            agent.assign_sess(sess)

            input_path = os.path.join(dataset_dir, "canopy_input.jsonl")
            output_path = os.path.join(dataset_dir, "canopy_output.jsonl")
            experience_path = os.path.join(dataset_dir, "canopy_experience.jsonl")

            all_ok = True

            # Actor check
            result = check_actor(
                agent, input_path, args.max_samples, args.rtol, args.atol
            )
            if result[0] is None:
                print(f"Warning: skipping actor check ({result[1]}): {input_path}")
            else:
                total, nfail, failures = result
                if nfail > 0:
                    all_ok = False
                    print(f"Actor (canopy_input.jsonl): checked {total}, failures {nfail}")
                    for idx, max_diff, err in failures[:10]:
                        if err:
                            print(f"  index {idx}: {err}")
                        else:
                            print(f"  index {idx}: max_diff={max_diff}")
                    if len(failures) > 10:
                        print(f"  ... and {len(failures) - 10} more")
                else:
                    print(f"Actor (canopy_input.jsonl): checked {total}, all passed")

            # Critic check
            result = check_critic(
                agent, output_path, args.max_samples, args.rtol, args.atol
            )
            if result[0] is None:
                print(f"Warning: skipping critic check ({result[1]}): {output_path}")
            else:
                total, nfail, failures = result
                if nfail > 0:
                    all_ok = False
                    print(f"Critic (canopy_output.jsonl): checked {total}, failures {nfail}")
                    for idx, max_diff, err in failures[:10]:
                        if err:
                            print(f"  index {idx}: {err}")
                        else:
                            print(f"  index {idx}: max_diff={max_diff}")
                    if len(failures) > 10:
                        print(f"  ... and {len(failures) - 10} more")
                else:
                    print(f"Critic (canopy_output.jsonl): checked {total}, all passed")

            # Experience schema check
            result = check_experience_schema(
                experience_path, args.max_samples, s_dim, a_dim
            )
            if result[0] is None:
                print(f"Warning: skipping experience check ({result[1]}): {experience_path}")
            else:
                total, nfail, failures = result
                if nfail > 0:
                    all_ok = False
                    print(f"Experience (canopy_experience.jsonl): checked {total}, failures {nfail}")
                    for idx, msg in failures[:10]:
                        print(f"  index {idx}: {msg}")
                    if len(failures) > 10:
                        print(f"  ... and {len(failures) - 10} more")
                else:
                    print(f"Experience (canopy_experience.jsonl): checked {total}, all passed")

    print("--- Sanity check done ---")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
