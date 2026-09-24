import json
import os
import socket
import sys
import time
import traceback

from alphagrad.approx.common.plan_log import jsonable

_COUNT = [0]


def job_log():
    for fd in (1, 2):
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if os.path.isfile(path):
            return path
    return None


def run_identity(args, config, wandb_run):
    return {
        "seed": int(args.seed),
        "commits": {k.split("/", 1)[1]: v for k, v in config.items()
                    if k.startswith("commit/")},
        "flags": dict(vars(args)),
        "wandb_id": None if wandb_run is None else wandb_run.id,
        "run_dir": os.getcwd() if wandb_run is None else wandb_run.dir,
    }


def _tag(value):
    return "na" if value is None else str(int(value))


def write(run, *, source, episode, env_index, plan, exception):
    log = job_log()
    folder = os.path.dirname(log) if log else run["run_dir"]
    job = os.environ.get("SLURM_JOB_ID") or "nojob"
    n = _COUNT[0]
    _COUNT[0] += 1
    path = os.path.join(
        folder, f"repro_{job}_ep{_tag(episode)}_env{_tag(env_index)}_{n}.json")
    bundle = {
        "source": source,
        "job_id": job,
        "node": socket.gethostname(),
        "pid": os.getpid(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "written_next_to": "job log" if log else "run directory",
        "job_log": log,
        "episode": episode,
        "env_index": env_index,
        "seed": run["seed"],
        "commits": run["commits"],
        "flags": run["flags"],
        "wandb_id": run["wandb_id"],
        "exception": exception,
        "plan": plan,
    }
    with open(path, "x") as fh:
        json.dump(jsonable(bundle), fh, indent=1, default=str)
    return path


def _raised(record):
    return str(record.get("refused") or "").startswith("raised:")


def write_refused(records, run):
    paths = []
    for rec in records:
        if not _raised(rec):
            continue
        paths.append(write(
            run(), source=f"measure actor {rec.get('actor')} pid {rec.get('pid')}",
            episode=rec.get("episode"), env_index=rec.get("env_index"),
            plan=rec,
            exception={
                "class": str(rec["refused"]).split(":", 1)[1],
                "message": None,
                "traceback": None,
                # The actor does not send the text; it prints it to the job log.
                "text_in_job_log": f"[SENTINEL-VERBOSE] lines of pid {rec.get('pid')}",
            }))
    return paths


def guard(fn, *, source, episode, run, records=()):
    def _guarded(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            try:
                plan = next((r for r in reversed(records) if _raised(r)), None)
                path = write(
                    run(), source=source, episode=episode,
                    env_index=None if plan is None else plan.get("env_index"),
                    plan=plan,
                    exception={
                        "class": type(exc).__name__,
                        "message": str(exc),
                        "traceback": "".join(traceback.format_exception(exc)),
                    })
            except Exception as err:
                print(f"[repro] {source} ep{episode} raised "
                      f"{type(exc).__name__}; the bundle was NOT written: "
                      f"{err!r}", file=sys.stderr, flush=True)
            else:
                print(f"[repro] {source} ep{episode} raised "
                      f"{type(exc).__name__}: bundle {path}", flush=True)
            raise
    return _guarded
