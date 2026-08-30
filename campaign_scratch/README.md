# campaign_scratch/

Verbatim archive of the 2026-08 agent-campaign scripts (a1b, a2, a3, a5-a8, ab,
apxaudit, aux, bwd, compress, decode..decode6, diag, lean, perf2, pfm, poolgate,
probe, s2/s3, tlm3, vprobe) that produced the results committed on this branch.
They lived at the repo root through `d904cc2`; ticket 27 moved them here so the
tree has one place to look for the provenance of a committed number.

**Contents are preserved byte-for-byte as they were run.** They are a record of
how the jobs were actually invoked, not maintained code, so nothing inside this
directory has been rewritten -- including the paths in it.

Reading the paths inside these files:

* Every bare filename an `.sbatch` or `.sh` here invokes (`python decode3_data.py`,
  `python pfm_measure.py`, ...) is **relative to the repo root as of `d904cc2`**,
  because each script `cd`s to the repo root first. Those siblings all moved here
  together, so today the same reference resolves to `campaign_scratch/<name>`.
* Six of these files were already tracked at the repo root in `d904cc2`
  (`apxaudit_matrix2.py`, `decode2_face_data.py`, `decode3_data.py`,
  `decode5_summary.py`, `decode6_summary.py`, `pfm_measure.py`). They moved with
  their untracked siblings for the same reason. Citations to them from maintained
  source outside this directory were updated to the `campaign_scratch/` path.
* Job outputs, compile caches and `*_out/` dumps these scripts wrote are NOT here;
  they remain ignored at the repo root (see `.gitignore`). The one exception is
  `decode2_out/face_today_sizes.log`, which is tracked at the repo root because
  `src/alphagrad/approx/common/feature_probe.py:72` cites it as the provenance of
  its numbers.

Nothing here is expected to run as-is today.
