# Archive

Non-functional debug/one-off scripts and configs, kept for historical
reference only — do not run.

- `old_scripts/` — one-off investigation/plotting scripts from early
  debugging sessions (gamma-plateau investigation, truth-vs-data checks,
  amplitude-boost/delta-m-ref experiments). Not maintained against the
  current `destacking`/`xlensing` APIs.
- `hisnr_*.yaml` — inference-only configs pointing `data_folder` at
  specific `gamma_scan_outputs/run_2026-0[5-7]-*` directories that were
  deleted from disk during the 2026-07-27 data-layout reorganization (see
  project history: real/sim data moved to `~/destacking_data/`). Running
  these will fail with a missing-directory error; the runs they refer to
  no longer exist and cannot be regenerated from this repo alone.
