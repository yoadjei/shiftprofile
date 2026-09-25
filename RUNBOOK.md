# Shiftprofile Pilot: Execution Runbook

**This runbook is your guide to running the pilot evaluation on Kaggle.**

The pilot is designed to complete in one session (12 hours on T4) with the budget set to 660 minutes, leaving 60 minutes of headroom for saving the cache before session preemption.

---

## One-Time Setup (Account + Datasets)

These steps are performed **once**. After this, you can run the pilot notebooks any number of times.

### Step 1: Create a Kaggle Account

- Go to [kaggle.com](https://kaggle.com) and create a free account, and verify your email.
- Turn on **phone verification** (Settings → Phone Verification). Without it Kaggle will not give a
  notebook GPU or internet access, and both are required here.
- You do **not** need a Kaggle API token for this workflow. The token is only for driving Kaggle from
  your own machine with the `kaggle` CLI; everything below runs inside Kaggle's own notebooks.

> **On credentials generally.** If you ever paste an API token into a chat, a notebook cell, or a commit,
> treat it as public and rotate it immediately (Settings → Account → Expire API Token). Saved Kaggle
> notebook versions retain their source, so a token pasted into a cell is a published token. For anything
> secret, use **Add-ons → Secrets**, which keeps the value out of the notebook source.
>
> For reference, the classic Kaggle CLI reads `~/.kaggle/kaggle.json` containing
> `{"username": "...", "key": "..."}` — not a bare token file — and needs `chmod 600`.

### Step 2: Put the code on GitHub

The code lives in a **GitHub repository**, not a Kaggle Dataset. Each notebook clones it in its first
cell. This is the better route during a pilot because the code changes often and a `git push` is
instant, whereas a Kaggle Dataset would need re-uploading and re-versioning every time.

1. Create a repository (public or private) named `shiftprofile`.
2. Push this project to it, including the `shiftprofile/`, `configs/`, `tests/` and `notebooks/`
   directories and `pyproject.toml`. There is no `setup.py`; `pyproject.toml` is the only build file.
3. Note your username, the repo name, and the branch (`main`).
4. **If the repository is private**, create a GitHub fine-grained personal access token with
   *Contents: Read-only* scope, then in each Kaggle notebook go to **Add-ons → Secrets**, add a secret
   named `GITHUB_TOKEN` with that value, and set `USE_SECRET = True` in the bootstrap cell.
   Never paste the token into a cell.

Every notebook starts with the same bootstrap cell. Edit the four lines at the top of it:

```python
GITHUB_USER = "yoadjei"
GITHUB_REPO = "shiftprofile"
BRANCH      = "main"
USE_SECRET  = False   # True if the repo is private
```

It clones (or fast-forwards) the repo, `pip install -e`s it, and prints the **commit SHA**. Record that
SHA: every result record stores the code version that produced it, which is what makes a number traceable
back to the code months later.

### Step 3: Create the cache Dataset (initially empty)

1. Create a **new blank dataset**:
   - Name: `shiftprofile-cache`
   - Description: "Pilot evaluation artifacts cache"
   - Visibility: Private
2. Create a dummy file (e.g., `.gitkeep`) and upload it
3. Publish the dataset and note its **slug** (e.g., `yoadjei/shiftprofile-cache`)

Do NOT upload large files to this dataset yet — only create it. The cache will be populated by notebook 01.

### Step 4: Prepare CIFAR-10-C Dataset

Choose one of these two paths:

#### Option A (Recommended): Upload CIFAR-10-C to Kaggle

Either layout works. `load_cifar10c` accepts the corruption arrays nested in a
`CIFAR-10-C/` folder (as the Zenodo tar extracts them) or sitting flat at the top of the
dataset, and says which paths it tried if it finds neither. Upload whichever is less
effort.

What it will **not** tolerate is a missing `labels.npy`. Every corruption is scored
against it, so leaving it out breaks every cell.

1. Download CIFAR-10-C from [Zenodo](https://zenodo.org/record/2535967) and extract it.
   The tar produces a folder named `CIFAR-10-C`.
2. Delete everything from that folder except the corruptions Version A uses, plus the
   labels. Each file is ~147 MB, so this takes the upload from ~2.8 GB to ~735 MB:
   ```
   gaussian_noise.npy  shot_noise.npy  defocus_blur.npy
   fog.npy  jpeg_compression.npy  labels.npy
   ```
   The pilot only reads `gaussian_noise`, `defocus_blur` and `fog`, but uploading the
   five now avoids a second upload when the pilot passes. **Keep `labels.npy`** — every
   corruption is scored against it.
3. Create a **new Kaggle dataset**:
   - Name: `cifar-10-c`
   - Visibility: Private
   - Drag in the `CIFAR-10-C` folder, or its six files directly
4. Publish and note the slug (e.g. `yoadjei/cifar-10-c`)

#### Option B (Fallback): Download in notebook

- Skip this step
- Notebook 00 has a Zenodo download cell (runs once, takes 5-10 minutes)
- After notebook 01, save the `/kaggle/working/` directory as a new Dataset version

---

## Per-Session Execution

Each session follows this order: Setup → Fill → Report → Save.

### Session 1: Setup and Run Pilot Fill

#### Step 1: Create a Kaggle notebook

1. Go to Kaggle Notebooks → **New Notebook** (Python kernel). Name it "Pilot 00".
2. Open the right-hand settings panel and set:
   - **Accelerator: GPU T4 x2** (or P100). Notebook 00 does not need it, but 01 does.
   - **Internet: On.** This is easy to miss and the bootstrap cell fails without it — `git clone`
     cannot reach GitHub, and torchvision cannot download CIFAR-10. It requires phone verification.
3. Upload the notebook file directly: **File → Import Notebook →** pick
   `notebooks/00_setup_and_data.ipynb` from your machine. There is no need to paste cells.

#### Step 2: Attach datasets

Click **Add Input** in the right panel and attach:

- `cifar-10-c` — if you used Option A above
- `shiftprofile-cache` — empty on the first run; attach it anyway so the path exists

You do **not** attach the code: the bootstrap cell clones it from GitHub. Note the mount paths, normally
`/kaggle/input/cifar-10-c` and `/kaggle/input/shiftprofile-cache`.

#### Step 3: Run notebook 00

1. Edit the four lines at the top of the bootstrap cell (`GITHUB_USER` is already `yoadjei`).
2. Run all cells and check each prints what it should:
   - Bootstrap: `shiftprofile ready at ...` and a **commit SHA** — write the SHA down
   - GPU: `CUDA available: True`
   - CIFAR-10 and CIFAR-10-C resolve
   - **Alignment: clean/corrupted verified** — this is the one that matters most. A misalignment
     would silently invalidate every paired comparison in the study, and it costs seconds to check.
   - Evaluation indices checksum
   - Cache directory created

If any cell fails, **stop and fix it here.** Everything downstream is more expensive.

#### Step 4: Create the pilot-fill notebook

1. New notebook "Pilot 01", same settings (**GPU on, Internet on**).
2. Attach the same two datasets.
3. **File → Import Notebook →** `notebooks/01_pilot_fill.ipynb`.
4. Set the same four bootstrap lines, and check `cache_read` matches your cache mount path.

#### Step 5: Run notebook 01

1. Run all cells in order
2. The timing estimate (Step 3) will tell you if the budget is safe
   - If `> 600 minutes predicted`, the budget is tight — stop and wait for GPU time in a later session
   - If `< 600 minutes predicted`, proceed
3. The fill pipeline (Step 4) will run for up to 660 minutes
   - Watch the progress output
   - Each stage (predict, explain, curves) prints completions and failures
   - If you see `budget_exhausted: True`, that is expected and correct
4. When complete, the FillReport summary will show:
   - `Completed X cells` — cells that finished
   - `Skipped Y cached` — cells already in the cache (re-runs skip these)
   - `Failed Z cells` — if any errors occurred
   - `Elapsed time` — wall-clock seconds

#### Step 6: Save the cache

1. In notebook 01 Step 5, follow the save instructions:
   - Click the orange "Save Version" button (top right)
   - Type a description: "Pilot cache — [stage name] complete"
   - Save as a new version (NOT as a Dataset creation)
2. Copy the new **Dataset ID** from the output panel
3. This ID will be used to attach the cache in the next session

### Session 2+: Resume From Cache

If your session dies or you need to resume:

1. Create a new Kaggle notebook
2. Attach datasets:
   - `cifar-10-c` (data)
   - **The newest cache dataset version** (click "Add version" and select the latest)
3. Import `notebooks/01_pilot_fill.ipynb` again — the bootstrap cell re-clones the code, so any fixes
   you have pushed to GitHub since the last session are picked up automatically
4. Check the `cache_read` path matches the new cache mount
5. Run it — every cell already in the cache is skipped and the fill resumes where it stopped
6. Save the cache again (Step 6 above)
7. Repeat until `budget_exhausted: False` and no cells remain

This resume loop is the whole point of the design: a session that dies costs you only the cell that was
in flight, not the session.

### Final Session: Pilot Report

Once notebook 01 completes successfully:

1. Create a new Kaggle notebook "Pilot 02" — **no GPU needed**, so this costs nothing from your quota.
   You can equally run it on your own machine: the notebook only reads the cache.
2. Attach `cifar-10-c` and **the final cache dataset** from the last session.
3. **File → Import Notebook →** `notebooks/02_pilot_report.ipynb`, then run all cells.
4. Read the **P3 Gate Decision**:
   - **PASS** — the faithfulness interval half-width is under 10% of the clean-to-severity-5 change.
     The measurement can carry the claim. Proceed to the Version A grid (`configs/version_a.yaml`,
     240 cells, roughly one week of quota).
   - **FAIL** — the intervals are wider than the effect. Do **not** simply run more cells hoping the
     noise averages out. The pre-registered response is to change the faithfulness metric, or to pivot
     to the measurement-validity framing, in which "current faithfulness metrics cannot support claims
     under shift" is itself the result. See `prereg/preregistration.md` §8.
5. Send the gate output, the measured wall-clock and the FillReport summary back for the scope decision.
   The wall-clock matters as much as the verdict: it replaces planning estimates with measured cost.

---

## Budget and Quota Arithmetic

**Kaggle free tier**: 30 GPU-hours per week (cumulative across all notebooks)

**Pilot workload**: ~2 GPU-hours on T4

- 1 model × 1 seed × (1 clean + 3 corruptions × 3 severities) = 10 cells
- Predict (~30s per cell) + Explain (~120s per cell × 3 explainers) + Curves (~60s per cell) = ~3500s total
- ≈ 60 minutes per 1000 evaluation samples
- With 1000 eval samples (pilot) = ~1 hour core GPU time
- Headroom for training, I/O, session warmup = ~2 hours total

**Budget in notebooks**: 660 minutes (11 hours)
- Leaves 60 minutes for session save before 12-hour preemption

**Headroom**: the pilot is ~2 of your 30 weekly GPU-hours, so roughly 7% of one week. There is no need
to ration it — if a run fails, re-running costs little, and the cache means a re-run repeats only what
was lost. The quota only becomes a real constraint at the Version A grid (~20 GPU-h, about one week)
and the Version B extension (~45 GPU-h, two to three weeks of accumulated quota).

**Do not** leave a GPU session idle. Kaggle bills wall-clock while the accelerator is attached, not
compute actually used, so an idle open notebook quietly spends the quota the full grid needs.

---

## Troubleshooting

### Problem: CUDA not available

**Cause**: You are not on a GPU kernel

**Fix**:
1. Create a new notebook
2. Go to the kernel selector (top right)
3. Select "GPU" and confirm the T4 GPU is available
4. Delete the old notebook

### Problem: CIFAR-10-C path not found

**Cause**: Dataset is not attached or the path is wrong

**Fix**:
1. Go to "Add input" in the notebook
2. Search for "cifar-10-c" and attach it
3. Update the `cifar10c_root` path in notebook 00 Step 5 to match the mounted location
4. Re-run notebook 00

### Problem: Cache Dataset not found

**Cause**: First run, or the Dataset ID has changed

**Fix**:
1. On first run, this is expected — create the dataset in setup
2. After notebook 01, save the cache as a new Dataset version (Step 6)
3. In the next session, attach the **new version** (not the old empty one)
4. Update the `cache_read` path in notebook 01 Step 2

### Problem: the bootstrap cell fails — `git clone` cannot reach GitHub

**Cause**: Kaggle notebooks have **Internet switched off by default**. This is the single most common
first-run failure with the GitHub workflow, and the error can look like a DNS or TLS problem rather than
a settings problem.

**Fix**:
1. Open the right-hand settings panel → **Internet: On**
2. This requires phone verification on your Kaggle account (Settings → Phone Verification)
3. Re-run the bootstrap cell

### Problem: the bootstrap cell fails with `Authentication failed` or `Repository not found`

**Cause**: the repo is private and the notebook has no credential, or the token lacks access.

**Fix**:
1. Confirm `GITHUB_USER`, `GITHUB_REPO` and `BRANCH` are spelled correctly — a wrong branch reports as
   "Remote branch not found", not as an auth error
2. For a private repo: create a GitHub **fine-grained** PAT with *Contents: Read-only* on that repo,
   add it under **Add-ons → Secrets** as `GITHUB_TOKEN`, and set `USE_SECRET = True`
3. Never paste the token into a cell. If you already have, rotate it on GitHub immediately — a saved
   notebook version keeps its source

### Problem: `import shiftprofile` fails right after the bootstrap

**Cause**: `pip install -e` does not always register in an already-running kernel.

**Fix**: the bootstrap cell already inserts the repo on `sys.path` to cover this. If it still fails,
restart the kernel (Run → Restart) and re-run the cell — the clone is preserved.

### Problem: Out of memory during training or prediction

**Cause**: Batch size is too large for this GPU

**Fix**:
1. In notebook 01 Step 1, lower the `batch_size` in the config
2. The pilot uses ResNet-18, which is small — this is unlikely to happen on T4

### Problem: Session dies mid-run

**This is expected on Kaggle.** Sessions are preempted randomly.

**Fix**:
1. Go to notebook 01 and save the current cache as a Dataset version
2. Start a new session
3. Attach the new cache version
4. Re-run notebook 01 — it will resume from cached work

### Problem: Notebook 02 shows "INCOMPLETE" for the P3 gate

**Cause**: Notebook 01 did not reach the explain or curves stages

**Fix**:
1. Check the FillReport in notebook 01 Step 4
   - If `budget_exhausted: True` and `failed` is empty, this is expected
   - The workload was too large for one session
   - Use budget to resume in the next session
2. If there are failures (`failed: [...]`), check the error messages
   - Most failures are out-of-memory or timeout; these are transient
   - Re-run notebook 01 to retry

---

## Success Criteria

- **Notebook 00**: All cells print success, alignment verified, cache created
- **Notebook 01**: FillReport shows cells completed/skipped, no critical failures
- **Notebook 02**: P3 Gate decision is PASS or FAIL (not INCOMPLETE)

---

## Manual Steps That Cannot Be Automated

These require you, in a browser. Nothing in this repository can do them.

1. **Create the Kaggle account**, verify email, and complete **phone verification** — without it there
   is no GPU and no internet
2. **Push the code to GitHub** and, if the repo is private, create the PAT and add it as a Kaggle Secret
3. **Create the `shiftprofile-cache` Dataset** and the `cifar-10-c` Dataset (or run the Zenodo fallback)
4. **Turn Internet On and select a GPU** in each notebook's settings panel
5. **Attach the datasets** via "Add Input"
6. **Click "Save Version"** after notebook 01 — this is what persists the cache between sessions, and
   skipping it loses the session's GPU work entirely
7. **Select the newest cache version** when resuming

Items 4 and 6 are the two that most often go wrong: the first stops the run before it starts, the second
silently throws away everything the run produced.

---

## Questions?

Refer to the notebook markdown cells for more detail on each step. Each notebook cell prints what it is doing and what succeeded.

Good luck with the pilot!
