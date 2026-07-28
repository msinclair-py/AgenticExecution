---
name: polaris-jobs
description: >-
  Submit and monitor jobs on the ALCF Polaris supercomputer (NVIDIA A100, PBS
  scheduler) from a laptop, by opening ONE persistent multiplexed SSH connection
  (authenticated once with an MFA token) and then running qsub / qstat over it
  non-interactively. Use when you need to launch a GPU job on Polaris, and
  especially when you need the correct mpiexec CPU/GPU affinity lines to add to a
  PBS script so the job actually uses all 4 GPUs per node. Covers login,
  connection reuse, the PBS script template, and the set_affinity_gpu_polaris.sh
  GPU-binding wrapper.
---

# Submitting jobs to ALCF Polaris

Polaris uses the **PBS Pro** scheduler. Each compute node has **64 CPU
hardware-threads (1× AMD EPYC "Milan")** and **4× NVIDIA A100 GPUs**. You submit
from a login node; jobs land on compute nodes.

The intended pattern here: open **one persistent SSH connection** to Polaris
(you authenticate interactively once with your MFA token), then submit and poll
jobs over that same connection **without re-authenticating** — so an agent /
reasoner can drive `qsub`, `qstat`, and file staging programmatically.

## Prerequisites

- An ALCF account, an active **project allocation** (the `-A <project>` value),
  and a **SafeNet MobilePASS+** or **CRYPTOCard** token — every fresh login to
  `polaris.alcf.anl.gov` needs a one-time passcode from it.
- `ssh` and `scp` locally.

## Step 1 — Open the persistent (multiplexed) SSH connection

ALCF requires an MFA passcode on **every** new login, which is why you cannot
script plain `ssh user@polaris ...` per command. SSH **connection multiplexing**
(ControlMaster) solves this: authenticate once, then reuse the open socket for
all later commands with no prompt.

Add to `~/.ssh/config` (once):

```
Host polaris
    HostName polaris.alcf.anl.gov
    User <your-alcf-username>
    ControlMaster auto
    ControlPath ~/.ssh/cm/%r@%h:%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Then:

```bash
mkdir -p ~/.ssh/cm
# Interactive: this ONE command prompts for your MFA passcode and opens the master.
ssh polaris "echo connected on \$(hostname)"
```

After that, every subsequent `ssh polaris <cmd>` and `scp ... polaris:...`
**reuses the master socket with no prompt**, until `ControlPersist` (8h here)
elapses or the ALCF session ends — then re-run the interactive command above to
re-establish it. Check / tear down the master:

```bash
ssh -O check polaris     # is the master alive?
ssh -O exit  polaris     # close it
```

An agent submitting jobs should treat "master down" as the one action needing a
human: prompt the user to run the interactive `ssh polaris ...` once, then resume
non-interactively.

## Step 2 — Stage files over the same connection

Run jobs from project/scratch space (e.g. `/eagle/<project>/...`), **not** from
`$HOME`. Copy your workload + PBS script up:

```bash
ssh polaris "mkdir -p /eagle/<project>/<run-dir>"
scp -r ./job/ polaris:/eagle/<project>/<run-dir>/
```

## Step 3 — The PBS script, with the affinity lines that make it work

This is the part jobs most often get wrong. Cray MPICH on Polaris **does not
bind MPI ranks to GPUs on its own**, so without a GPU-affinity wrapper every rank
piles onto GPU 0 and the other 3 A100s sit idle. Two pieces are required:

1. `mpiexec` **CPU** binding: `--ppn <ranks/node> --depth=<threads/rank>
   --cpu-bind depth`.
2. A **GPU** wrapper placed right before the executable that sets
   `CUDA_VISIBLE_DEVICES` per rank from `PMI_LOCAL_RANK`.

`set_affinity_gpu_polaris.sh` (put it next to your executable, `chmod +x`):

```bash
#!/bin/bash -l
num_gpus=4
# Assign in reverse order to respect the node's GPU/NUMA topology.
gpu=$(( (num_gpus - 1) - (PMI_LOCAL_RANK % num_gpus) ))
export CUDA_VISIBLE_DEVICES=$gpu
echo "RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}"
exec "$@"
```

`submit.sh` — one rank per A100 (the common layout for GPU work):

```bash
#!/bin/bash -l
#PBS -N bindify_job
#PBS -l select=1:ncpus=64
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A <project>

cd "$PBS_O_WORKDIR"

NNODES=$(wc -l < "$PBS_NODEFILE")
NRANKS_PER_NODE=4          # 1 MPI rank per A100
NDEPTH=8                   # CPU threads per rank
NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
export OMP_NUM_THREADS=$NDEPTH

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} \
    --cpu-bind depth \
    ./set_affinity_gpu_polaris.sh ./your_executable
```

For a **single-GPU, single-process** job (e.g. one folding/inference task), you
can skip MPI entirely and just run the program after selecting a GPU with
`export CUDA_VISIBLE_DEVICES=0`. Use the wrapper only when you launch multiple
ranks and want them spread across the 4 GPUs.

### Queues (pick with `-q`)

| Queue | Nodes | Walltime | Notes |
|---|---|---|---|
| `debug` | 1–2 | ≤1 h | fast turnaround for testing |
| `debug-scaling` | 1–10 | ≤1 h | 1 job/user |
| `prod` | 10–496 | ≤24 h | routing queue → small/medium/large |
| `preemptable` | 1–10 | ≤72 h | can be killed by `demand` |
| `capacity` | 1–4 | ≤168 h | long small jobs |

## Step 4 — Submit and capture the job id

```bash
JOBID=$(ssh polaris "cd /eagle/<project>/<run-dir> && qsub submit.sh")
echo "submitted $JOBID"
```

## Step 5 — Monitor and collect results (non-interactive, reuses the socket)

```bash
ssh polaris "qstat -x -u <your-alcf-username>"   # -x also shows finished jobs
ssh polaris "qstat -f $JOBID | grep job_state"    # single job status
# when done, pull stdout/stderr + outputs back down:
scp polaris:/eagle/<project>/<run-dir>/bindify_job.o* ./
scp -r polaris:/eagle/<project>/<run-dir>/outputs ./
```

An agent can poll `qstat` on an interval until the job leaves `R`/`Q`, then fetch
outputs. For large result sets prefer Globus over `scp`.

## Notes / gotchas

- **Every** login prompt = a token. The whole point of Step 1 is to pay that
  cost once; if you see repeated prompts, the master socket died — re-establish
  it interactively.
- Submit from project space (`/eagle`, `/grand`), not `$HOME`, and set
  `-l filesystems=` to list **every** filesystem the job touches. A job whose
  files live on `/grand/<project>/...` needs `-l filesystems=home:grand` (use
  `home:eagle` for `/eagle`); a mismatch gets the job rejected or fails I/O at
  runtime. (E.g. files under `/grand/<project>` need `-l filesystems=home:grand`.)
- **Compute nodes have NO outbound internet — only login nodes do.** Anything
  that would hit the network at runtime (HuggingFace model weights, `pip
  install`, dataset downloads) must be staged **on the login node first**, then
  read from cache in the job — otherwise the job hangs on a network timeout.
  For HuggingFace models: load the model once on a login node (this populates
  `~/.cache/huggingface`, which is on shared `home` and visible to compute
  nodes), then in the PBS script add `export HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1` so it loads from cache. (Verified with
  all-MiniLM-L6-v2: cached by the login-node run, the compute-node job ran fully
  offline.)
- `ncpus=64` requests the node's full 64 threads; `--depth` × `--ppn` should stay
  ≤ 64.
- This skill is about *driving* Polaris. What the job actually runs (a bindify
  tool, a parsl worker, an inference script) is up to you — wrap it as
  `your_executable` above.

Source: ALCF User Guides — Running Jobs on Polaris and Using GPUs on Polaris
(docs.alcf.anl.gov/polaris/).
