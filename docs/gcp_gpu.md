# GCP GPU VMs

Choose **Run on → GCP · GPU VM**, then GPU and count. Local and Modal remain unchanged.

| GPU | Counts | Machine types |
| --- | --- | --- |
| L4 (24 GB per GPU) | 1, 2, 4, 8 | g2-standard-4 / 24 / 48 / 96 |
| A100 (40 GB per GPU) | 1, 2, 4, 8, 16 | a2-highgpu-1g / 2g / 4g / 8g; a2-megagpu-16g |

Mappings follow [Google's accelerator-optimized machine table](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines).
Capacity, region support, quotas and pricing vary. Selecting a count provisions **one VM with that many GPUs**, not multiple VMs.

## Multi-GPU semantics

Cell/seed trials run concurrently in separate processes, each pinned to one GPU with `CUDA_VISIBLE_DEVICES`. Folds remain sequential within each trial. Results are merged into the existing CSV, summary and forecast-diagnostics formats. Final-test refits use the original validation runs for epoch selection.

This is experiment parallelism, **not** distributed training or pooled GPU memory. A single model must fit on one GPU. Quick's default single cell/seed only uses one GPU; selecting more still bills the whole VM. Use multiple seeds/cells to utilise more GPUs. No live per-epoch progress is streamed yet; logs/results download on completion.

## One-time operator setup

This integration intentionally does not create projects, enable billing, grant IAM roles, build images, or create networks/buckets automatically.

1. Install a current Google Cloud CLI locally and authenticate with `gcloud auth login` (or your organisation's supported CLI identity).
2. Use a billed project with Compute Engine and Cloud Storage APIs enabled and sufficient GPU quota in the selected zone.
3. Supply an existing private bucket. Grant the local identity object read/create/list/delete access scoped to this bucket, VM create/get/delete and disk permissions in the project, image access, subnet use, and permission to act as the selected VM service account. Have your cloud administrator scope these permissions; do not use Owner credentials just for this feature.
4. Use a dedicated VM service account with only bucket object read/create permissions needed for payload/results. No credential files are uploaded: the VM uses its attached identity. The `cloud-platform` access scope does not itself grant IAM permissions.
5. Supply a subnet in the zone's region with **Private Google Access** (or suitable NAT/routing) for Cloud Storage. VMs have no external IP and need no inbound SSH/firewall changes.
6. Supply a **prepared Linux image**, pinned by image name, with the Google guest agent/startup-script support, `bash`, `gcloud`, and `python3` available to root. That Python must have CUDA-enabled PyTorch, all runtime dependencies from `pyproject.toml`, and a compatible NVIDIA driver for both chosen GPU families. Test `python3 -c 'import torch; print(torch.cuda.is_available())'` on a separately authorised image validation VM. A bare Ubuntu image is not sufficient. No packages or drivers are installed automatically during a job.

Set these environment variables in the shell that starts the playground (replace all examples):

```sh
export HYPERCAST_GCP_PROJECT=my-project
export HYPERCAST_GCP_ZONE=us-central1-a
export HYPERCAST_GCP_BUCKET=my-private-job-bucket
export HYPERCAST_GCP_SERVICE_ACCOUNT=hypercast-worker@my-project.iam.gserviceaccount.com
export HYPERCAST_GCP_IMAGE=my-tested-pytorch-gpu-image
export HYPERCAST_GCP_IMAGE_PROJECT=my-project
export HYPERCAST_GCP_SUBNET=my-private-subnet
export HYPERCAST_GCP_MAX_HOURS=6
```

The runtime limit defaults to 6 hours and accepts 1–24. “Configured” checks configuration and an active CLI identity, not quota, image compatibility, IAM or capacity. Those may still fail at launch. Account/project values and credentials cannot be supplied through a submitted architecture.

## Costs, data and cleanup

Pressing Run uploads the package's Python source, selected dataset, request, and (for final tests) parent validation CSV to a random job-specific bucket prefix. It creates one on-demand GPU VM with a 100 GB auto-delete boot disk. These resources and data transfers incur charges.

On completion, failure or cancellation the local controller attempts to delete that VM and its exact storage prefix. Cleanup warnings are persisted in job status and shown in Runs. Requests never delete the bucket, network, image or unrelated VMs.

A provider-side [maximum runtime with DELETE action](https://docs.cloud.google.com/compute/docs/instances/limit-vm-runtime) is the fallback if the local process disappears. This is not a billing guarantee: quota, provider failures, manual stops and lost permissions can prevent cleanup. Do not manually stop/restart these VMs; stopping resets scheduling semantics. Configure a bucket lifecycle policy for old `hypercast4d/` objects, billing alerts, and monitor resources labelled `app=hypercast4d`. A killed controller cannot clean its storage prefix automatically; consult `gcp_instance`, `gcp_project`, `gcp_zone`, `gcp_prefix` and `gcp_cleanup` in persisted status. Bucket soft-delete/versioning/retention may retain billed copies after deletion.

On bootstrap failures inspect the downloaded bootstrap log, or VM serial logs while the VM exists. Failed jobs do not unlock final-test evaluation. This feature is tested with mocked cloud commands and CPU-backed worker tests; a real billable GPU smoke test is required before production use.
