# GPU Runners for vLLM Recording

This guide explains how to use GPU-enabled self-hosted runners to re-record vLLM integration tests with larger models like `gpt-oss:20b`.

## Overview

GPU runners allow us to:

- Test larger models (20B parameters) that don't fit on CPU runners
- Faster inference with GPU acceleration
- More realistic production-like test environment
- On-demand re-recording via workflow_dispatch

**Cost**: ~$0.43 per run (30 min on g6.2xlarge), ~$1.72/month for weekly runs

## Quick Start

### Trigger a GPU Recording Run

1. Go to **Actions** tab in GitHub
2. Select **vLLM GPU Recording** workflow
3. Click **Run workflow**
4. Configure:
   - **Suite**: `base` (default)
5. Click **Run workflow**

The workflow will:

1. Launch a GPU EC2 instance (5 min)
2. Set up the CUDA environment and install the pinned vLLM runtime (5 min)
3. Run tests in record mode (~20 min)
4. Upload recordings as artifacts
5. Terminate the EC2 instance

**Total time**: ~30 minutes

### Download Recordings

1. Wait for the workflow to complete
2. Go to the workflow run summary
3. Download the `vllm-gpu-recordings-*` artifact
4. Extract and commit the recordings to your PR

## Architecture

```text
┌─────────────────────────────────────────────────┐
│  Workflow Trigger (manual)                      │
│  - Select test suite                            │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Job 1: Start GPU EC2 Runner                    │
│  - AWS OIDC authentication (no long-lived keys!)│
│  - Multi-AZ fallback in us-east-2               │
│  - Launch g6.2xlarge with GPU AMI               │
│  - Register as GitHub Actions runner            │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Job 2: Run vLLM Recording Tests                │
│  - Runs on GPU runner (permissions: {})         │
│  - Install vLLM with CUDA support               │
│  - Start vLLM server with gpt-oss:20b           │
│  - Run integration tests in record mode         │
│  - Upload recordings as artifacts               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Job 3: Stop GPU EC2 Runner                     │
│  - Wait for the GPU job or queued-job timeout   │
│  - Terminate instance                           │
└─────────────────────────────────────────────────┘
```

## AWS Prerequisites

### Required AWS Resources

You must set up the following in AWS before using GPU runners:

#### 1. IAM Role for OIDC Authentication

Create an IAM role that GitHub Actions can assume via OIDC:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:YOUR_ORG/llama-stack:*"
        }
      }
    }
  ]
}
```

Attach this policy to the role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:CreateTags",
        "ec2:DescribeImages",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": ["us-east-2"]
        }
      }
    }
  ]
}
```

#### 2. VPC and Subnets

You need subnets in `us-east-2` for the first version:

**us-east-2 (Primary)**:

- us-east-2a: `subnet-02d230cffd9385bd4`
- us-east-2b: `subnet-024298cefa3bedd61`
- us-east-2c: `subnet-04701a08396b2ed01`

#### 3. Security Groups

Create a security group in `us-east-2` with:

**Inbound Rules**:

- None (runners connect outbound only)

**Outbound Rules**:

- Port 443 (HTTPS): `0.0.0.0/0` - GitHub API, HuggingFace, PyPI
- Port 80 (HTTP): `0.0.0.0/0` - Package downloads

#### 4. GPU-Enabled AMI

Use a GPU-capable AMI in `us-east-2` with:

- Base OS: RHEL 9
- NVIDIA drivers
- CUDA 13.0 runtime
- Docker with NVIDIA Container Toolkit
- Python 3.12

The current DevOps AMI is `ami-090a815de2a7461f2`
(`vllm-rhel9-nvidia-ami-1781031435`). It includes NVIDIA drivers, CUDA 13.0,
and vLLM 0.22.1 from the image build. The workflow still creates `/tmp/vllm-env`
and installs the pinned vLLM version there during each run, so the AMI supplies
the GPU driver/CUDA base while the action controls the Python runtime used by
the tests.

### GitHub Configuration

#### Secrets

Add these to **Settings > Secrets and variables > Actions > Secrets**:

- `AWS_ROLE_ARN`: ARN of the IAM role for OIDC (e.g., `arn:aws:iam::123456789012:role/GitHubActionsRole`)
- `RELEASE_PAT`: GitHub Personal Access Token with `repo` scope

#### Variables

Add these to **Settings > Secrets and variables > Actions > Variables**:

**us-east-2**:

- `SUBNET_US_EAST_2A`: `subnet-02d230cffd9385bd4`
- `SUBNET_US_EAST_2B`: `subnet-024298cefa3bedd61`
- `SUBNET_US_EAST_2C`: `subnet-04701a08396b2ed01`
- `AWS_EC2_AMI_US_EAST_2`: `ami-090a815de2a7461f2`
- `SECURITY_GROUP_ID_US_EAST_2`: `sg-06300447c4a5fbef3`

## Security

### OIDC Authentication

We use **OpenID Connect (OIDC)** to authenticate with AWS instead of long-lived access keys:

- ✅ No static AWS credentials stored in GitHub
- ✅ Automatic token rotation
- ✅ Fine-grained permissions per workflow
- ✅ Better audit trail in AWS CloudTrail

The workflow requests temporary credentials from AWS STS using OIDC tokens from GitHub.

### Test Job Isolation

The test job runs with **no permissions** (`permissions: {}`):

- ✅ Cannot access GitHub secrets
- ✅ Cannot write to repository
- ✅ Prevents credential theft from untrusted code

This is critical because the test job runs potentially untrusted code on PRs.

### Cleanup Guarantees

The cleanup job always runs on a hosted runner (`if: always()`):

- ✅ EC2 instance terminated even on failure
- ✅ EC2 instance terminated even on manual cancellation
- ✅ EC2 instance terminated if the self-hosted test job never starts
- ✅ Prevents orphaned instances and cost overruns

The hosted cleanup job polls the GPU test job before termination. If the GPU job
does not leave the queue within the configured wait period, cleanup proceeds so
the EC2 runner is not left running indefinitely.

## Instance Types

| Instance | GPU | Memory | vCPUs | Cost/hr | Best For |
|----------|-----|--------|-------|---------|----------|
| **g6.2xlarge** | 1x L4 (24GB) | 24 GB | 8 | $0.86 | **gpt-oss:20b (recommended)** |
| g5.2xlarge | 1x A10G (24GB) | 24 GB | 8 | $1.21 | Alternative for gpt-oss:20b |
| g6.8xlarge | 1x L4 (24GB) | 24 GB | 32 | $1.38 | More vCPUs if needed |
| g6e.12xlarge | 4x L40S (192GB) | 192 GB | 48 | $5.44 | 70B+ models (future) |

**Note**: `gpt-oss:20b` ships with MXFP4-quantized MoE weights and is served without an extra vLLM `--quantization` flag in this workflow.

## Cost Estimates

| Scenario | Frequency | Instance | Cost/Run | Monthly Cost |
|----------|-----------|----------|----------|--------------|
| Weekly re-recording | 1x/week | g6.2xlarge | $0.43 | **$1.72** |
| Daily testing | 1x/day | g6.2xlarge | $0.43 | **$12.90** |
| On-demand (PRs) | 10x/month | g6.2xlarge | $0.43 | **$4.30** |
| With spot instances | 1x/week | g6.2xlarge (spot) | $0.09-$0.17 | **$0.36-$0.68** |

**Recommendation**: Use on-demand workflow_dispatch only. Add scheduled runs later if needed.

## Troubleshooting

### Workflow fails to launch EC2 instance

**Problem**: "InsufficientInstanceCapacity" error

**Solution**: The workflow automatically tries fallback subnet/AZ placements in `us-east-2`. If all fail:

1. Check AWS Service Health Dashboard for capacity issues
2. Try a different instance type (g5.2xlarge instead of g6.2xlarge)
3. Try again during off-peak hours

### vLLM server fails to start

**Problem**: Server doesn't respond to health checks

**Solutions**:

1. Check vLLM logs in workflow output
2. Verify GPU is detected: look for `nvidia-smi` output
3. Check CUDA installation: `nvcc --version`
4. Try a lower `max-model-len` or `gpu-memory-utilization` if startup fails due to memory pressure

### Tests fail but recordings not uploaded

**Problem**: No artifacts in workflow run

**Solutions**:

1. Check if tests actually created recordings
2. Verify `tests/integration/*/recordings/` directories exist
3. Check workflow logs for artifact upload errors

### EC2 instance not terminated

**Problem**: Instance still running after workflow completes

**Solutions**:

1. Check stop-gpu-runner job logs for errors
2. Manually terminate instance via AWS console
3. Set up CloudWatch alarm for long-running instances (see Phase 2)

### Cost overruns

**Problem**: Unexpected AWS charges

**Solutions**:

1. Check for orphaned instances in AWS EC2 console (filter by tag: `Purpose: vllm-gpu-recording`)
2. Set up AWS Budget alerts (see `IMPLEMENTATION_PLAN.md` Phase 2)
3. Review CloudWatch metrics for runner usage

## Performance Tuning

### Reduce Model Load Time

**Current**: ~5 minutes to download gpt-oss:20b

**Options**:

1. **Pre-cache in AMI**: Include model in GPU AMI (~0 min load time)
2. **EBS snapshot**: Attach pre-loaded model volume (~1 min)
3. **S3 cache**: Download from S3 instead of HuggingFace (~2 min)

See `IMPLEMENTATION_PLAN.md` Task #5 for implementation.

### Reduce Costs with Spot Instances

**Current**: $0.43 per run (on-demand)
**With spot**: $0.09-$0.17 per run (60-90% savings)

Spot instances can be interrupted, but for test workloads this is acceptable.

See `IMPLEMENTATION_PLAN.md` Task #3 for implementation.

## Adding New Models

To add a new model for GPU testing, first add a model input to
`.github/workflows/record-vllm-gpu-tests.yml`, then wire that value through
`setup-vllm-gpu`.

After that, add the new setup to the test matrix:

   ```json
   "gpu-vllm": [
     {"suite": "base", "setup": "vllm-gpu-gpt-oss"},
     {"suite": "base", "setup": "vllm-gpu-your-model"}
   ]
   ```

Create the setup in `tests/integration/suites.py`:

   ```python
   "vllm-gpu-your-model": Setup(
       name="vllm-gpu",
       defaults={"text_model": "vllm/your-model"},
   )
   ```

Choose an instance type:

- < 20B params: `g6.2xlarge` (24GB)
- 20-70B params: `g6.8xlarge` or `g6e.12xlarge` (192GB)
- 70B+ params: `g6e.12xlarge` (192GB) or `g6e.48xlarge` (384GB)

## Monitoring

### CloudWatch Dashboards

Create a dashboard to track:

- Total GPU runner costs (daily/weekly/monthly)
- Instance launch success rate
- Average test duration
- Failures by reason

See `IMPLEMENTATION_PLAN.md` Task #4 for setup.

### Cost Allocation Tags

All EC2 instances are tagged with:

- `Project`: llama-stack
- `Purpose`: vllm-gpu-recording
- `Model`: gpt-oss:20b
- `GitHubRepository`: your-org/llama-stack
- `GitHubRunId`: 12345

Enable cost allocation in **AWS Billing > Cost Allocation Tags** to track costs by tag.

## References

- **Implementation Plan**: `IMPLEMENTATION_PLAN.md`
- **AWS EC2 Instance Types**: <https://aws.amazon.com/ec2/instance-types/g6/>
- **vLLM Documentation**: <https://docs.vllm.ai/>
- **GitHub OIDC**: <https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services>

## Support

For issues or questions:

- Create an issue in the repository
- Check existing issues for similar problems
- Review troubleshooting section above
- Contact: Charles Doern (@cdoern)
