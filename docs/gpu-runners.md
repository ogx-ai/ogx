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
   - **Model**: `gpt-oss:20b` (default)
   - **Instance Type**: `g6.2xlarge` (default)
   - **Suite**: `base` (default)
5. Click **Run workflow**

The workflow will:

1. Launch a GPU EC2 instance (5 min)
2. Setup vLLM with the model (5 min)
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
│  - Select model and instance type               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Job 1: Start GPU EC2 Runner                    │
│  - AWS OIDC authentication (no long-lived keys!)│
│  - Multi-region/AZ fallback                     │
│  - Launch g6.2xlarge with GPU AMI               │
│  - Register as GitHub Actions runner            │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Job 2: Run vLLM Recording Tests                │
│  - Runs on GPU runner (permissions: {})         │
│  - Install vLLM with CUDA support               │
│  - Start vLLM server with AWQ quantization      │
│  - Run integration tests in record mode         │
│  - Upload recordings as artifacts               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│  Job 3: Stop GPU EC2 Runner                     │
│  - Terminate instance (always runs!)            │
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
          "aws:RequestedRegion": ["us-east-1", "us-east-2"]
        }
      }
    }
  ]
}
```

#### 2. VPC and Subnets

You need subnets in two regions for fallback:

**us-east-2 (Primary)**:

- us-east-2a: subnet-xxxxx
- us-east-2b: subnet-xxxxx
- us-east-2c: subnet-xxxxx

**us-east-1 (Fallback)**:

- us-east-1a: subnet-xxxxx
- us-east-1b: subnet-xxxxx
- us-east-1c: subnet-xxxxx

#### 3. Security Groups

Create security groups in both regions with:

**Inbound Rules**:

- None (runners connect outbound only)

**Outbound Rules**:

- Port 443 (HTTPS): `0.0.0.0/0` - GitHub API, HuggingFace, PyPI
- Port 80 (HTTP): `0.0.0.0/0` - Package downloads

#### 4. GPU-Enabled AMI

Create AMIs in both regions with:

- Base OS: Amazon Linux 2023 or Ubuntu 22.04
- NVIDIA drivers
- CUDA 12.4 runtime
- Docker with NVIDIA Container Toolkit
- Python 3.12

See `GPU_RUNNERS_DESIGN.md` Appendix C for AMI build script.

### GitHub Configuration

#### Secrets

Add these to **Settings > Secrets and variables > Actions > Secrets**:

- `AWS_ROLE_ARN`: ARN of the IAM role for OIDC (e.g., `arn:aws:iam::123456789012:role/GitHubActionsRole`)
- `RELEASE_PAT`: GitHub Personal Access Token with `repo` scope

#### Variables

Add these to **Settings > Secrets and variables > Actions > Variables**:

**us-east-2**:

- `SUBNET_US_EAST_2A`: subnet-xxxxx
- `SUBNET_US_EAST_2B`: subnet-xxxxx
- `SUBNET_US_EAST_2C`: subnet-xxxxx
- `AWS_EC2_AMI_US_EAST_2`: ami-xxxxx
- `SECURITY_GROUP_ID_US_EAST_2`: sg-xxxxx

**us-east-1**:

- `SUBNET_US_EAST_1A`: subnet-xxxxx
- `SUBNET_US_EAST_1B`: subnet-xxxxx
- `SUBNET_US_EAST_1C`: subnet-xxxxx
- `AWS_EC2_AMI_US_EAST_1`: ami-xxxxx
- `SECURITY_GROUP_ID_US_EAST_1`: sg-xxxxx

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

The cleanup job always runs (`if: always()`):

- ✅ EC2 instance terminated even on failure
- ✅ EC2 instance terminated even on manual cancellation
- ✅ Prevents orphaned instances and cost overruns

## Instance Types

| Instance | GPU | Memory | vCPUs | Cost/hr | Best For |
|----------|-----|--------|-------|---------|----------|
| **g6.2xlarge** | 1x L4 (24GB) | 24 GB | 8 | $0.86 | **gpt-oss:20b (recommended)** |
| g5.2xlarge | 1x A10G (24GB) | 24 GB | 8 | $1.21 | Alternative for gpt-oss:20b |
| g6.8xlarge | 1x L4 (24GB) | 24 GB | 32 | $1.38 | More vCPUs if needed |
| g6e.12xlarge | 4x L40S (192GB) | 192 GB | 48 | $5.44 | 70B+ models (future) |

**Note**: gpt-oss:20b requires ~40GB in FP16, but we use AWQ quantization to fit in 24GB GPU memory.

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

**Solution**: The workflow automatically tries fallback regions/AZs. If all fail:

1. Check AWS Service Health Dashboard for capacity issues
2. Try a different instance type (g5.2xlarge instead of g6.2xlarge)
3. Try again during off-peak hours

### vLLM server fails to start

**Problem**: Server doesn't respond to health checks

**Solutions**:

1. Check vLLM logs in workflow output
2. Verify GPU is detected: look for `nvidia-smi` output
3. Check CUDA installation: `nvcc --version`
4. Try different quantization: change `quantization: 'awq'` to `quantization: 'none'`

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

To add a new model for GPU testing:

1. **Update workflow input** (`.github/workflows/record-vllm-gpu-tests.yml`):

   ```yaml
   model:
     options:
       - gpt-oss:20b
       - gpt-oss:latest
       - your-new-model
   ```

2. **Add to test matrix** (`tests/integration/ci_matrix.json`):

   ```json
   "gpu-vllm": [
     {"suite": "base", "setup": "vllm-gpu-gpt-oss"},
     {"suite": "base", "setup": "vllm-gpu-your-model"}
   ]
   ```

3. **Create setup** (`tests/integration/suites.py`):

   ```python
   "vllm-gpu-your-model": Setup(
       name="vllm-gpu",
       defaults={"text_model": "vllm/your-model"},
   )
   ```

4. **Choose instance type**:
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

- **Design Document**: `GPU_RUNNERS_DESIGN.md`
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
