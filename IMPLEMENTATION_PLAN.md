# GPU Runners Implementation Plan

This document outlines the step-by-step implementation plan for adding GPU-enabled self-hosted runners for vLLM re-recording with gpt-oss:20b.

## Overview

**Goal**: Enable re-recording of vLLM integration tests with gpt-oss:20b on GPU-enabled EC2 instances via GitHub Actions.

**Key Benefits**:

- Test larger models (20B parameters) that don't fit on CPU runners
- Faster inference times with GPU acceleration
- More realistic production-like test environment
- On-demand re-recording via workflow_dispatch

**Estimated Cost**: ~$0.43 per run (30 min on g6.2xlarge), ~$1.72/month for weekly runs

---

## Phase 1: Core Infrastructure (Week 1-2)

**Goal**: Set up basic GPU runner capability and test end-to-end.

### Tasks

#### 1. Set up AWS infrastructure (Task #11) 🔧

**Priority**: Critical - blocking all other work
**Owner**: DevOps/Charles

**Actions**:

- [ ] Create or identify VPC in us-east-2 and us-east-1
- [ ] Create subnets (3 AZs per region = 6 total)
- [ ] Configure security groups (SSH, HTTPS, HTTP)
- [ ] Set up IAM role for OIDC authentication
- [ ] Document all IDs and add to GitHub repo variables

**Repository Variables** (add via Settings > Secrets and variables > Actions):

```text
SUBNET_US_EAST_2A=subnet-xxxxx
SUBNET_US_EAST_2B=subnet-xxxxx
SUBNET_US_EAST_2C=subnet-xxxxx
SUBNET_US_EAST_1A=subnet-xxxxx
SUBNET_US_EAST_1B=subnet-xxxxx
SUBNET_US_EAST_1C=subnet-xxxxx
AWS_EC2_AMI_US_EAST_2=ami-xxxxx
AWS_EC2_AMI_US_EAST_1=ami-xxxxx
SECURITY_GROUP_ID_US_EAST_2=sg-xxxxx
SECURITY_GROUP_ID_US_EAST_1=sg-xxxxx
```

**Repository Secrets**:

```text
AWS_ROLE_ARN=arn:aws:iam::123456789012:role/GitHubActionsRole
RELEASE_PAT=ghp_xxxxx (GitHub PAT with 'repo' scope)
```

**Dependencies**: None
**Estimated Time**: 2-4 hours

---

#### 2. Create GPU-enabled AMI (Task #10) 🖼️

**Priority**: Critical - needed for runner launch
**Depends On**: Task #11 (AWS infrastructure)

**Actions**:

- [ ] Launch base EC2 instance (g6.2xlarge with Amazon Linux 2023 or Ubuntu 22.04)
- [ ] Install NVIDIA drivers and CUDA 12.4
- [ ] Install Docker with NVIDIA Container Toolkit
- [ ] Install system packages (gcc, g++, make, git, python3.12, python3.12-devel)
- [ ] Configure CUDA environment variables
- [ ] Verify with `nvidia-smi`
- [ ] Create AMI in both us-east-2 and us-east-1
- [ ] Document AMI IDs

**Alternative**: Use AWS Deep Learning AMI and customize

**Validation**:

```bash
nvidia-smi  # Should show GPU
nvcc --version  # Should show CUDA 12.4
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**Dependencies**: Task #11
**Estimated Time**: 3-5 hours (includes testing)

---

#### 3. Create launch-gpu-runner action (Task #12) ⚙️

**Priority**: Critical - core functionality
**Depends On**: Tasks #10, #11

**File**: `.github/actions/launch-gpu-runner/action.yml`

**Key Features**:

- Multi-region fallback (us-east-2 → us-east-1)
- Multi-AZ fallback (3 AZs per region)
- Dynamic runner label generation
- Resource tagging for cost tracking
- Error handling and retries

**Implementation Options**:

1. Use `machulav/ec2-github-runner@v2.3.6` with wrapper logic
2. Fork and customize `instructlab/ci-actions` (if available)
3. Build custom JavaScript action

**Recommended**: Option 1 (machulav with wrapper)

**Dependencies**: Tasks #10, #11
**Estimated Time**: 4-6 hours

---

#### 4. Create setup-vllm-gpu action (Task #6) 🚀

**Priority**: Critical - needed for test execution
**Depends On**: Task #10 (AMI with CUDA)

**File**: `.github/actions/setup-vllm-gpu/action.yml`

**Key Features**:

- Install vLLM with GPU support
- Pull gpt-oss:20b model (or specified model)
- Start vLLM server with optimal settings:
  - AWQ quantization for 24GB GPUs
  - GPU memory utilization: 0.85
  - Tool calling support (hermes parser)
- Health check with timeout
- Support both Ollama and HuggingFace models

**vLLM Server Command**:

```bash
vllm serve gpt-oss:20b \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --dtype auto \
  --quantization awq
```

**Dependencies**: Task #10
**Estimated Time**: 3-4 hours

---

#### 5. Create record-vllm-gpu-tests workflow (Task #1) 📋

**Priority**: Critical - ties everything together
**Depends On**: Tasks #6, #12

**File**: `.github/workflows/record-vllm-gpu-tests.yml`

**Structure**:

```yaml
name: vLLM GPU Recording

on:
  workflow_dispatch:
    inputs:
      model: [gpt-oss:20b, gpt-oss:latest, Qwen/Qwen3-0.6B]
      instance_type: [g6.2xlarge, g5.2xlarge, g6.8xlarge, g6e.12xlarge]
      suite: [base, responses, vllm-reasoning]
      pr_number: (optional)

jobs:
  start-gpu-runner:
    # Launch EC2 with launch-gpu-runner action

  record-vllm-tests:
    runs-on: ${{ needs.start-gpu-runner.outputs.label }}
    permissions: {}  # CRITICAL: No permissions
    # Setup vLLM GPU, run tests, upload artifacts

  stop-gpu-runner:
    if: always()  # CRITICAL: Always cleanup
    # Terminate EC2 instance
```

**Security Highlights**:

- Test job has `permissions: {}` (prevents secret theft)
- OIDC authentication (no long-lived AWS credentials)
- `if: always()` on cleanup job (prevents orphaned instances)

**Dependencies**: Tasks #6, #12
**Estimated Time**: 4-6 hours

---

#### 6. Update test configuration files (Tasks #7, #9) 📝

**Priority**: High - needed for test execution

#### 6a. Update tests/integration/suites.py (Task #7)

Add new setup to SETUP_DEFINITIONS dict:

```python
SETUP_DEFINITIONS = {
    # ... existing setups ...
    "vllm-gpu-gpt-oss": Setup(
        name="vllm-gpu",
        description="vLLM GPU provider with gpt-oss:20b model",
        env={
            "VLLM_URL": "http://0.0.0.0:8000/v1",
        },
        defaults={
            "text_model": "vllm/gpt-oss:20b",
        },
    ),
}
```

#### 6b. Update tests/integration/ci_matrix.json (Task #9)

Add GPU matrix:

```json
"gpu-vllm": [
  {"suite": "base", "setup": "vllm-gpu-gpt-oss", "model": "gpt-oss:20b"},
  {"suite": "responses", "setup": "vllm-gpu-gpt-oss", "model": "gpt-oss:20b"},
  {"suite": "vllm-reasoning", "setup": "vllm-gpu-gpt-oss", "model": "gpt-oss:20b"}
]
```

**Dependencies**: None (can be done in parallel)
**Estimated Time**: 1-2 hours total

---

#### 7. End-to-end testing (Task #8) ✅

**Priority**: Critical - validates entire system
**Depends On**: Tasks #1, #6, #7, #9, #10, #11, #12

**Test Plan**:

1. **Happy Path Test**:
   - [ ] Trigger workflow via workflow_dispatch
   - [ ] Verify EC2 launches in us-east-2a
   - [ ] Verify runner registration
   - [ ] Verify vLLM starts with gpt-oss:20b
   - [ ] Verify tests execute successfully
   - [ ] Verify recordings uploaded
   - [ ] Verify EC2 cleanup
   - [ ] Check execution time (< 30 min)
   - [ ] Check AWS cost (~$0.43)

2. **Failure Scenarios**:
   - [ ] Capacity issue → verify fallback to us-east-2b
   - [ ] Region capacity issue → verify fallback to us-east-1
   - [ ] Test failure → verify cleanup still happens
   - [ ] Manual cancellation → verify cleanup

3. **Performance Validation**:
   - [ ] Runner startup: < 5 min
   - [ ] vLLM startup: < 5 min
   - [ ] Test execution: ~20 min
   - [ ] Total: < 30 min

**Success Criteria**:

- 3 consecutive successful runs
- 95%+ success rate over 10 runs
- Average cost < $0.50 per run
- Zero orphaned instances

**Dependencies**: All Phase 1 tasks
**Estimated Time**: 4-8 hours (includes iteration)

---

#### 8. Documentation (Task #2) 📚

**Priority**: Medium - needed for team adoption
**Depends On**: Task #8 (successful testing)

**Actions**:

- [ ] Create `docs/gpu-runners.md` with usage guide
- [ ] Update README.md with GPU runner section
- [ ] Document troubleshooting steps
- [ ] Add inline comments to workflow files
- [ ] Document cost estimates and monitoring

**Content**:

- How to trigger manual re-recording
- Expected costs and runtime
- AWS prerequisites
- Troubleshooting guide
- Architecture diagram (reference GPU_RUNNERS_DESIGN.md)

**Dependencies**: Task #8
**Estimated Time**: 2-3 hours

---

## Phase 1 Summary

**Total Estimated Time**: 2-3 weeks (part-time)
**Key Deliverables**:

- ✅ Working GPU runner infrastructure
- ✅ Manual workflow_dispatch for re-recording
- ✅ End-to-end tested with gpt-oss:20b
- ✅ Documentation for team

**Phase 1 Completion Criteria**:

- [ ] All Phase 1 tasks completed
- [ ] 10+ successful GPU recording runs
- [ ] Zero orphaned EC2 instances
- [ ] Documentation reviewed and approved

---

## Phase 2: Optimization (Week 3-4)

**Goal**: Reduce costs and improve performance.

### Phase 2 Tasks

#### 9. Set up cost monitoring (Task #4) 💰

**Priority**: High - prevents cost overruns

**Actions**:

- [ ] Create AWS Budget ($50/month alert)
- [ ] CloudWatch alarm for long-running instances (> 2 hours)
- [ ] Lambda for auto-cleanup of orphaned instances
- [ ] Enable cost allocation tags
- [ ] Create CloudWatch dashboard

**Metrics to Track**:

- Total monthly GPU costs
- Average run duration
- Success rate
- Spot vs on-demand usage

**Dependencies**: Task #8 (completed Phase 1)
**Estimated Time**: 3-4 hours

---

#### 10. Implement spot instances (Task #3) 💵

**Priority**: High - 70-80% cost savings

**Actions**:

- [ ] Update launch-gpu-runner action for spot support
- [ ] Add spot/on-demand fallback logic
- [ ] Update workflow to use spot by default
- [ ] Test spot reliability (10+ runs)
- [ ] Update cost documentation

**Expected Savings**: $0.43 → $0.09-$0.17 per run

**Dependencies**: Task #8 (completed Phase 1)
**Estimated Time**: 4-6 hours

---

#### 11. Add model caching (Task #5) ⚡

**Priority**: Medium - performance optimization

**Actions**:

- [ ] Pre-cache gpt-oss:20b in AMI
- [ ] Test with cached model
- [ ] Measure time savings
- [ ] Update documentation

**Expected Improvement**: 30 min → 20 min total runtime

**Dependencies**: Task #10 (AMI creation)
**Estimated Time**: 2-3 hours

---

## Phase 2 Summary

**Total Estimated Time**: 1-2 weeks (part-time)
**Key Deliverables**:

- ✅ Cost monitoring and alerts
- ✅ Spot instance support (70-80% cost reduction)
- ✅ Model caching (33% faster runs)

**Expected Outcomes**:

- Monthly cost: $1.72 → $0.34-$0.69 (with spot instances)
- Run time: 30 min → 20 min (with caching)

---

## Phase 3: Automation (Future)

**Goal**: Integrate GPU runners into existing CI/CD.

### Potential Tasks

1. **Add scheduled GPU runs**:
   - Weekly full test suite on gpt-oss:20b
   - Update ci_matrix.json schedules

2. **Integrate with record-integration-tests.yml**:
   - Add vllm-gpu to provider matrix
   - Support manual trigger for GPU re-recording

3. **PR comment integration**:
   - Notify when GPU recordings complete
   - Link to artifacts

4. **Auto-scaling**:
   - Queue-based runner provisioning
   - Scale based on pending workflow runs

---

## Phase 4: Advanced Features (Future)

**Goal**: Support multiple models and advanced use cases.

### Phase 4 Tasks

1. **Multi-model support**:
   - Add more models to GPU matrix
   - Model-specific optimizations

2. **Distributed inference**:
   - Multi-GPU support (g6e.12xlarge with 4x L40S)
   - Tensor parallelism for 70B+ models

3. **Custom metrics dashboard**:
   - Real-time cost tracking
   - Performance trends
   - Success rate by model

---

## Risk Mitigation

### Risk 1: EC2 Capacity Issues

**Mitigation**: Multi-region, multi-AZ fallback (9 AZs total)
**Probability**: Low (<5% with fallback)

### Risk 2: Cost Overruns

**Mitigation**: AWS Budgets, CloudWatch alarms, auto-cleanup Lambda
**Probability**: Very Low (<1% with monitoring)

### Risk 3: Orphaned Instances

**Mitigation**: `if: always()` cleanup, CloudWatch alarms, auto-cleanup
**Probability**: Very Low (<1%)

### Risk 4: Security Issues

**Mitigation**: OIDC auth, `permissions: {}`, read-only secrets
**Probability**: Very Low (<1%)

---

## Success Metrics

### Functional

- [ ] Successfully record gpt-oss:20b tests within 30 minutes
- [ ] 95%+ success rate for runner provisioning
- [ ] Zero leaked AWS credentials

### Performance

- [ ] Runner startup time < 5 minutes
- [ ] vLLM startup time < 5 minutes
- [ ] Total execution time < 30 minutes (20 min with caching)

### Cost

- [ ] Monthly AWS costs < $20 for on-demand
- [ ] Monthly AWS costs < $5 with spot instances
- [ ] Spot instance utilization > 70%

### Reliability

- [ ] Multi-region fallback prevents <2% of failures
- [ ] 100% runner cleanup rate
- [ ] Zero orphaned instances over 30 days

---

## Next Steps

### Immediate (This Week)

1. **Task #11**: Set up AWS infrastructure (Charles + DevOps)
2. **Task #10**: Create GPU AMI with CUDA (Charles)
3. **Task #12**: Create launch-gpu-runner action (Charles)

### Week 2

1. **Task #6**: Create setup-vllm-gpu action (Charles)
2. **Task #1**: Create record-vllm-gpu-tests workflow (Charles)
3. **Tasks #7, #9**: Update test configuration (Charles)

### Week 3

1. **Task #8**: End-to-end testing (Charles + team)
2. **Task #2**: Documentation (Charles)

### Week 4

1. **Task #4**: Set up cost monitoring (DevOps)
2. **Task #3**: Implement spot instances (Charles)
3. **Task #5**: Add model caching (Charles)

---

## Resources

- **Design Document**: `GPU_RUNNERS_DESIGN.md`
- **AWS Documentation**: [EC2 Instance Types](https://aws.amazon.com/ec2/instance-types/g6/)
- **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai/)
- **GitHub Actions Security**: [Security hardening](https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions)
- **Reference Implementations**:
  - [instructlab/instructlab](https://github.com/instructlab/instructlab/tree/main/.github/workflows)
  - [opendatahub-io/data-processing](https://github.com/opendatahub-io/data-processing/blob/main/.github/workflows/execute-all-notebooks.yml)

---

## Questions or Issues?

Contact: Charles Doern (@cdoern)
