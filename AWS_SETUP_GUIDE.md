# AWS Setup Guide for GPU Runners

This guide walks through setting up the AWS infrastructure required for GPU-enabled self-hosted runners.

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI installed and configured
- Access to create IAM roles and OIDC providers

## Step 1: Set up OIDC Provider for GitHub Actions

GitHub Actions can authenticate to AWS using OpenID Connect (OIDC) instead of long-lived access keys.

### 1.1 Create OIDC Provider in IAM

```bash
# Using AWS CLI
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

**Via AWS Console**:

1. Go to IAM > Identity providers
2. Click **Add provider**
3. Provider type: **OpenID Connect**
4. Provider URL: `https://token.actions.githubusercontent.com`
5. Audience: `sts.amazonaws.com`
6. Click **Add provider**

### 1.2 Create IAM Role for GitHub Actions

Create a file `trust-policy.json`:

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

Replace:

- `YOUR_ACCOUNT_ID`: Your AWS account ID (e.g., `123456789012`)
- `YOUR_ORG/llama-stack`: Your GitHub repository (e.g., `meta-llama/llama-stack`)

Create the role:

```bash
aws iam create-role \
  --role-name GitHubActionsLlamaStackGPU \
  --assume-role-policy-document file://trust-policy.json
```

### 1.3 Attach Permissions Policy

Create a file `permissions-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2Management",
      "Effect": "Allow",
      "Action": [
        "ec2:RunInstances",
        "ec2:TerminateInstances",
        "ec2:DescribeInstances",
        "ec2:DescribeInstanceStatus",
        "ec2:DescribeInstanceTypes",
        "ec2:CreateTags",
        "ec2:DescribeImages",
        "ec2:DescribeSubnets",
        "ec2:DescribeSecurityGroups",
        "ec2:DescribeKeyPairs",
        "ec2:DescribeVolumes"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": ["us-east-1", "us-east-2"]
        }
      }
    },
    {
      "Sid": "IAMPassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::YOUR_ACCOUNT_ID:role/GitHubActionsLlamaStackGPU",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": "ec2.amazonaws.com"
        }
      }
    }
  ]
}
```

Replace `YOUR_ACCOUNT_ID` with your AWS account ID.

Attach the policy:

```bash
aws iam put-role-policy \
  --role-name GitHubActionsLlamaStackGPU \
  --policy-name EC2GPURunnerPermissions \
  --policy-document file://permissions-policy.json
```

### 1.4 Save Role ARN

Get the role ARN:

```bash
aws iam get-role --role-name GitHubActionsLlamaStackGPU --query 'Role.Arn' --output text
```

Save this ARN - you'll need it for GitHub secrets:

```text
arn:aws:iam::123456789012:role/GitHubActionsLlamaStackGPU
```

## Step 2: Set up VPC and Networking

### 2.1 Option A: Use Existing VPC

If you already have a VPC with internet access:

```bash
# List VPCs
aws ec2 describe-vpcs --region us-east-2

# List subnets in VPC
aws ec2 describe-subnets --region us-east-2 --filters "Name=vpc-id,Values=vpc-xxxxx"
```

### 2.2 Option B: Create New VPC

```bash
# Create VPC in us-east-2
aws ec2 create-vpc \
  --region us-east-2 \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=llama-stack-gpu}]'

# Enable DNS hostnames
aws ec2 modify-vpc-attribute \
  --region us-east-2 \
  --vpc-id vpc-xxxxx \
  --enable-dns-hostnames

# Create Internet Gateway
aws ec2 create-internet-gateway \
  --region us-east-2 \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=llama-stack-gpu-igw}]'

# Attach to VPC
aws ec2 attach-internet-gateway \
  --region us-east-2 \
  --vpc-id vpc-xxxxx \
  --internet-gateway-id igw-xxxxx
```

### 2.3 Create Subnets

Create 3 subnets in us-east-2 (one per AZ):

```bash
# us-east-2a
aws ec2 create-subnet \
  --region us-east-2 \
  --vpc-id vpc-xxxxx \
  --cidr-block 10.0.1.0/24 \
  --availability-zone us-east-2a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=llama-stack-gpu-2a}]'

# us-east-2b
aws ec2 create-subnet \
  --region us-east-2 \
  --vpc-id vpc-xxxxx \
  --cidr-block 10.0.2.0/24 \
  --availability-zone us-east-2b \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=llama-stack-gpu-2b}]'

# us-east-2c
aws ec2 create-subnet \
  --region us-east-2 \
  --vpc-id vpc-xxxxx \
  --cidr-block 10.0.3.0/24 \
  --availability-zone us-east-2c \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=llama-stack-gpu-2c}]'
```

Repeat for us-east-1 with appropriate CIDR blocks.

### 2.4 Configure Route Table

```bash
# Create route table
aws ec2 create-route-table \
  --region us-east-2 \
  --vpc-id vpc-xxxxx \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=llama-stack-gpu-rt}]'

# Add route to internet gateway
aws ec2 create-route \
  --region us-east-2 \
  --route-table-id rtb-xxxxx \
  --destination-cidr-block 0.0.0.0/0 \
  --gateway-id igw-xxxxx

# Associate subnets with route table
aws ec2 associate-route-table \
  --region us-east-2 \
  --route-table-id rtb-xxxxx \
  --subnet-id subnet-xxxxx
```

### 2.5 Create Security Group

```bash
aws ec2 create-security-group \
  --region us-east-2 \
  --group-name llama-stack-gpu-runners \
  --description "Security group for llama-stack GPU runners" \
  --vpc-id vpc-xxxxx \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=Name,Value=llama-stack-gpu-sg}]'

# Add outbound rules (allow all - default)
# No inbound rules needed (runners connect outbound only)
```

## Step 3: Create GPU-Enabled AMI

### 3.1 Launch Base Instance

```bash
# Launch Ubuntu 22.04 instance with GPU
aws ec2 run-instances \
  --region us-east-2 \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g6.2xlarge \
  --key-name your-key-pair \
  --subnet-id subnet-xxxxx \
  --security-group-ids sg-xxxxx \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=llama-stack-gpu-ami-builder}]'
```

### 3.2 SSH and Configure Instance

```bash
ssh -i your-key.pem ubuntu@<instance-public-ip>
```

Run the setup script:

```bash
#!/bin/bash
set -e

# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system packages
sudo apt-get install -y \
  build-essential \
  gcc \
  g++ \
  make \
  git \
  curl \
  wget \
  ca-certificates \
  gnupg \
  lsb-release

# Install NVIDIA drivers
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Install CUDA 12.4
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-4

# Configure CUDA environment
echo 'export CUDA_HOME=/usr/local/cuda-12.4' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export PATH=$PATH:$CUDA_HOME/bin' | sudo tee -a /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' | sudo tee -a /etc/profile.d/cuda.sh

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Install Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Verify installation
nvidia-smi
nvcc --version
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

echo "Setup complete! Ready to create AMI."
```

### 3.3 Create AMI

After setup completes:

```bash
# From your local machine
aws ec2 create-image \
  --region us-east-2 \
  --instance-id i-xxxxx \
  --name "llama-stack-gpu-ubuntu-2204-cuda-12.4-$(date +%Y%m%d)" \
  --description "Ubuntu 22.04 with NVIDIA drivers, CUDA 12.4, Docker, Python 3.12" \
  --tag-specifications 'ResourceType=image,Tags=[{Key=Name,Value=llama-stack-gpu-ami}]'

# Copy AMI to us-east-1
aws ec2 copy-image \
  --region us-east-1 \
  --source-region us-east-2 \
  --source-image-id ami-xxxxx \
  --name "llama-stack-gpu-ubuntu-2204-cuda-12.4-$(date +%Y%m%d)"
```

### 3.4 Test AMI

Launch a test instance:

```bash
aws ec2 run-instances \
  --region us-east-2 \
  --image-id ami-xxxxx \
  --instance-type g6.2xlarge \
  --subnet-id subnet-xxxxx \
  --security-group-ids sg-xxxxx

# SSH and verify
ssh ubuntu@<ip>
nvidia-smi
nvcc --version
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Step 4: Create GitHub Personal Access Token

1. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. Click **Generate new token (classic)**
3. Name: `llama-stack-gpu-runners`
4. Scopes: Select `repo` (full control of private repositories)
5. Click **Generate token**
6. **Save the token** - you won't see it again!

## Step 5: Configure GitHub Secrets and Variables

### 5.1 Add Secrets

Go to your GitHub repository > Settings > Secrets and variables > Actions > Secrets

Click **New repository secret** for each:

- **Name**: `AWS_ROLE_ARN`
  - **Value**: `arn:aws:iam::123456789012:role/GitHubActionsLlamaStackGPU`

- **Name**: `RELEASE_PAT`
  - **Value**: `ghp_xxxxxxxxxxxxx` (from Step 4)

### 5.2 Add Variables

Go to your GitHub repository > Settings > Secrets and variables > Actions > Variables

Click **New repository variable** for each:

**us-east-2**:

- `SUBNET_US_EAST_2A`: `subnet-xxxxx`
- `SUBNET_US_EAST_2B`: `subnet-xxxxx`
- `SUBNET_US_EAST_2C`: `subnet-xxxxx`
- `AWS_EC2_AMI_US_EAST_2`: `ami-xxxxx`
- `SECURITY_GROUP_ID_US_EAST_2`: `sg-xxxxx`

**us-east-1**:

- `SUBNET_US_EAST_1A`: `subnet-xxxxx`
- `SUBNET_US_EAST_1B`: `subnet-xxxxx`
- `SUBNET_US_EAST_1C`: `subnet-xxxxx`
- `AWS_EC2_AMI_US_EAST_1`: `ami-xxxxx`
- `SECURITY_GROUP_ID_US_EAST_1`: `sg-xxxxx`

## Step 6: Test the Setup

### 6.1 Trigger Test Workflow

1. Go to Actions tab in GitHub
2. Select **vLLM GPU Recording** workflow
3. Click **Run workflow**
4. Use default values
5. Click **Run workflow**

### 6.2 Verify Success

Check that:

- [ ] EC2 instance launches in us-east-2
- [ ] Runner registers and picks up job
- [ ] GPU is detected (`nvidia-smi` output)
- [ ] vLLM server starts successfully
- [ ] Tests run and complete
- [ ] Recordings uploaded as artifacts
- [ ] EC2 instance terminates

### 6.3 Check AWS Console

1. Go to EC2 > Instances
2. Verify no instances with tag `Purpose: vllm-gpu-recording` are still running
3. Check terminated instances - should see your test instance

## Troubleshooting

### OIDC Authentication Fails

**Error**: "Not authorized to perform sts:AssumeRoleWithWebIdentity"

**Solutions**:

1. Verify OIDC provider is created correctly
2. Check trust policy allows your repository
3. Verify `token.actions.githubusercontent.com:sub` matches your repo

### EC2 Launch Fails

**Error**: "InsufficientInstanceCapacity"

**Solutions**:

1. Try different AZ (workflow does this automatically)
2. Try different instance type
3. Check service quotas in AWS console

### AMI Not Found

**Error**: "Invalid AMI ID"

**Solutions**:

1. Verify AMI exists in the region you're trying to use
2. Check AMI ID is correct in GitHub variables
3. Ensure AMI is not deregistered

### Security Group Issues

**Error**: "UnauthorizedOperation"

**Solutions**:

1. Verify security group exists in same VPC as subnet
2. Check security group allows outbound HTTPS (443)
3. Ensure IAM role has `ec2:DescribeSecurityGroups` permission

## Cost Monitoring

### Enable Cost Allocation Tags

1. Go to AWS Billing > Cost Allocation Tags
2. Activate these tags:
   - `Project`
   - `Purpose`
   - `GitHubRepository`
   - `GitHubRunId`
3. Wait 24 hours for tags to appear in Cost Explorer

### Create Budget Alert

```bash
aws budgets create-budget \
  --account-id 123456789012 \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

`budget.json`:

```json
{
  "BudgetName": "llama-stack-gpu-runners",
  "BudgetLimit": {
    "Amount": "50",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST",
  "CostFilters": {
    "TagKeyValue": ["user:Purpose$vllm-gpu-recording"]
  }
}
```

`notifications.json`:

```json
[
  {
    "Notification": {
      "NotificationType": "ACTUAL",
      "ComparisonOperator": "GREATER_THAN",
      "Threshold": 80
    },
    "Subscribers": [
      {
        "SubscriptionType": "EMAIL",
        "Address": "your-email@example.com"
      }
    ]
  }
]
```

## Security Best Practices

### 1. Principle of Least Privilege

The IAM role only has permissions to:

- Launch/terminate EC2 instances
- Only in us-east-1 and us-east-2 regions
- Only for llama-stack repository

### 2. No Long-Lived Credentials

Using OIDC means:

- No AWS access keys stored in GitHub
- Tokens expire after use
- Better audit trail in CloudTrail

### 3. Regular Audits

Monthly:

- [ ] Review EC2 instances for orphaned runners
- [ ] Check AWS costs vs budget
- [ ] Review CloudTrail logs for unusual activity
- [ ] Rotate GitHub PAT if needed

## Next Steps

After completing this setup:

1. ✅ Test workflow runs successfully
2. ✅ No orphaned EC2 instances
3. ✅ Costs are as expected (~$0.43 per run)
4. Read `docs/gpu-runners.md` for usage guide
5. Consider implementing Phase 2 optimizations (spot instances, model caching)

## Support

For issues during setup:

- Check AWS CloudTrail for API errors
- Review GitHub Actions logs for OIDC errors
- Verify all ARNs and IDs are correct
- Contact: Charles Doern (@cdoern)
