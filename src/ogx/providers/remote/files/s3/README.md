# S3 Files Provider

A remote S3-based implementation of the OGX Files API that provides scalable cloud file storage with metadata persistence.

## Features

- **AWS S3 Storage**: Store files in AWS S3 buckets for scalable, durable storage
- **Metadata Management**: Uses SQL database for efficient file metadata queries
- **OpenAI API Compatibility**: Full compatibility with OpenAI Files API endpoints
- **Flexible Authentication**: Support for IAM roles and access keys
- **Custom S3 Endpoints**: Support for MinIO and other S3-compatible services

## Configuration

### Basic Configuration

```yaml
api: files
provider_type: remote::s3
config:
  bucket_name: my-ogx-files
  region: us-east-1
  metadata_store:
    type: sqlite
    db_path: ./s3_files_metadata.db
```

### Advanced Configuration

```yaml
api: files
provider_type: remote::s3
config:
  bucket_name: my-ogx-files
  region: us-east-1
  aws_access_key_id: YOUR_ACCESS_KEY
  aws_secret_access_key: YOUR_SECRET_KEY
  endpoint_url: https://s3.amazonaws.com  # Optional for custom endpoints
  metadata_store:
    type: sqlite
    db_path: ./s3_files_metadata.db
```

### Environment Variables

The configuration supports environment variable substitution:

```yaml
config:
  bucket_name: "${env.S3_BUCKET_NAME}"
  region: "${env.AWS_REGION:=us-east-1}"
  aws_access_key_id: "${env.AWS_ACCESS_KEY_ID:=}"
  aws_secret_access_key: "${env.AWS_SECRET_ACCESS_KEY:=}"
  endpoint_url: "${env.S3_ENDPOINT_URL:=}"
```

Note: `S3_BUCKET_NAME` has no default value since S3 bucket names must be globally unique.

## Authentication

### IAM Roles (Recommended)

For production deployments, use IAM roles:

```yaml
config:
  bucket_name: my-bucket
  region: us-east-1
  # No credentials needed - will use IAM role
```

### Access Keys

For development or specific use cases:

```yaml
config:
  bucket_name: my-bucket
  region: us-east-1
  aws_access_key_id: AKIAIOSFODNN7EXAMPLE
  aws_secret_access_key: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

## S3 Bucket Setup

### Required Permissions

The S3 provider requires the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

### Automatic Bucket Creation

By default, the S3 provider expects the bucket to already exist. If you want the provider to automatically create the bucket when it doesn't exist, set `auto_create_bucket: true` in your configuration:

```yaml
config:
  bucket_name: my-bucket
  auto_create_bucket: true  # Will create bucket if it doesn't exist
  region: us-east-1
```

**Note**: When `auto_create_bucket` is enabled, the provider will need additional permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:CreateBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

### Bucket Policy (Optional)

For additional security, you can add a bucket policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "OGXAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::YOUR-ACCOUNT:role/OGXRole"
      },
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    },
    {
      "Sid": "OGXBucketAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::YOUR-ACCOUNT:role/OGXRole"
      },
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name"
    }
  ]
}
```

## Implementation Details

### Metadata Persistence

File metadata is stored in a SQL database for fast queries and OpenAI API compatibility. The metadata includes:

- File ID
- Original filename
- Purpose (assistants, batch, etc.)
- File size in bytes
- Created and expiration timestamps

### TTL and Cleanup

Files currently have a fixed long expiration time (100 years).

## Development and Testing

### Using MinIO

For self-hosted S3-compatible storage:

```yaml
config:
  bucket_name: test-bucket
  region: us-east-1
  endpoint_url: http://localhost:9000
  aws_access_key_id: minioadmin
  aws_secret_access_key: minioadmin
```

### Using OCI Object Storage with S3 Compatibility

[Official Object Storage Amazon S3 Compatibility API Documentation](https://docs.oracle.com/en-us/iaas/Content/Object/Tasks/s3compatibleapi.htm)

OCI Object Storage can be utilized through the OCI S3 Compatibility API. Simply Update the `config.yaml` and set the env-vars as below.

#### config.yaml

```yaml
provider_type: remote::s3
config:
  bucket_name: "${env.S3_BUCKET_NAME}"
  region: "${env.AWS_REGION:=us-east-1}"
  aws_access_key_id: "${env.AWS_ACCESS_KEY_ID:=}"
  aws_secret_access_key: "${env.AWS_SECRET_ACCESS_KEY:=}"
  endpoint_url: "${env.S3_ENDPOINT_URL:=}"
  metadata_store:
    table_name: files_metadata
    backend: sql_default
```

#### .env

```bash
AWS_ACCESS_KEY_ID=OCI_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=OCI_SECRET_KEY
S3_BUCKET_NAME=OCI_BUCKET_NAME
S3_ENDPOINT_URL=https://<namespace>.compat.objectstorage.<region>.oci.customer-oci.com
AWS_REQUEST_CHECKSUM_CALCULATION=when_required
AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
```

## Monitoring and Logging

The provider logs important operations and errors. For production deployments, consider:

- CloudWatch monitoring for S3 operations
- Custom metrics for file upload/download rates
- Error rate monitoring
- Performance metrics tracking

## Error Handling

The provider handles various error scenarios:

- S3 connectivity issues
- Bucket access permissions
- File not found errors
- Metadata consistency checks

## Known Limitations

- Fixed long TTL (100 years) instead of configurable expiration
- No server-side encryption enabled by default
- No support for AWS session tokens
- No S3 key prefix organization support
- No multipart upload support (all files uploaded as single objects)
