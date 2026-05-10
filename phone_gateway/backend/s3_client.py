"""
S3/MinIO helper for phone_gateway.

Uploads call audio to S3 and returns an s3://bucket/key reference.
"""
import os
from typing import Optional

import boto3
from botocore.client import Config


def _client():
    endpoint = os.getenv("S3_ENDPOINT_URL")
    region = os.getenv("S3_REGION", "us-east-1")

    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
        region_name=region,
        config=Config(signature_version="s3v4"),
    )


def is_enabled() -> bool:
    return bool(os.getenv("S3_ENDPOINT_URL") and os.getenv("S3_BUCKET"))


def ensure_bucket(bucket: str) -> None:
    c = _client()
    try:
        c.head_bucket(Bucket=bucket)
    except Exception:
        c.create_bucket(Bucket=bucket)


def upload_file(local_path: str, key: str, content_type: Optional[str] = None) -> str:
    """
    Upload local_path -> s3://bucket/key, returns the s3:// reference.
    """
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET is not set")

    c = _client()
    ensure_bucket(bucket)

    extra = {}
    if content_type:
        extra["ContentType"] = content_type

    c.upload_file(local_path, bucket, key, ExtraArgs=extra or None)
    return f"s3://{bucket}/{key}"

