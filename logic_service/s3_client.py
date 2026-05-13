"""
S3/MinIO helper for logic_service.

Generates presigned URLs for audio objects stored as s3://bucket/key strings.
"""
import os
from typing import Optional, Tuple

import boto3
from botocore.client import Config


def _client(endpoint=None):
    if not endpoint:
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
    return bool(os.getenv("S3_ENDPOINT_URL") and os.getenv("S3_ACCESS_KEY_ID") and os.getenv("S3_SECRET_ACCESS_KEY"))


def parse_s3_ref(ref: str) -> Optional[Tuple[str, str]]:
    if not ref or not ref.startswith("s3://"):
        return None
    rest = ref[len("s3://") :]
    if "/" not in rest:
        return None
    bucket, key = rest.split("/", 1)
    return bucket, key


def presign_get_url(ref: str, expires_seconds: int = 600) -> Optional[str]:
    parsed = parse_s3_ref(ref)
    if not parsed:
        return None
    bucket, key = parsed
    
    # If a public endpoint is provided, use it for the presigned URL so the browser can reach it.
    # Otherwise, fall back to the internal endpoint.
    public_endpoint = os.getenv("S3_PUBLIC_ENDPOINT_URL")
    c = _client(endpoint=public_endpoint if public_endpoint else None)
    
    return c.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_seconds,
    )

