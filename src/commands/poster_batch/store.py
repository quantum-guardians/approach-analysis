"""Store abstractions for distributed batch results."""

from __future__ import annotations

import json
from typing import Any, Iterable, Protocol

from src.commands.poster_batch.schema import json_default


class Store(Protocol):
    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        """Store a JSON payload by key."""

    def iter_json(self, prefix: str) -> Iterable[dict[str, Any]]:
        """Yield JSON payloads under prefix."""


class S3Store:
    def __init__(self, s3_client: Any, bucket: str):
        self.s3_client = s3_client
        self.bucket = bucket

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(
                payload,
                indent=2,
                default=json_default,
                allow_nan=True,
            ).encode("utf-8"),
            ContentType="application/json",
        )

    def iter_json(self, prefix: str) -> Iterable[dict[str, Any]]:
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if not key.endswith(".json"):
                    continue
                obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
                yield json.loads(obj["Body"].read().decode("utf-8"))


def get_s3_client() -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Install boto3 package to use poster-batch S3 storage.") from exc

    return boto3.client("s3")
