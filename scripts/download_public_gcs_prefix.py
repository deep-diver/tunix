#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Downloads a public GCS prefix over HTTPS without gcloud credentials."""

from __future__ import annotations

import argparse
from concurrent import futures
import json
import pathlib
import tempfile
import time
import urllib.parse
import urllib.request


def _json_event(**payload) -> None:
  print(json.dumps(payload, sort_keys=True), flush=True)


def _list_objects(bucket: str, prefix: str) -> list[dict[str, str]]:
  items = []
  page_token = None
  while True:
    query = {
        "prefix": prefix,
        "fields": "items(name,size),nextPageToken",
    }
    if page_token:
      query["pageToken"] = page_token
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?"
        + urllib.parse.urlencode(query)
    )
    with urllib.request.urlopen(url, timeout=120) as response:
      data = json.load(response)
    items.extend(data.get("items", []))
    page_token = data.get("nextPageToken")
    if not page_token:
      return items


def _object_url(bucket: str, name: str) -> str:
  return f"https://storage.googleapis.com/{bucket}/" + urllib.parse.quote(
      name, safe="/"
  )


def _download_one(
    *,
    bucket: str,
    obj: dict[str, str],
    prefix: str,
    dest: pathlib.Path,
) -> dict[str, object]:
  name = obj["name"]
  size = int(obj.get("size", 0))
  rel = pathlib.Path(name[len(prefix) :])
  target = dest / rel
  target.parent.mkdir(parents=True, exist_ok=True)

  if target.exists() and target.stat().st_size == size:
    return {"name": name, "size": size, "status": "cached"}

  url = _object_url(bucket, name)
  with tempfile.NamedTemporaryFile(
      dir=str(target.parent), delete=False
  ) as tmp_file:
    tmp_path = pathlib.Path(tmp_file.name)
    with urllib.request.urlopen(url, timeout=120) as response:
      while True:
        chunk = response.read(16 * 1024 * 1024)
        if not chunk:
          break
        tmp_file.write(chunk)

  if tmp_path.stat().st_size != size:
    tmp_path.unlink(missing_ok=True)
    raise IOError(
        f"Short download for {name}: got {tmp_path.stat().st_size}, want {size}"
    )
  tmp_path.replace(target)
  return {"name": name, "size": size, "status": "downloaded"}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bucket", required=True)
  parser.add_argument("--prefix", required=True)
  parser.add_argument("--dest", required=True)
  parser.add_argument("--workers", type=int, default=8)
  args = parser.parse_args()

  prefix = args.prefix
  if not prefix.endswith("/"):
    prefix += "/"
  dest = pathlib.Path(args.dest)
  dest.mkdir(parents=True, exist_ok=True)

  t0 = time.time()
  objects = _list_objects(args.bucket, prefix)
  total_bytes = sum(int(obj.get("size", 0)) for obj in objects)
  _json_event(
      event="download_plan",
      objects=len(objects),
      total_bytes=total_bytes,
      total_gib=round(total_bytes / 2**30, 3),
      dest=str(dest),
  )

  done_bytes = 0
  with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
    future_to_obj = {
        executor.submit(
            _download_one,
            bucket=args.bucket,
            obj=obj,
            prefix=prefix,
            dest=dest,
        ): obj
        for obj in objects
    }
    for future in futures.as_completed(future_to_obj):
      result = future.result()
      done_bytes += int(result["size"])
      _json_event(
          event="download_object",
          done_gib=round(done_bytes / 2**30, 3),
          total_gib=round(total_bytes / 2**30, 3),
          **result,
      )

  _json_event(
      event="download_complete",
      objects=len(objects),
      total_gib=round(total_bytes / 2**30, 3),
      seconds=round(time.time() - t0, 3),
      dest=str(dest),
  )


if __name__ == "__main__":
  main()
