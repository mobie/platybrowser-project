# Release 1.0.1b

`1.0.1b` is a metadata-only re-point: the data in `data/1.0.1` is unchanged, but
every `bdv.n5.s3` source now targets the new NetApp bucket
(`https://buckets.embl.de`, bucket `platybrowser-n5-generic`).

There is **no new dataset directory**. The path `data/1.0.1` stays exactly as it
is; the release is marked **only by a git tag**.

## Steps (after the re-point is merged)

1. Confirm the change is metadata-only:
   ```bash
   git diff --stat <previous-tag>..HEAD    # only *.xml loader lines should appear
   ```
2. Tag the release commit and push the tag:
   ```bash
   git tag -a 1.0.1b -m "PlatyBrowser 1.0.1b: S3 sources re-pointed to buckets.embl.de (platybrowser-n5-generic)"
   git push origin 1.0.1b
   ```
3. No `data/project.json` change and no `data/1.1.0` (or `data/1.0.1b`) directory
   are needed.

## Notes

- Version digits are significant in this project: minor bumps (`1.1.0`) denoted
  segmentation updates, so a metadata-only re-point must NOT take a version
  number — hence the `b` suffix.
- Bucket layout is 1:1 with the old Minio buckets, using the old bucket name as
  a key prefix:
  - old bucket `platybrowser` → key `platybrowser/…`
  - old bucket `platybrowser-2025` → key `platybrowser-2025/…`
  - old bucket `platybrowser-all-hcrs-temp` → key `platybrowser-all-hcrs-temp/…`
- Historical: `misc/outdated/update_minor.py` was the old automated release
  helper; it is not used here.
