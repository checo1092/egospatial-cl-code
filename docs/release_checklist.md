# Release Checklist

## Initial GitHub Repository

- [x] Create public repository.
- [x] Push this scaffold.
- [x] Confirm repository URL: `https://github.com/checo1092/egospatial-cl-code`.
- [x] Add repository URL to `CITATION.cff`.
- [x] Use `checo1092/egospatial-cl-code` as the code repository unless a later conflict is found.

## Before Adding Generation Scripts

- [ ] Audit scripts for local paths.
- [ ] Remove or parameterize machine-specific paths.
- [ ] Separate reusable configs from experiment outputs.
- [ ] Document Python environments and dependencies.
- [ ] Verify no dataset shards or checkpoints are committed.

## Before Code DOI

- [ ] Add audited generation scripts.
- [ ] Add minimal smoke-test or validation command.
- [ ] Freeze a code release tag.
- [ ] Archive the release through Zenodo or equivalent.
- [ ] Update `CITATION.cff` with the code DOI.

## Before Dataset DOI

- [ ] Dataset repository is public and non-gated.
- [ ] Dataset license is CC BY 4.0.
- [ ] Version is `v1.0.0`.
- [ ] Dataset card has no local paths.
- [ ] Checksums and shard manifest are published.
- [ ] Dataset DOI is minted only after final metadata review.
