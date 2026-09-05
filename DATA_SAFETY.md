# Data Safety and Ethical Release Notes

This repository is a sanitized code release.

It intentionally excludes raw password datasets and password-level prediction audit files. The manuscript and repository report only aggregate metrics and figures.

## Excluded by design

- raw password files (`*.pw`)
- converted benchmark CSV files containing a `password` column
- prediction audit CSV files
- original benchmark ZIP archives
- user identifiers, emails, IP addresses, phone numbers, account IDs
- virtual environments and IDE metadata

## Public release principle

The repository should allow reviewers to inspect and run the code, but should not redistribute sensitive password material.

Users who want to reproduce the experiments must obtain the benchmark data separately from the original source and process it locally.
