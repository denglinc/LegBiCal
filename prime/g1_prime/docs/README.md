# G1 GitHub Pages

[`index.html`](index.html) immediately loads the first of two self-contained G1
Meshcat replays. Its compact toolbar identifies the estimated and ground-truth
robots and selects the segment and speed; generated replay HTML is not stored
in Git history.

## Build locally

```bash
python scripts/publish_pages.py
```

Run this from `prime/g1_prime`. It creates `out/pages_site/`, starts segment 1
at `0.5x`, alternates the two segments in a loop, and retains the selected
`0.5x` or `1x` rate. Only one replay occupies the iframe at a time.

## GitHub Pages setting

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions** as the source.
3. Open **Actions**, select **Build and deploy G1 PRIME calibrated pages**, and
   run the workflow manually.

The [Pages workflow](../../../.github/workflows/pages.yml) uploads
`out/pages_site` as an artifact. Its approximately 104 MiB total is below the
[1 GB Pages limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits),
and its roughly 54 MB replay files bypass regular-Git size warnings because
they are generated artifacts, not commits.

Return to the [G1 implementation](../README.md).
