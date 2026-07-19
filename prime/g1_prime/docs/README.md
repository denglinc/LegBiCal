# G1 GitHub Pages

Source landing page for the calibrated G1 Meshcat segments. The two
self-contained replays are generated from the shipped solutions and are not
stored in Git history.

## Contents

| Path | Responsibility |
|---|---|
| [`index.html`](index.html) | One auto-starting viewer with compact segment and speed selectors |

## Build locally

From the G1 implementation directory:

```bash
python scripts/publish_pages.py
```

This creates `out/pages_site/` with the landing page and
`media/run{1,2}_calibrated.html`. The landing page immediately starts segment 1
at `0.5x`, loads segment 2 when its animation finishes, and loops. The two
compact selectors can restart either segment at `0.5x` or `1x`; the selected
rate continues across the loop. Only one self-contained replay is resident in
the iframe at a time.

## GitHub Pages setting

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions** as the source.
3. Open **Actions**, select **Build and deploy G1 PRIME calibrated pages**, and
   run the workflow manually.

The workflow at
[`../../../.github/workflows/pages.yml`](../../../.github/workflows/pages.yml)
builds and uploads the generated `out/pages_site` tree as one Pages artifact. Its
approximately 104 MiB total is below GitHub Pages' 1 GB published-site limit;
the generated files bypass the 50 MiB regular-Git warning because they are
artifact content, not commits.

See GitHub's official documentation for
[selecting a Pages publishing source](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
and [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).

Return to the [G1 implementation](../README.md).
