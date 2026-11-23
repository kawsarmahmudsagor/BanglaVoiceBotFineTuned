# UI check: model catalog and web redesign

Date: 2026-11-23. Branch `feat/catalog-ui`. Spec §6.4.

## Setup

- API, worker and web dev server run locally on Windows, with SQLite and an isolated data directory, `CATALOG_DIR=<repo>/catalog`.
- Seeded through the real API:
  - dataset `bn-ecommerce-chat` (the README `smoke.jsonl`);
  - recipes `smoke` (20 steps) and `smoke-long` (30 steps), backend `fake`;
  - runs `r-8b856411` and `r-3b19c700`, both done and promoted (the first is LLM champion), and run `r-e46c07a7`, cancelled while training.
- Screenshots were taken with Playwright (`playwright-core`, Chrome) with `prefers-reduced-motion: reduce` emulated. Ten pages, each in dark and light at 1440×900 and 375×812, full page. Files are named `<page>-<theme>-<width>.png`.
- Automatic checks ran on every page shot; `results.json` holds the raw output.

## Results

| Check | Result | How |
|---|---|---|
| Text contrast ≥ 4.5:1 | Pass after fix | Computed for every text token pair (table below). The spec's light saffron `#A86A00` gave 4.43:1 on white, so it is `#9A6100` (4.80:1 on ink, 5.14:1 under white button text). |
| Chart lines ≥ 3:1 | Pass after fix | Dark magenta `#B5367A` was 2.82:1 on panels, now `#D14A92`. Light saffron and orange lines are darkened to `#9A6100` and `#C9542A`. |
| Keyboard focus visible | Pass | On `/runs`, 8 Tab presses in each theme; every focused element showed a 2 px solid saffron outline. Ctrl+K opens the palette with the search input focused. |
| No horizontal page scroll at 375 px | Pass | `scrollWidth <= clientWidth` on all 20 narrow shots. Tables scroll inside their panel (see `runs-*-375.png`). |
| `prefers-reduced-motion` respected | Pass | With reduce emulated, there were no running CSS animations on any page. The global rule sets transitions to 0 s. |
| Icon buttons have accessible names | Pass | No visible button, link, tab or form control without an accessible name on any page shot. |
| Status not shown by colour alone | Pass | Run states use icon + label + colour (`StatusBadge`). The phase timeline has icons and hidden text. Compare marks "Differs" and "Best" in text. Licenses read "commercial OK" / "non-commercial". |
| Theme applied before paint | Pass | The stored theme matched `<html class>` on every shot. |
| No runtime errors | Pass | No page errors or console errors on any shot. |

### Contrast ratios

| Pair | Colours | Ratio |
|---|---|---|
| dark paper on ink | `#E7E9F0` on `#161A24` | 14.34:1 |
| dark muted on slate | `#9BA2B6` on `#1E2331` | 6.15:1 |
| dark muted on raised | `#9BA2B6` on `#272D3E` | 5.38:1 |
| dark saffron on ink | `#F4B740` on `#161A24` | 9.69:1 |
| dark ink on saffron (button) | `#161A24` on `#F4B740` | 9.69:1 |
| dark running on slate | `#5CC8E0` on `#1E2331` | 8.05:1 |
| dark done on slate | `#62D394` on `#1E2331` | 8.41:1 |
| dark failed on slate | `#F06B7E` on `#1E2331` | 5.31:1 |
| light paper on ink | `#1A1F2C` on `#F6F7FA` | 15.36:1 |
| light muted on slate | `#5A6275` on `#FFFFFF` | 6.11:1 |
| light muted on raised | `#5A6275` on `#EEF0F5` | 5.36:1 |
| light saffron on ink | `#9A6100` on `#F6F7FA` | 4.80:1 |
| light white on saffron (button) | `#FFFFFF` on `#9A6100` | 5.14:1 |
| light running on slate | `#0E7490` on `#FFFFFF` | 5.36:1 |
| light done on slate | `#15803D` on `#FFFFFF` | 5.02:1 |
| light failed on slate | `#BE123C` on `#FFFFFF` | 6.29:1 |
| light chart saffron | `#9A6100` on `#FFFFFF` | 5.14:1 |
| light chart orange | `#C9542A` on `#FFFFFF` | 4.39:1 |
| light chart magenta | `#B5367A` on `#FFFFFF` | 5.57:1 |
| dark chart magenta | `#B5367A` on `#1E2331` | 2.82:1 (failed; replaced by `#D14A92`, about 3.9:1) |
| dark chart purple | `#7A5CC7` on `#1E2331` | 3.12:1 |

## Problems found by looking at the screenshots, and fixed

1. JetBrains Mono ligatures drew `<|bn|>` as arrows. Ligatures are now off for mono text, code and pre.
2. The saffron ring meant for "first is recommended" appeared on every tag group. It now marks only the first fine-tune method.
3. The active nav item's matra read as an underline of the item above it. Active items now also get a raised background.
4. The recipe editor flagged "name must be a kebab-case slug" before anything was typed. An empty name now shows "Name the recipe to save it."
5. Next's floating dev badge covered the sidebar's Collapse button. It is turned off with `devIndicators: false`.
6. A finished run still showed its last status line ("0 steps exporting"). The line is hidden once a run ends.

## Verification

- `uv run pytest -q`: `208 passed, 10 skipped in 45.37s`
- `npm run test`: `Test Files 25 passed (25)`, `Tests 164 passed (164)`
- `npm run typecheck`: no errors
- `npm run build`: `✓ Compiled successfully`, all routes built, including `/catalog`, `/catalog/[id]`, `/runs/compare` and `/registry`.
