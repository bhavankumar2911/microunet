# Wilcoxon Statistical Test Report

Generated: 2026-07-18 23:44:50

Datasets: 40
Significance threshold: α = 0.05
Non-inferiority δ range: 0.001 – 0.050

---

## Depth-wise separable layers (16-32-64) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 318.0 | 0.8919 | not significant |

> Cannot claim superiority (p = 0.8919). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 326.0 | 0.8704 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 332.0 | 0.8526 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 354.0 | 0.7741 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 387.0 | 0.6227 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 425.0 | 0.4236 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 468.0 | 0.2219 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 494.0 | 0.1324 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 525.0 | 0.0624 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 558.0 | 0.0233 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 584.0 | 0.0093 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 609.0 | 0.0033 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 641.0 | 0.0007 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.035**  
> Non-inferior as long as a Dice loss up to 0.035 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (16-32-64) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7234 |
| Mean Δ Dice | -0.0118 |
| Std Δ Dice | 0.0699 |
| Datasets improved | 15/40 |
| Datasets regressed | 25/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 17,383 |
| Parameter reduction | 78,534 (81.9%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (8-16-32-52) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 124.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 133.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 142.0 | 0.9999 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 171.0 | 0.9996 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 218.0 | 0.9957 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 293.0 | 0.9425 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 332.0 | 0.8526 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 394.0 | 0.5868 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 432.0 | 0.3875 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 462.0 | 0.2465 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 497.0 | 0.1240 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 517.0 | 0.0769 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 550.0 | 0.0301 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.050**  
> Non-inferior as long as a Dice loss up to 0.050 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (8-16-32-52) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6976 |
| Mean Δ Dice | -0.0376 |
| Std Δ Dice | 0.0626 |
| Datasets improved | 7/40 |
| Datasets regressed | 33/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 15,833 |
| Parameter reduction | 80,084 (83.5%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (8-16-32-64) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 125.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 132.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 134.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 158.0 | 0.9998 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 213.0 | 0.9965 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 299.5 | 0.9325 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 375.0 | 0.6816 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 417.0 | 0.4656 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 459.0 | 0.2593 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 507.0 | 0.0984 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 538.0 | 0.0433 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 570.0 | 0.0155 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 608.0 | 0.0035 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.040**  
> Non-inferior as long as a Dice loss up to 0.040 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (8-16-32-64) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7026 |
| Mean Δ Dice | -0.0325 |
| Std Δ Dice | 0.0599 |
| Datasets improved | 6/40 |
| Datasets regressed | 34/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 18,101 |
| Parameter reduction | 77,816 (81.1%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (10-20-40-80) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 222.0 | 0.9949 | not significant |

> Cannot claim superiority (p = 0.9949). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 234.0 | 0.9917 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 244.0 | 0.9880 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 320.0 | 0.8867 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 445.0 | 0.3232 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 505.0 | 0.1032 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 606.0 | 0.0038 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 649.0 | 0.0005 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 674.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 722.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 738.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 763.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 773.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.020**  
> Non-inferior as long as a Dice loss up to 0.020 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (10-20-40-80) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7253 |
| Mean Δ Dice | -0.0099 |
| Std Δ Dice | 0.0330 |
| Datasets improved | 9/40 |
| Datasets regressed | 31/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 27,265 |
| Parameter reduction | 68,652 (71.6%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (12-24-48-96) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 330.0 | 0.8587 | not significant |

> Cannot claim superiority (p = 0.8587). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 344.0 | 0.8123 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 361.5 | 0.7450 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 435.0 | 0.3723 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 536.0 | 0.0459 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 589.0 | 0.0076 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 646.0 | 0.0006 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 695.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 741.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 767.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 787.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 801.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 808.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.010**  
> Non-inferior as long as a Dice loss up to 0.010 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (12-24-48-96) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7327 |
| Mean Δ Dice | -0.0025 |
| Std Δ Dice | 0.0323 |
| Datasets improved | 14/40 |
| Datasets regressed | 26/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 38,286 |
| Parameter reduction | 57,631 (60.1%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (14-28-56-112) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 472.5 | 0.2063 | not significant |

> Cannot claim superiority (p = 0.2063). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 494.0 | 0.1324 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 520.0 | 0.0712 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 596.0 | 0.0058 | **NON-INFERIOR ✓** | Loss within 0.005 |
| 0.010 | 673.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 727.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 746.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 752.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 760.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 772.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 777.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.005**  
> Non-inferior as long as a Dice loss up to 0.005 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (14-28-56-112) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7410 |
| Mean Δ Dice | +0.0059 |
| Std Δ Dice | 0.0540 |
| Datasets improved | 20/40 |
| Datasets regressed | 20/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 51,162 |
| Parameter reduction | 44,755 (46.7%) |


</td>
</tr>
</table>

---

## Depth-wise separable convolution (16, 32, 64, 128) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 387.5 | 0.6227 | not significant |

> Cannot claim superiority (p = 0.6227). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 415.0 | 0.4762 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 433.0 | 0.3824 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 489.0 | 0.1474 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 574.0 | 0.0134 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 616.0 | 0.0024 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 669.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 706.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 728.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 742.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 753.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 765.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 781.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.010**  
> Non-inferior as long as a Dice loss up to 0.010 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable convolution (16, 32, 64, 128) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7346 |
| Mean Δ Dice | -0.0006 |
| Std Δ Dice | 0.0368 |
| Datasets improved | 17/40 |
| Datasets regressed | 23/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 65,895 |
| Parameter reduction | 30,022 (31.3%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (4-8-16-32-64) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 21.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 24.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 24.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 31.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 54.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 80.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 107.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 124.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 151.0 | 0.9999 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 176.0 | 0.9994 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 204.0 | 0.9977 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 233.0 | 0.9921 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 259.0 | 0.9796 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (4-8-16-32-64) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6346 |
| Mean Δ Dice | -0.1005 |
| Std Δ Dice | 0.1301 |
| Datasets improved | 2/40 |
| Datasets regressed | 38/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 18,332 |
| Parameter reduction | 77,585 (80.9%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (5-10-20-40-80) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 32.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 34.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 36.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 54.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 86.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 112.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 134.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 156.0 | 0.9998 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 181.0 | 0.9992 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 214.0 | 0.9964 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 251.0 | 0.9845 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 283.0 | 0.9567 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 322.0 | 0.8815 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (5-10-20-40-80) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6617 |
| Mean Δ Dice | -0.0734 |
| Std Δ Dice | 0.0790 |
| Datasets improved | 3/40 |
| Datasets regressed | 37/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 27,594 |
| Parameter reduction | 68,323 (71.2%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (8-16-32-64-128) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 100.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 105.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 114.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 149.0 | 0.9999 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 203.0 | 0.9978 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 258.0 | 0.9803 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 313.0 | 0.9039 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 358.0 | 0.7577 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 409.0 | 0.5080 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 456.0 | 0.2724 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 491.0 | 0.1413 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 530.0 | 0.0544 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 562.0 | 0.0204 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.050**  
> Non-inferior as long as a Dice loss up to 0.050 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (8-16-32-64-128) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6848 |
| Mean Δ Dice | -0.0503 |
| Std Δ Dice | 0.1014 |
| Datasets improved | 10/40 |
| Datasets regressed | 30/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 66,613 |
| Parameter reduction | 29,304 (30.6%) |


</td>
</tr>
</table>

---

## Additive skip connection vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 132.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 162.0 | 0.9997 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 199.0 | 0.9982 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 312.0 | 0.9062 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 394.0 | 0.5868 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 453.0 | 0.2859 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 515.0 | 0.0809 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 536.0 | 0.0459 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 565.0 | 0.0184 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 587.0 | 0.0083 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 631.0 | 0.0012 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 647.0 | 0.0005 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 656.0 | 0.0003 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.025**  
> Non-inferior as long as a Dice loss up to 0.025 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Additive skip connection |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7071 |
| Mean Δ Dice | -0.0281 |
| Std Δ Dice | 0.0701 |
| Datasets improved | 6/40 |
| Datasets regressed | 34/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 83,821 |
| Parameter reduction | 12,096 (12.6%) |


</td>
</tr>
</table>

---

## Single convolution per block (8-16-32-52) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 103.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 111.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 118.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 147.0 | 0.9999 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 198.0 | 0.9982 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 253.0 | 0.9834 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 297.0 | 0.9359 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 348.0 | 0.7975 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 398.0 | 0.5660 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 437.0 | 0.3623 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 483.0 | 0.1668 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 515.0 | 0.0809 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 541.0 | 0.0397 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.050**  
> Non-inferior as long as a Dice loss up to 0.050 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Single convolution per block (8-16-32-52) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6954 |
| Mean Δ Dice | -0.0398 |
| Std Δ Dice | 0.0574 |
| Datasets improved | 8/40 |
| Datasets regressed | 32/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 47,389 |
| Parameter reduction | 48,528 (50.6%) |


</td>
</tr>
</table>

---

## Single convolution per block (8-16-32-64) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 74.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 80.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 88.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 120.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 195.0 | 0.9985 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 267.0 | 0.9735 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 339.0 | 0.8298 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 400.0 | 0.5555 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 463.0 | 0.2423 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 531.0 | 0.0529 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 591.0 | 0.0071 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 632.0 | 0.0011 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 658.0 | 0.0003 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.040**  
> Non-inferior as long as a Dice loss up to 0.040 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Single convolution per block (8-16-32-64) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7066 |
| Mean Δ Dice | -0.0286 |
| Std Δ Dice | 0.0424 |
| Datasets improved | 5/40 |
| Datasets regressed | 35/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 51,229 |
| Parameter reduction | 44,688 (46.6%) |


</td>
</tr>
</table>

---

## Single convolution per block (10-20-40-80) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 231.0 | 0.9927 | not significant |

> Cannot claim superiority (p = 0.9927). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 243.0 | 0.9884 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 253.0 | 0.9834 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 292.0 | 0.9441 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 379.0 | 0.6624 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 435.0 | 0.3723 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 498.0 | 0.1212 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 557.0 | 0.0241 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 606.0 | 0.0038 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 650.0 | 0.0004 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 676.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 702.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 718.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.025**  
> Non-inferior as long as a Dice loss up to 0.025 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Single convolution per block (10-20-40-80) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7190 |
| Mean Δ Dice | -0.0161 |
| Std Δ Dice | 0.0382 |
| Datasets improved | 12/40 |
| Datasets regressed | 28/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 79,995 |
| Parameter reduction | 15,922 (16.6%) |


</td>
</tr>
</table>

---

## Single convolution per block (4-8-16-32-64) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 56.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 57.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 63.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 84.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 139.0 | 0.9999 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 182.0 | 0.9992 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 220.0 | 0.9953 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 255.0 | 0.9822 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 294.0 | 0.9409 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 334.0 | 0.8463 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 357.0 | 0.7619 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 413.0 | 0.4867 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 451.0 | 0.2950 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Single convolution per block (4-8-16-32-64) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6775 |
| Mean Δ Dice | -0.0576 |
| Std Δ Dice | 0.0827 |
| Datasets improved | 3/40 |
| Datasets regressed | 37/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 51,760 |
| Parameter reduction | 44,157 (46.0%) |


</td>
</tr>
</table>

---

## Single convolution per block (5-10-20-40-80) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 76.0 | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 78.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 84.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 111.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 171.0 | 0.9996 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 223.0 | 0.9947 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 269.0 | 0.9717 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 339.0 | 0.8298 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 404.0 | 0.5344 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 459.0 | 0.2593 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 504.0 | 0.1057 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 549.0 | 0.0311 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 586.0 | 0.0086 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.045**  
> Non-inferior as long as a Dice loss up to 0.045 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Single convolution per block (5-10-20-40-80) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6969 |
| Mean Δ Dice | -0.0382 |
| Std Δ Dice | 0.0520 |
| Datasets improved | 5/40 |
| Datasets regressed | 35/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 80,849 |
| Parameter reduction | 15,068 (15.7%) |


</td>
</tr>
</table>

---
