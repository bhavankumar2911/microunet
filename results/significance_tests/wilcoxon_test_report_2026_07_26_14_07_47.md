# Wilcoxon Statistical Test Report

Generated: 2026-07-26 14:07:46

Datasets: 40
Significance threshold: α = 0.05
Non-inferiority δ range: 0.001 – 0.050
P-value correction: bonferroni (applied once across all 40 superiority tests in this report)

---

## Depth-wise separable layers (8-16-32) + triple convolution vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 30.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 30.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 33.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 40.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 66.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 109.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 149.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 184.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 215.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 269.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 324.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 370.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 400.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (8-16-32) + triple convolution |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6663 |
| Mean Δ Dice | -0.0689 |
| Std Δ Dice | 0.0844 |
| Datasets improved | 2/40 |
| Datasets regressed | 38/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 7,493 |
| Parameter reduction | 88,424 (92.2%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (8-16-32) + triple convolution + residual block vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 47.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 53.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 57.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 69.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 94.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 120.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 150.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 179.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 205.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 234.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 265.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 299.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 334.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (8-16-32) + triple convolution + residual block |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6661 |
| Mean Δ Dice | -0.0691 |
| Std Δ Dice | 0.0770 |
| Datasets improved | 5/40 |
| Datasets regressed | 35/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 8,787 |
| Parameter reduction | 87,129 (90.8%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (12-24-48) + triple convolution vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 188.0 | 0.9989 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9989). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 195.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 202.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 227.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 280.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 336.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 394.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 436.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 477.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 520.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 544.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 581.0 | 0.4151 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 608.0 | 0.1391 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (12-24-48) + triple convolution |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7100 |
| Mean Δ Dice | -0.0252 |
| Std Δ Dice | 0.0559 |
| Datasets improved | 11/40 |
| Datasets regressed | 29/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 15,174 |
| Parameter reduction | 80,743 (84.2%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (12-24-48) + triple convolution + residual block vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 72.5 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 79.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 80.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 89.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 120.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 148.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 176.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 208.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 252.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 292.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 323.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 356.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 393.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (12-24-48) + triple convolution + residual block |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6728 |
| Mean Δ Dice | -0.0624 |
| Std Δ Dice | 0.0750 |
| Datasets improved | 6/40 |
| Datasets regressed | 34/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 18,076 |
| Parameter reduction | 77,841 (81.2%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (12-24-48) + triple convolution per block + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 123.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 129.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 140.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 177.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 226.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 287.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 351.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 393.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 442.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 482.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 506.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 550.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 576.0 | 0.4995 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (12-24-48) + triple convolution per block + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7029 |
| Mean Δ Dice | -0.0323 |
| Std Δ Dice | 0.0478 |
| Datasets improved | 8/40 |
| Datasets regressed | 32/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 16,934 |
| Parameter reduction | 78,983 (82.3%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (16-32-64) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 318.0 | 0.8919 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.8919). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 326.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 332.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 354.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 387.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 425.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 468.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 494.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 525.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 558.0 | 0.9322 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 584.0 | 0.3705 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 609.0 | 0.1332 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 641.0 | 0.0288 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.050**  
> Non-inferior as long as a Dice loss up to 0.050 is acceptable.


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

## Depth-wise separable layers (16-32-64) + triple convolution vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 310.0 | 0.9107 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9107). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 323.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 328.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 371.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 429.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 481.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 535.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 584.0 | 0.3705 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 628.0 | 0.0554 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 653.0 | 0.0151 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 680.0 | 0.0030 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 700.0 | 0.0008 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 718.0 | 0.0002 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.035**  
> Non-inferior as long as a Dice loss up to 0.035 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (16-32-64) + triple convolution |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7235 |
| Mean Δ Dice | -0.0117 |
| Std Δ Dice | 0.0431 |
| Datasets improved | 14/40 |
| Datasets regressed | 26/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 25,479 |
| Parameter reduction | 70,438 (73.4%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (16-32-64) + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 262.0 | 0.9775 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9775). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 273.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 281.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 299.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 341.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 383.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 435.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 473.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 505.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 540.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 571.0 | 0.5979 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 607.0 | 0.1453 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 631.0 | 0.0478 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.050**  
> Non-inferior as long as a Dice loss up to 0.050 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (16-32-64) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7153 |
| Mean Δ Dice | -0.0198 |
| Std Δ Dice | 0.0757 |
| Datasets improved | 12/40 |
| Datasets regressed | 28/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 19,719 |
| Parameter reduction | 76,198 (79.4%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (16-32-64) + triple convolution + residual block vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 156.0 | 0.9998 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9998). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 161.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 167.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 191.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 238.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 291.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 340.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 384.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 432.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 473.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 508.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 537.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 559.0 | 0.9020 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (16-32-64) + triple convolution + residual block |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6892 |
| Mean Δ Dice | -0.0459 |
| Std Δ Dice | 0.0922 |
| Datasets improved | 9/40 |
| Datasets regressed | 31/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 30,628 |
| Parameter reduction | 65,288 (68.1%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 345.0 | 0.8087 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.8087). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 358.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 373.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 413.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 454.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 510.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 561.0 | 0.8439 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 595.0 | 0.2403 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 637.0 | 0.0354 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 666.0 | 0.0071 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 685.0 | 0.0022 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 705.0 | 0.0005 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 728.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.030**  
> Non-inferior as long as a Dice loss up to 0.030 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7339 |
| Mean Δ Dice | -0.0012 |
| Std Δ Dice | 0.0617 |
| Datasets improved | 16/40 |
| Datasets regressed | 24/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 27,815 |
| Parameter reduction | 68,102 (71.0%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (18-36-72) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 261.0 | 0.9782 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9782). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 265.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 271.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 298.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 338.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 380.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 425.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 467.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 500.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 535.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 577.0 | 0.4815 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 609.0 | 0.1332 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 633.0 | 0.0433 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.050**  
> Non-inferior as long as a Dice loss up to 0.050 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (18-36-72) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7160 |
| Mean Δ Dice | -0.0191 |
| Std Δ Dice | 0.0612 |
| Datasets improved | 14/40 |
| Datasets regressed | 26/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 21,571 |
| Parameter reduction | 74,346 (77.5%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (18-36-72) + triple convolution vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 264.0 | 0.9759 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9759). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 277.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 287.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 327.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 392.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 442.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 500.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 563.0 | 0.7889 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 603.0 | 0.1725 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 646.0 | 0.0221 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 687.0 | 0.0019 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 721.0 | 0.0002 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 749.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.035**  
> Non-inferior as long as a Dice loss up to 0.035 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (18-36-72) + triple convolution |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7231 |
| Mean Δ Dice | -0.0121 |
| Std Δ Dice | 0.0391 |
| Datasets improved | 13/40 |
| Datasets regressed | 27/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 31,615 |
| Parameter reduction | 64,302 (67.0%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (18-36-72) + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 312.0 | 0.9062 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9062). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 320.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 325.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 344.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 382.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 416.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 452.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 482.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 520.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 555.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 577.0 | 0.4815 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 607.0 | 0.1453 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 634.0 | 0.0412 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.050**  
> Non-inferior as long as a Dice loss up to 0.050 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (18-36-72) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7206 |
| Mean Δ Dice | -0.0146 |
| Std Δ Dice | 0.0682 |
| Datasets improved | 14/40 |
| Datasets regressed | 26/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 24,195 |
| Parameter reduction | 71,722 (74.8%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (20-40-80) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 342.5 | 0.8194 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.8194). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 349.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 355.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 372.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 428.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 471.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 507.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 554.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 582.0 | 0.3998 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 609.0 | 0.1332 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 645.0 | 0.0233 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 676.0 | 0.0039 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 691.0 | 0.0014 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.040**  
> Non-inferior as long as a Dice loss up to 0.040 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (20-40-80) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7288 |
| Mean Δ Dice | -0.0064 |
| Std Δ Dice | 0.0656 |
| Datasets improved | 14/40 |
| Datasets regressed | 26/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 26,208 |
| Parameter reduction | 69,709 (72.7%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (20-40-80) + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 263.0 | 0.9767 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9767). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 270.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 278.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 298.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 349.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 394.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 437.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 481.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 513.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 549.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 583.0 | 0.3849 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 608.0 | 0.1391 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 630.0 | 0.0503 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (20-40-80) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7130 |
| Mean Δ Dice | -0.0222 |
| Std Δ Dice | 0.0666 |
| Datasets improved | 13/40 |
| Datasets regressed | 27/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 29,120 |
| Parameter reduction | 66,797 (69.6%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (24-48-96) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 328.0 | 0.8065 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.8065). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 356.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 364.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 399.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 448.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 499.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 556.0 | 0.9952 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 590.0 | 0.2936 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 621.0 | 0.0774 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 657.0 | 0.0121 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 682.0 | 0.0026 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 708.0 | 0.0004 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 724.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.035**  
> Non-inferior as long as a Dice loss up to 0.035 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (24-48-96) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7326 |
| Mean Δ Dice | -0.0025 |
| Std Δ Dice | 0.0626 |
| Datasets improved | 13/40 |
| Datasets regressed | 26/40 |
| Datasets unchanged | 1/40 |
| Parameters before | 95,917 |
| Parameters after | 36,825 |
| Parameter reduction | 59,092 (61.6%) |


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 124.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 133.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 142.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 171.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 218.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 293.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 332.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 394.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 432.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 462.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 497.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 517.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 550.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 125.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 132.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 134.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 158.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 213.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 299.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 375.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 417.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 459.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 507.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 538.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 570.0 | 0.6194 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 608.0 | 0.1391 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


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

## Depth-wise separable layers (8-16-32-64) + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 84.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 87.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 93.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 130.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 197.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 254.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 323.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 379.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 426.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 455.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 492.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 511.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 527.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (8-16-32-64) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6903 |
| Mean Δ Dice | -0.0449 |
| Std Δ Dice | 0.0751 |
| Datasets improved | 4/40 |
| Datasets regressed | 36/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 20,837 |
| Parameter reduction | 75,080 (78.3%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (8-16-32-64) + triple convolution per block vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 69.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 77.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 84.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 116.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 165.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 195.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 241.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 308.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 345.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 378.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 422.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 450.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 476.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (8-16-32-64) + triple convolution per block |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6772 |
| Mean Δ Dice | -0.0580 |
| Std Δ Dice | 0.0862 |
| Datasets improved | 5/40 |
| Datasets regressed | 35/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 26,469 |
| Parameter reduction | 69,448 (72.4%) |


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 222.0 | 0.9949 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9949). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 234.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 244.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 320.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 445.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 505.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 606.0 | 0.1517 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 649.0 | 0.0188 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 674.0 | 0.0044 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 722.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 738.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 763.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 773.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.025**  
> Non-inferior as long as a Dice loss up to 0.025 is acceptable.


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

## Depth-wise separable convolution + Additive skip connection (10-20-40-80) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 109.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 115.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 123.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 158.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 255.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 377.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 469.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 543.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 592.0 | 0.2712 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 631.0 | 0.0478 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 662.0 | 0.0090 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 708.0 | 0.0004 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 744.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.035**  
> Non-inferior as long as a Dice loss up to 0.035 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable convolution + Additive skip connection (10-20-40-80) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7151 |
| Mean Δ Dice | -0.0201 |
| Std Δ Dice | 0.0296 |
| Datasets improved | 6/40 |
| Datasets regressed | 34/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 24,535 |
| Parameter reduction | 71,382 (74.4%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (10-20-40-80) + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 233.5 | 0.9921 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9921). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 253.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 273.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 336.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 444.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 526.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 566.0 | 0.7121 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 601.0 | 0.1876 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 654.0 | 0.0143 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 697.0 | 0.0009 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 726.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 755.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 772.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.030**  
> Non-inferior as long as a Dice loss up to 0.030 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (10-20-40-80) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7227 |
| Mean Δ Dice | -0.0124 |
| Std Δ Dice | 0.0300 |
| Datasets improved | 12/40 |
| Datasets regressed | 28/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 30,673 |
| Parameter reduction | 65,244 (68.0%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (10-20-40-80) + triple convolution per block vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 219.0 | 0.9955 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9955). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 228.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 244.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 296.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 392.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 471.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 557.0 | 0.9633 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 606.0 | 0.1517 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 643.0 | 0.0260 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 677.0 | 0.0036 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 704.0 | 0.0006 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 725.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 733.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.030**  
> Non-inferior as long as a Dice loss up to 0.030 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (10-20-40-80) + triple convolution per block |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7220 |
| Mean Δ Dice | -0.0132 |
| Std Δ Dice | 0.0453 |
| Datasets improved | 12/40 |
| Datasets regressed | 28/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 39,845 |
| Parameter reduction | 56,072 (58.5%) |


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 330.0 | 0.8587 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.8587). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 344.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 361.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 435.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 536.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 589.0 | 0.3053 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 646.0 | 0.0221 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 695.0 | 0.0011 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 741.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 767.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 787.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 801.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 808.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.020**  
> Non-inferior as long as a Dice loss up to 0.020 is acceptable.


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

## Depth-wise separable convolution + Additive skip connection (12-24-48-96) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 93.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 104.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 120.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 178.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 293.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 401.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 465.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 511.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 527.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 544.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 576.0 | 0.4995 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 591.0 | 0.2822 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 616.0 | 0.0974 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable convolution + Additive skip connection (12-24-48-96) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.6989 |
| Mean Δ Dice | -0.0363 |
| Std Δ Dice | 0.0769 |
| Datasets improved | 6/40 |
| Datasets regressed | 34/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 34,506 |
| Parameter reduction | 61,411 (64.0%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (12-24-48-96) + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 218.0 | 0.9957 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9957). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 233.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 246.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 316.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 461.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 568.0 | 0.6644 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 629.0 | 0.0528 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 663.0 | 0.0085 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 694.0 | 0.0012 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 709.0 | 0.0004 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 719.0 | 0.0002 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 737.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 773.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.025**  
> Non-inferior as long as a Dice loss up to 0.025 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (12-24-48-96) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7265 |
| Mean Δ Dice | -0.0087 |
| Std Δ Dice | 0.0345 |
| Datasets improved | 9/40 |
| Datasets regressed | 31/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 42,366 |
| Parameter reduction | 53,551 (55.8%) |


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 472.5 | 0.2063 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.2063). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 494.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 520.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 596.0 | 0.2307 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 673.0 | 0.0047 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 727.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 746.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 752.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 760.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 772.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 777.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.010**  
> Non-inferior as long as a Dice loss up to 0.010 is acceptable.


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

## Depth-wise separable convolution + Additive skip connection (14-28-56-112) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 372.0 | 0.6957 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.6957). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 387.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 434.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 536.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 678.0 | 0.0034 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 745.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 775.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 785.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 786.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 787.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 794.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 810.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 817.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.010**  
> Non-inferior as long as a Dice loss up to 0.010 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable convolution + Additive skip connection (14-28-56-112) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7384 |
| Mean Δ Dice | +0.0032 |
| Std Δ Dice | 0.0278 |
| Datasets improved | 14/40 |
| Datasets regressed | 26/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 46,164 |
| Parameter reduction | 49,753 (51.9%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (14-28-56-112) + attention gate vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 354.0 | 0.7741 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.7741). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 377.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 403.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 475.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 575.0 | 0.5180 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 648.0 | 0.0199 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 685.0 | 0.0022 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 705.0 | 0.0005 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 717.0 | 0.0002 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 723.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 736.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 747.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 759.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.015**  
> Non-inferior as long as a Dice loss up to 0.015 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (14-28-56-112) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7352 |
| Mean comparison Dice | 0.7331 |
| Mean Δ Dice | -0.0020 |
| Std Δ Dice | 0.0389 |
| Datasets improved | 16/40 |
| Datasets regressed | 24/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 55,914 |
| Parameter reduction | 40,003 (41.7%) |


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 387.5 | 0.6227 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.6227). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 415.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 433.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 489.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 574.0 | 0.5371 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 616.0 | 0.0974 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 669.0 | 0.0060 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 706.0 | 0.0005 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 728.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 742.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 753.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 765.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 781.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.020**  
> Non-inferior as long as a Dice loss up to 0.020 is acceptable.


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 21.0 | 1.0000 | not significant | 1.0000 | not significant |

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
| 0.030 | 151.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 176.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 204.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 233.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 259.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 32.0 | 1.0000 | not significant | 1.0000 | not significant |

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
| 0.025 | 156.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 181.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 214.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 251.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 283.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 322.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 100.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 105.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 114.5 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 149.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 203.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 258.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 313.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 358.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 409.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 456.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 491.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 530.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 562.0 | 0.8160 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 132.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 162.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 199.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 312.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 394.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 453.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 515.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 536.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 565.0 | 0.7370 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 587.0 | 0.3301 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 631.0 | 0.0478 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 647.0 | 0.0210 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 656.0 | 0.0128 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.040**  
> Non-inferior as long as a Dice loss up to 0.040 is acceptable.


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 103.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 111.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 118.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 147.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 198.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 253.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 297.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 348.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 398.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 437.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 483.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 515.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 541.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 74.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 80.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 88.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 120.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 195.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 267.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 339.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 400.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 463.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 531.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 591.0 | 0.2822 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 632.0 | 0.0455 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 658.0 | 0.0114 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.045**  
> Non-inferior as long as a Dice loss up to 0.045 is acceptable.


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 231.0 | 0.9927 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 0.9927). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 243.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 253.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 292.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 379.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 435.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 498.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 557.0 | 0.9633 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 606.0 | 0.1517 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 650.0 | 0.0178 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 676.0 | 0.0039 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 702.0 | 0.0007 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 718.0 | 0.0002 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.035**  
> Non-inferior as long as a Dice loss up to 0.035 is acceptable.


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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 56.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 57.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 63.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 84.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 139.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 182.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 220.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 255.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 294.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 334.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 357.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 413.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 451.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.050 |

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

| W | Raw p-value | Raw Result | Corrected p-value (bonferroni) | Corrected Result |
|---|-------------|------------|--------------------------------------|-------------------|
| 76.0 | 1.0000 | not significant | 1.0000 | not significant |

> Cannot claim superiority (p = 1.0000). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 78.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 84.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 111.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 171.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 223.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 269.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.020 |
| 0.025 | 339.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.025 |
| 0.030 | 404.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.030 |
| 0.035 | 459.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.035 |
| 0.040 | 504.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.040 |
| 0.045 | 549.0 | 1.0000 | inconclusive | Cannot confirm loss < 0.045 |
| 0.050 | 586.0 | 0.3432 | inconclusive | Cannot confirm loss < 0.050 |

> **Non-inferiority not confirmed up to δ = 0.050.**  
> Degradation too large/inconsistent within the tested tolerance range.


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
