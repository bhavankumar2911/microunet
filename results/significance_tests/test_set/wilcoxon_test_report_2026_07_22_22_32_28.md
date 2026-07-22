# Wilcoxon Statistical Test Report

Generated: 2026-07-22 22:32:28

Datasets: 40
Significance threshold: α = 0.05
Non-inferiority δ range: 0.001 – 0.050

---

## Depth-wise separable layers (14-28-56-112) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 510.5 | 0.0916 | not significant |

> Cannot claim superiority (p = 0.0916). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 549.0 | 0.0311 | **NON-INFERIOR ✓** | Loss within 0.001 |
| 0.002 | 584.0 | 0.0093 | **NON-INFERIOR ✓** | Loss within 0.002 |
| 0.005 | 640.0 | 0.0008 | **NON-INFERIOR ✓** | Loss within 0.005 |
| 0.010 | 698.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 737.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 748.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 759.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 772.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 776.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.001**  
> Non-inferior as long as a Dice loss up to 0.001 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (14-28-56-112) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7237 |
| Mean Δ Dice | +0.0039 |
| Std Δ Dice | 0.0510 |
| Datasets improved | 21/40 |
| Datasets regressed | 19/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 51,162 |
| Parameter reduction | 44,755 (46.7%) |


</td>
</tr>
</table>

---

## Depth-wise separable convolution + additive skip connection (14-28-56-112) vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 434.5 | 0.3773 | not significant |

> Cannot claim superiority (p = 0.3773). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 454.5 | 0.2813 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 507.0 | 0.0984 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 600.0 | 0.0049 | **NON-INFERIOR ✓** | Loss within 0.005 |
| 0.010 | 701.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 740.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 748.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 754.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 766.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 791.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 810.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 816.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 817.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.005**  
> Non-inferior as long as a Dice loss up to 0.005 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable convolution + additive skip connection (14-28-56-112) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7210 |
| Mean Δ Dice | +0.0012 |
| Std Δ Dice | 0.0234 |
| Datasets improved | 18/40 |
| Datasets regressed | 22/40 |
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

| W | p-value | Result |
|---|---------|--------|
| 467.5 | 0.2259 | not significant |

> Cannot claim superiority (p = 0.2259). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 493.0 | 0.1353 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 516.0 | 0.0789 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 582.0 | 0.0100 | **NON-INFERIOR ✓** | Loss within 0.005 |
| 0.010 | 669.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 698.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 708.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 711.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 717.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 733.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 740.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 747.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 748.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.005**  
> Non-inferior as long as a Dice loss up to 0.005 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (14-28-56-112) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7191 |
| Mean Δ Dice | -0.0007 |
| Std Δ Dice | 0.0395 |
| Datasets improved | 24/40 |
| Datasets regressed | 16/40 |
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

| W | p-value | Result |
|---|---------|--------|
| 385.0 | 0.5278 | not significant |

> Cannot claim superiority (p = 0.5278). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 422.0 | 0.4393 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 450.0 | 0.2996 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 552.0 | 0.0283 | **NON-INFERIOR ✓** | Loss within 0.005 |
| 0.010 | 631.0 | 0.0012 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 674.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 687.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 722.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 744.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 763.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 776.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 777.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 786.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.005**  
> Non-inferior as long as a Dice loss up to 0.005 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable convolution (16, 32, 64, 128) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7187 |
| Mean Δ Dice | -0.0011 |
| Std Δ Dice | 0.0316 |
| Datasets improved | 21/40 |
| Datasets regressed | 18/40 |
| Datasets unchanged | 1/40 |
| Parameters before | 95,917 |
| Parameters after | 65,895 |
| Parameter reduction | 30,022 (31.3%) |


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
| 387.5 | 0.6227 | not significant |

> Cannot claim superiority (p = 0.6227). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 403.0 | 0.5397 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 426.0 | 0.4184 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 497.0 | 0.1240 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 580.0 | 0.0108 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 636.0 | 0.0009 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 701.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 726.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 755.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 786.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 802.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 808.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 814.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

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
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7198 |
| Mean Δ Dice | +0.0000 |
| Std Δ Dice | 0.0296 |
| Datasets improved | 18/40 |
| Datasets regressed | 22/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 38,286 |
| Parameter reduction | 57,631 (60.1%) |


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

| W | p-value | Result |
|---|---------|--------|
| 272.0 | 0.9689 | not significant |

> Cannot claim superiority (p = 0.9689). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 291.5 | 0.9456 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 320.0 | 0.8867 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 427.0 | 0.4132 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 539.0 | 0.0421 | **NON-INFERIOR ✓** | Loss within 0.010 |
| 0.015 | 605.0 | 0.0040 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 647.0 | 0.0005 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 681.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 707.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 733.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 748.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 771.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 785.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.010**  
> Non-inferior as long as a Dice loss up to 0.010 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (12-24-48-96) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7137 |
| Mean Δ Dice | -0.0061 |
| Std Δ Dice | 0.0322 |
| Datasets improved | 12/40 |
| Datasets regressed | 28/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 42,366 |
| Parameter reduction | 53,551 (55.8%) |


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
| 166.0 | 0.9997 | not significant |

> Cannot claim superiority (p = 0.9997). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 186.0 | 0.9990 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 215.0 | 0.9962 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 348.0 | 0.7975 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 522.0 | 0.0675 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 553.0 | 0.0274 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 599.0 | 0.0051 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 646.0 | 0.0006 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 719.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 731.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 758.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 768.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 787.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.015**  
> Non-inferior as long as a Dice loss up to 0.015 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (10-20-40-80) |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7092 |
| Mean Δ Dice | -0.0105 |
| Std Δ Dice | 0.0267 |
| Datasets improved | 8/40 |
| Datasets regressed | 32/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 27,265 |
| Parameter reduction | 68,652 (71.6%) |


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

| W | p-value | Result |
|---|---------|--------|
| 255.0 | 0.9822 | not significant |

> Cannot claim superiority (p = 0.9822). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 279.5 | 0.9615 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 296.0 | 0.9376 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 377.0 | 0.6721 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 495.0 | 0.1296 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 545.0 | 0.0352 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 579.0 | 0.0112 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 625.0 | 0.0016 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 673.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 698.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 712.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 750.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 773.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.015**  
> Non-inferior as long as a Dice loss up to 0.015 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (10-20-40-80) + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7087 |
| Mean Δ Dice | -0.0111 |
| Std Δ Dice | 0.0307 |
| Datasets improved | 11/40 |
| Datasets regressed | 29/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 30,673 |
| Parameter reduction | 65,244 (68.0%) |


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

| W | p-value | Result |
|---|---------|--------|
| 382.0 | 0.6477 | not significant |

> Cannot claim superiority (p = 0.6477). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 392.0 | 0.5971 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 400.0 | 0.5555 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 442.0 | 0.3376 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 507.0 | 0.0984 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 567.0 | 0.0172 | **NON-INFERIOR ✓** | Loss within 0.015 |
| 0.020 | 608.0 | 0.0035 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 649.0 | 0.0005 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 670.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 701.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 721.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 739.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 749.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.015**  
> Non-inferior as long as a Dice loss up to 0.015 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (16-32-64) + triple convolution per block + attention gate |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7209 |
| Mean Δ Dice | +0.0011 |
| Std Δ Dice | 0.0550 |
| Datasets improved | 17/40 |
| Datasets regressed | 23/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 27,815 |
| Parameter reduction | 68,102 (71.0%) |


</td>
</tr>
</table>

---

## Depth-wise separable layers (16-32-64) + triple convolution per block vs. Stable Baseline

<table>
<tr>
<td valign="top" width="60%">

### Superiority Test

*One-sided Wilcoxon signed-rank test (alternative: comparison > baseline)*

| W | p-value | Result |
|---|---------|--------|
| 302.0 | 0.9270 | not significant |

> Cannot claim superiority (p = 0.9270). Proceeding to non-inferiority test.

### Non-Inferiority Test

*One-sided Wilcoxon signed-rank test on adjusted Δ = (comparison − baseline) + δ (alternative: adjusted Δ > 0). Significant result means comparison is non-inferior within tolerance δ.*

| δ | W | p-value | Result | Interpretation |
|---|---|---------|--------|----------------|
| 0.001 | 314.0 | 0.9016 | inconclusive | Cannot confirm loss < 0.001 |
| 0.002 | 325.0 | 0.8733 | inconclusive | Cannot confirm loss < 0.002 |
| 0.005 | 371.0 | 0.7004 | inconclusive | Cannot confirm loss < 0.005 |
| 0.010 | 447.0 | 0.3136 | inconclusive | Cannot confirm loss < 0.010 |
| 0.015 | 517.0 | 0.0769 | inconclusive | Cannot confirm loss < 0.015 |
| 0.020 | 572.0 | 0.0144 | **NON-INFERIOR ✓** | Loss within 0.020 |
| 0.025 | 619.0 | 0.0021 | **NON-INFERIOR ✓** | Loss within 0.025 |
| 0.030 | 651.0 | 0.0004 | **NON-INFERIOR ✓** | Loss within 0.030 |
| 0.035 | 670.0 | 0.0001 | **NON-INFERIOR ✓** | Loss within 0.035 |
| 0.040 | 687.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.040 |
| 0.045 | 705.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.045 |
| 0.050 | 718.0 | 0.0000 | **NON-INFERIOR ✓** | Loss within 0.050 |

> **Smallest δ confirmed: 0.020**  
> Non-inferior as long as a Dice loss up to 0.020 is acceptable.


</td>
<td valign="top" width="40%">

### Summary

| Metric | Value |
|--------|-------|
| Baseline | Stable Baseline |
| Comparison | Depth-wise separable layers (16-32-64) + triple convolution per block |
| Datasets matched | 40 |
| Mean baseline Dice | 0.7197 |
| Mean comparison Dice | 0.7093 |
| Mean Δ Dice | -0.0104 |
| Std Δ Dice | 0.0435 |
| Datasets improved | 15/40 |
| Datasets regressed | 25/40 |
| Datasets unchanged | 0/40 |
| Parameters before | 95,917 |
| Parameters after | 25,479 |
| Parameter reduction | 70,438 (73.4%) |


</td>
</tr>
</table>

---
