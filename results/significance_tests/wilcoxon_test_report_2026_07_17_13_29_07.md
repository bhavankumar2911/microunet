# Wilcoxon Statistical Test Report

Generated: 2026-07-17 13:29:07

Datasets: 40
Significance threshold: α = 0.05
Non-inferiority δ range: 0.001 – 0.050

---

## Depth-wise separable layers (8-16-32-52) vs. Stable Baseline

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
| 0.050 | 550.0 | 0.0301 | **NON-INFERIOR ✓** | Dice loss within 0.050 tolerance |

> **Smallest δ at which non-inferiority is confirmed: 0.050**  
> Comparison is statistically non-inferior to baseline as long as a Dice loss of up to 0.050 is considered acceptable.

---

## Depth-wise separable layers (10-20-40-80) vs. Stable Baseline

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
| 0.020 | 606.0 | 0.0038 | **NON-INFERIOR ✓** | Dice loss within 0.020 tolerance |
| 0.025 | 649.0 | 0.0005 | **NON-INFERIOR ✓** | Dice loss within 0.025 tolerance |
| 0.030 | 674.0 | 0.0001 | **NON-INFERIOR ✓** | Dice loss within 0.030 tolerance |
| 0.035 | 722.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.035 tolerance |
| 0.040 | 738.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.040 tolerance |
| 0.045 | 763.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.045 tolerance |
| 0.050 | 773.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.050 tolerance |

> **Smallest δ at which non-inferiority is confirmed: 0.020**  
> Comparison is statistically non-inferior to baseline as long as a Dice loss of up to 0.020 is considered acceptable.

---

## Depth-wise separable layers (12-24-48-96) vs. Stable Baseline

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
| 0.010 | 536.0 | 0.0459 | **NON-INFERIOR ✓** | Dice loss within 0.010 tolerance |
| 0.015 | 589.0 | 0.0076 | **NON-INFERIOR ✓** | Dice loss within 0.015 tolerance |
| 0.020 | 646.0 | 0.0006 | **NON-INFERIOR ✓** | Dice loss within 0.020 tolerance |
| 0.025 | 695.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.025 tolerance |
| 0.030 | 741.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.030 tolerance |
| 0.035 | 767.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.035 tolerance |
| 0.040 | 787.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.040 tolerance |
| 0.045 | 801.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.045 tolerance |
| 0.050 | 808.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.050 tolerance |

> **Smallest δ at which non-inferiority is confirmed: 0.010**  
> Comparison is statistically non-inferior to baseline as long as a Dice loss of up to 0.010 is considered acceptable.

---

## Depth-wise separable layers (14-28-56-112) vs. Stable Baseline

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
| 0.005 | 596.0 | 0.0058 | **NON-INFERIOR ✓** | Dice loss within 0.005 tolerance |
| 0.010 | 673.0 | 0.0001 | **NON-INFERIOR ✓** | Dice loss within 0.010 tolerance |
| 0.015 | 727.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.015 tolerance |
| 0.020 | 746.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.020 tolerance |
| 0.025 | 752.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.025 tolerance |
| 0.030 | 760.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.030 tolerance |
| 0.035 | 772.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.035 tolerance |
| 0.040 | 777.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.040 tolerance |
| 0.045 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.045 tolerance |
| 0.050 | 779.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.050 tolerance |

> **Smallest δ at which non-inferiority is confirmed: 0.005**  
> Comparison is statistically non-inferior to baseline as long as a Dice loss of up to 0.005 is considered acceptable.

---

## Depth-wise separable convolution (16, 32, 64, 128) vs. Stable Baseline

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
| 0.010 | 574.0 | 0.0134 | **NON-INFERIOR ✓** | Dice loss within 0.010 tolerance |
| 0.015 | 616.0 | 0.0024 | **NON-INFERIOR ✓** | Dice loss within 0.015 tolerance |
| 0.020 | 669.0 | 0.0001 | **NON-INFERIOR ✓** | Dice loss within 0.020 tolerance |
| 0.025 | 706.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.025 tolerance |
| 0.030 | 728.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.030 tolerance |
| 0.035 | 742.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.035 tolerance |
| 0.040 | 753.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.040 tolerance |
| 0.045 | 765.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.045 tolerance |
| 0.050 | 781.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.050 tolerance |

> **Smallest δ at which non-inferiority is confirmed: 0.010**  
> Comparison is statistically non-inferior to baseline as long as a Dice loss of up to 0.010 is considered acceptable.

---

## Additive skip connection vs. Stable Baseline

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
| 0.025 | 536.0 | 0.0459 | **NON-INFERIOR ✓** | Dice loss within 0.025 tolerance |
| 0.030 | 565.0 | 0.0184 | **NON-INFERIOR ✓** | Dice loss within 0.030 tolerance |
| 0.035 | 587.0 | 0.0083 | **NON-INFERIOR ✓** | Dice loss within 0.035 tolerance |
| 0.040 | 631.0 | 0.0012 | **NON-INFERIOR ✓** | Dice loss within 0.040 tolerance |
| 0.045 | 647.0 | 0.0005 | **NON-INFERIOR ✓** | Dice loss within 0.045 tolerance |
| 0.050 | 656.0 | 0.0003 | **NON-INFERIOR ✓** | Dice loss within 0.050 tolerance |

> **Smallest δ at which non-inferiority is confirmed: 0.025**  
> Comparison is statistically non-inferior to baseline as long as a Dice loss of up to 0.025 is considered acceptable.

---

## Single convolution per block (8-16-32-52) vs. Stable Baseline

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
| 0.050 | 541.0 | 0.0397 | **NON-INFERIOR ✓** | Dice loss within 0.050 tolerance |

> **Smallest δ at which non-inferiority is confirmed: 0.050**  
> Comparison is statistically non-inferior to baseline as long as a Dice loss of up to 0.050 is considered acceptable.

---

## Single convolution per block (10-20-40-80) vs. Stable Baseline

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
| 0.025 | 557.0 | 0.0241 | **NON-INFERIOR ✓** | Dice loss within 0.025 tolerance |
| 0.030 | 606.0 | 0.0038 | **NON-INFERIOR ✓** | Dice loss within 0.030 tolerance |
| 0.035 | 650.0 | 0.0004 | **NON-INFERIOR ✓** | Dice loss within 0.035 tolerance |
| 0.040 | 676.0 | 0.0001 | **NON-INFERIOR ✓** | Dice loss within 0.040 tolerance |
| 0.045 | 702.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.045 tolerance |
| 0.050 | 718.0 | 0.0000 | **NON-INFERIOR ✓** | Dice loss within 0.050 tolerance |

> **Smallest δ at which non-inferiority is confirmed: 0.025**  
> Comparison is statistically non-inferior to baseline as long as a Dice loss of up to 0.025 is considered acceptable.

---
