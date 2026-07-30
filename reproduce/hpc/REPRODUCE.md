# Reproducibility Check — Instructions

**Note:** Do all steps in a non-interactive terminal session. Do not start an interactive job on the cluster first. Run every command directly on the login node, before any job starts.

**Goal:** Pick one row from `experiments.csv`. Re-run the same experiment. Check if your `mean_val_dice` and `std_val_dice` match the original row.

**Before you start:** Only use rows where `run_id` is after `2026_06_28_14_47_57`. Do not use rows before this run ID.

**Pick a dataset that fits in 30 minutes.** Use one of these three: `wbc`, `cellnuclei`, `isic2016`. Do not use any other dataset. Other datasets take longer than 30 minutes.

---

## Step 1: Clone the repo

1. Open a terminal.
2. Run this command:
   ```bash
   git clone https://github.com/bhavankumar2911/microunet.git
   ```
3. Move into the project folder:
   ```bash
   cd microunet
   ```

---

## Step 2: Set up the environment

1. Load Python:
   ```bash
   module load python
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step 3: Pick your row

1. Open `experiments/experiments.csv`.
2. Pick one row.
3. Check that the `run_id` of this row is after `2026_06_28_14_47_57`.
4. Check that the `dataset` of this row is `wbc`, `cellnuclei`, or `isic2016`.
5. Write down the `run_id`.
6. Write down `mean_val_dice`.
7. Write down `std_val_dice`.

---

## Step 4: Download the dataset

1. Look at the `dataset` value of the row you picked in Step 3.
2. Run only the **one** command below that matches your dataset.

For `wbc`:
```bash
mkdir -p data/Wbc && wget -O data/Wbc/wbc_256.npz "https://zenodo.org/records/13359660/files/wbc_256.npz?download=1"
```

For `cellnuclei`:
```bash
mkdir -p data/CellNuclei && wget -O data/CellNuclei/cellnuclei_256.npz "https://zenodo.org/records/13358372/files/cellnuclei_256.npz?download=1"
```

For `isic2016`:
```bash
mkdir -p data/Isic2016 && wget -O data/Isic2016/isic2016_256.npz "https://zenodo.org/records/13358372/files/isic2016_256.npz?download=1"
```

3. Wait until the download finishes before moving on.

---

## Step 5: Submit the job

1. The script `reproduce.sh` is already in the repo. Do not edit it. Do not create it.
2. Submit it with your `run_id` from Step 3. Put the `run_id` directly after the script name.
3. Use this command as a template:
   ```bash
   sbatch reproduce.sh <run_id>
   ```
4. Replace `<run_id>` with your own `run_id` from Step 3.

**Note:** Do not use `<` or `>` symbols anywhere in this command.

---

## Step 6: Wait for the job to finish

1. Check the queue:
   ```bash
   squeue
   ```
2. Look for your job in the list.
3. Wait until your job is no longer in the list.

---

## Step 7: Check your new result

1. Open `experiments/experiments.csv`.
2. Find the **last row** in the file.
3. Write down its `mean_val_dice`.
4. Write down its `std_val_dice`.

---

## Step 8: Compare

1. Compare your new `mean_val_dice` to the original `mean_val_dice` from Step 3.
2. Compare your new `std_val_dice` to the original `std_val_dice` from Step 3.
3. They should be equal.