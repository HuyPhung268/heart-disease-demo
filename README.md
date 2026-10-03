# Classroom demo — Heart disease classification

Teaching material for a machine learning course, built on `heart_disease.csv`
(10,000 rows x 21 columns). It ships a **Streamlit dashboard** for presenting and a
**Jupyter notebook** students can run themselves.

## Install

```bash
pip install -r requirements.txt              # to run the app
pip install -r requirements-notebook.txt     # plus extras, to run the notebook
```

## Run the Streamlit app

```bash
cd /Users/ts00018/DataScience/heart-disease
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Stop with `Ctrl + C`.

**Light / dark mode:** menu ☰ at the top right → *Settings* → *Appearance*. Both
palettes are declared in `.streamlit/config.toml`; the UI follows instantly, with no
reload.

`artifacts/` already contains trained models, so the app runs straight away. To
retrain from scratch:

```bash
python train.py    # 8 models + the control experiment   (~1 min)
python tune.py     # GridSearchCV over 4 models          (~2 min)
```

## Deploy to Streamlit Community Cloud

Streamlit Cloud only deploys from a **GitHub repo**, so the first step is pushing the
project there. This folder is already a git repo with a commit in place.

### 1. Create the repo on GitHub

Go to https://github.com/new, pick a name (for example `heart-disease-demo`), do
**not** tick "Add a README file", then click *Create repository*.

### 2. Push the code

```bash
cd /Users/ts00018/DataScience/heart-disease
git remote add origin https://github.com/<account>/heart-disease-demo.git
git push -u origin main
```

### 3. Deploy

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in: *Repository* = the repo you just created, *Branch* = `main`,
   *Main file path* = `app.py`.
4. Click **Deploy**. The first build takes 3-5 minutes to install dependencies.

The dependency set is tested on **both Python 3.13 and 3.14**, so there is no need to
pick a specific Python version under *Advanced settings*.

### Files Streamlit Cloud needs

| File | Purpose |
|---|---|
| `requirements.txt` | Python packages, **pinned to exact versions** |
| `packages.txt` | Linux system packages — `libgomp1` for LightGBM |
| `.streamlit/config.toml` | Light/dark theme |
| `artifacts/*.joblib` | Trained models, committed so the app runs immediately |

### A failure we hit: pyarrow will not build on Python 3.14

The first deploy failed with:

```
× Failed to download and build `pyarrow==21.0.0`
  error: command 'cmake' failed: No such file or directory
ERROR: Could not build wheels for pyarrow
```

The chain of causes:

1. Streamlit Cloud built the environment with **Python 3.14**.
2. `streamlit==1.51.0` pins `pyarrow<22`.
3. pyarrow only ships Python 3.14 wheels **from version 22 onward**.
4. No matching wheel → pip builds from source → no `cmake` → failure.

The fix: move up to `streamlit==1.65.0` (which allows `pyarrow<26`) and pin
`pyarrow==25.0.1`. The current pin set has been run on both 3.13 and 3.14.

One note if you customise the look: Streamlit ≤1.5x marks tabs with
`[data-baseweb="tab"]`, while ≥1.6x uses `[role="tab"]`. The active-tab indicator
differs too — newer versions draw it with `[role="tablist"]::after` and
`.react-aria-SelectionIndicator`. `src/theme.py` declares both groups of selectors so
the CSS survives an upgrade.

### Why versions are pinned exactly

`artifacts/*.joblib` are **pickles** of scikit-learn and LightGBM objects. If Cloud
installs versions different from the ones used for training, loading the models can
fail. That is why `requirements.txt` uses `==` rather than `>=`.

If you later upgrade libraries locally, remember to re-run `python train.py` and
`python tune.py`, then update `requirements.txt` to match.

### Classroom notes

- **The app sleeps when idle.** The first visit after a long gap takes about 30
  seconds to wake. Open it a few minutes before class.
- **A public repo means public data.** `heart_disease.csv` is synthetic, so privacy is
  not a concern here, but keep it in mind if you swap in real data. Streamlit Cloud
  does support private repos; check your account's limits under Settings.
- **To update the app**, just `git push` — Cloud redeploys automatically.

## Train and test sets

The project has a **single data source**: `Dataset/heart_disease.csv`. The train/test
split happens **in code** via `train_test_split(stratify=y)`, not through two
pre-supplied files the way Kaggle competitions usually work.

`train.py` is the *name of the training script*, not a data file. So that the split is
visible, the script also exports two reference files (which are not program inputs):

| File | Contents |
|---|---|
| `artifacts/train_set.csv` | 8,000 rows — what the models learn from |
| `artifacts/test_set.csv` | 2,000 rows — **never seen** by the models |

Both keep the 20.0% disease rate thanks to `stratify=y`.

## Layout

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app, 6 tabs |
| `notebooks/heart_disease_walkthrough.ipynb` | 57-cell walkthrough, 7 lessons, 5 exercises |
| `src/data.py` | Data loading (handles the `"None"` trap) + preprocessing pipeline |
| `src/models.py` | The 8 models under comparison |
| `src/theme.py` | Palette, CSS and the shared chart template |
| `.streamlit/config.toml` | Streamlit light/dark theme |
| `train.py` | Training, cross-validation, the control experiment |
| `tune.py` | Hyperparameter tuning with `GridSearchCV` |
| `requirements.txt` | App dependencies (the file Streamlit Cloud reads) |
| `requirements-notebook.txt` | Extra packages for the notebook |
| `packages.txt` | Linux system packages for Streamlit Cloud |
| `artifacts/` | Trained models, `metrics.csv`, `tuning.csv`, the train/test split |

## The six tabs

1. **Overview** — class imbalance, missing values, per-feature distributions
2. **Preprocessing** — the `"None"` trap, ordinal vs nominal, leakage prevention
3. **Model comparison** — metrics table, ROC, confusion matrix, feature importance
4. **Tuning** — `GridSearchCV`, gains measured against the noise band
5. **Data diagnostics** — three tests for signal in the data
6. **Predict** — an input form plus two extreme profiles side by side

The dashboard runs the full width of the screen and presents figures and charts only.
All the teaching commentary lives in the notebook, so the instructor controls the
narrative in class.

Every component drawn in `src/theme.py` is theme-independent: backgrounds use neutral
`rgba`, text uses `inherit`, chart backgrounds are transparent. That makes the
light/dark switch instant. (We deliberately avoid `st.context.theme`, which reports
the wrong value at the exact moment the user switches — see streamlit#11920.)

## Chart palette

`src/theme.py` uses a validated 8-slot categorical palette:

```
#3987e5  #d95926  #199e70  #c98500  #d55181  #008300  #9085e9  #e66767
```

All 8 slots pass **all five checks** (lightness band, chroma floor, colour-blind
separation, normal-vision floor, contrast ≥ 3:1) against **both** the light surface
`#FFFFFF` and the dark surface `#0B1120`. One palette therefore serves both modes with
no need to read the theme at runtime.

Two rules to keep if you edit it:

- **Assign colours in slot order, never cycle.** That order *is* the colour-blind
  safety mechanism, not a cosmetic choice.
- **Status colours are reserved** (`good` / `warning` / `serious` / `critical`) and are
  never reused as a data-series colour.

`primaryColor` in `.streamlit/config.toml` uses slot 1 so Streamlit's own widgets match
the charts.

## Training results

Every model lands at ROC-AUC ≈ 0.50. This is **not a bug in the code** — the dataset is
randomly generated synthetic data whose features are statistically independent of the
label:

- The best model's accuracy exactly matches the majority-class baseline (80%)
- Training on **shuffled labels** gives ROC-AUC 0.518 — *higher* than real labels (0.498)
- Every numeric feature correlates with the label within ±0.02
- Random Forest got **worse** on the test set after `GridSearchCV` (−0.0275)

The dataset is therefore well suited to teaching **process and critical thinking**, and
unsuited to drawing medical conclusions.

## A number that is easy to misread

The dataset is missing only **500 cells out of 200,000**, touching **500 rows (5.0%)**.

Read it with a default `pd.read_csv()` and the string `"None"` in `Alcohol Consumption`
becomes `NaN`, inflating that to **2,933 rows (29.3%)** — nearly six times too high. It
is a good teaching example: one data-loading mistake distorts the entire assessment of
data quality.

## Teaching suggestion

Run tabs 1→4 like an ordinary ML session and let students see the 80% number and find
it impressive. Tab 5, **Data diagnostics**, is where you turn the card over. That jolt
teaches more than saying it up front.

Tab 6, **Predict**, works well as the closing beat: the model hands out a confident
probability for any patient, and the two extreme profiles show it swinging from 0% to
100% — right after tab 5 established that it cannot actually predict anything.
