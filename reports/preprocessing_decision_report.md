# CICMalDroid 2020 — Preprocessing Decision Report

**Generated:** 2026-08-25
**Project:** Privacy-Preserving Malware Detection Using Federated Learning
**Dataset:** `feature_vectors_syscallsbinders_frequency_5_Cat.csv` (PRIMARY)
**Status:** Research-review pass — NO dataset modifications made

> **Constraint respected:** Raw dataset files have not been modified, deleted, or overwritten.

---

## 1. ZERO-FEATURE ROW INVESTIGATION

### 1.1 Complete Listing of All 35 Zero-Feature Rows

All 35 rows have **all 470 features equal to exactly zero**. All are feature-only duplicates
(they share the same all-zero feature vector). 34 of 35 are also exact duplicates (same class
as another zero-row). Row 6184 is the only SMS-labeled zero-row, so it is a feature-only
duplicate but NOT an exact duplicate.

| Row | Class | Category | Exact Dup | Feature Dup |
|-----|-------|----------|-----------|-------------|
| 1257 | 2 | Banking | ✅ | ✅ |
| 1442 | 2 | Banking | ✅ | ✅ |
| 1449 | 2 | Banking | ✅ | ✅ |
| 1568 | 2 | Banking | ✅ | ✅ |
| 1674 | 2 | Banking | ✅ | ✅ |
| 1751 | 2 | Banking | ✅ | ✅ |
| 2190 | 2 | Banking | ✅ | ✅ |
| 2219 | 2 | Banking | ✅ | ✅ |
| 2228 | 2 | Banking | ✅ | ✅ |
| 2358 | 2 | Banking | ✅ | ✅ |
| 2679 | 2 | Banking | ✅ | ✅ |
| 2716 | 2 | Banking | ✅ | ✅ |
| 2788 | 2 | Banking | ✅ | ✅ |
| 2802 | 2 | Banking | ✅ | ✅ |
| 2826 | 2 | Banking | ✅ | ✅ |
| 2923 | 2 | Banking | ✅ | ✅ |
| 2944 | 2 | Banking | ✅ | ✅ |
| 3046 | 2 | Banking | ✅ | ✅ |
| 3232 | 2 | Banking | ✅ | ✅ |
| 3300 | 2 | Banking | ✅ | ✅ |
| 6184 | 3 | SMS | ❌ | ✅ |
| 7708 | 4 | Riskware | ✅ | ✅ |
| 8297 | 4 | Riskware | ✅ | ✅ |
| 8764 | 4 | Riskware | ✅ | ✅ |
| 8771 | 4 | Riskware | ✅ | ✅ |
| 8782 | 4 | Riskware | ✅ | ✅ |
| 9154 | 4 | Riskware | ✅ | ✅ |
| 9293 | 4 | Riskware | ✅ | ✅ |
| 9334 | 4 | Riskware | ✅ | ✅ |
| 9699 | 4 | Riskware | ✅ | ✅ |
| 9719 | 4 | Riskware | ✅ | ✅ |
| 10253 | 5 | Benign | ✅ | ✅ |
| 11112 | 5 | Benign | ✅ | ✅ |
| 11353 | 5 | Benign | ✅ | ✅ |
| 11557 | 5 | Benign | ✅ | ✅ |

**Class breakdown:** Banking (20), Riskware (10), Benign (4), SMS (1)

### 1.2 Analysis

**Are these legitimate applications with no observed behavior?**

No. The following evidence rules this out:

1. **Multiple classes share the same all-zero vector.** If zero features meant "app does nothing",
   we would expect only Benign apps to produce this. Instead, 4 of 5 classes are represented —
   including Banking and Riskware malware. A banking trojan cannot be behaviorally indistinguishable
   from a benign app AND from riskware simultaneously. These are clearly different APKs whose
   dynamic traces were not captured.

2. **The zero vector is a sentinel, not behavioral data.** In Android dynamic analysis sandboxes
   (e.g., DroidBox, CuckooDroid), APKs that fail to install, crash at startup, detect the sandbox
   environment, or exceed an execution time limit produce zero-count feature records. The dataset
   contains these failed-execution records as rows with their original APK labels intact.

3. **20 of the 35 are Banking-labeled zeros.** Banking malware families are known for heavy
   sandbox evasion — they actively check for emulator/AV environment signals and refuse to
   execute if detected. This concentration of Banking zeros is consistent with this behavior.

4. **No classifier can learn from a zero vector.** Any model trained on these rows will assign
   a class prediction based entirely on the prior (majority class in training data), not on
   any feature. Including them inflates the class 2 count without contributing learning signal.

**Are they invalid/ambiguous records?**

Yes, conclusively. They are invalid because:
- The feature vector carries no behavioral information.
- The label cannot be verified via features — the class label exists only because the APK file
  had a known family label before analysis, but the analysis produced no data.
- Including zero-vector rows in the training set rewards classifiers for predicting the majority
  class on empty inputs, not for detecting malware behavior.

### 1.3 Recommendation

> ⛔ **DECISION: DROP all 35 zero-feature rows.**

**Evidence-based rationale:**
- Zero vectors provide no discriminative signal and will actively harm model performance.
- They span 4 classes with identical feature representations — impossible to classify correctly.
- Including them in a federated learning context would mean some clients receive "poisoned" data
  (empty samples with arbitrary labels), degrading local model quality disproportionately.
- **Impact:** Removes 35 of 11,598 rows (0.30%) — negligible data loss, large quality gain.

**No ambiguity — this is a clear DROP.**

---

## 2. LABEL-CONFLICT DUPLICATE INVESTIGATION

### 2.1 Conflict Group 1 — Zero-Feature Rows (35 rows)

Covered in Section 1 above. All 35 zero-feature rows form a single feature-duplicate group.
Since they have different class labels, they are also label-conflict duplicates.

**The 3 label-conflict rows from this group** (rows whose class differs from the Banking majority):
- Row 6184 (SMS)
- Row 7708 (Riskware — first Riskware zero-row, so counted as "extra")
- Row 10253 (Benign — first Benign zero-row)

**Decision:** DROP (subsumed by Section 1 decision — all 35 zero-feature rows are dropped).

---

### 2.2 Conflict Group 2 — Adware vs Riskware (rows 5 and 7261)

| Attribute | Row 5 | Row 7261 |
|-----------|-------|---------|
| Assigned class | 1 (Adware) | 4 (Riskware) |
| Non-zero features | **64** | **64** |
| Feature vector | Identical to Row 7261 | Identical to Row 5 |

**Analysis:**

These two samples have **identical 64-feature behavioral profiles** but conflicting labels.
64 non-zero syscall/binder frequency counts is a substantial behavioral fingerprint. This is
NOT a trivially sparse sample — it represents a real execution trace.

Three hypotheses:

1. **Same APK, labeled differently** — The same APK file was obtained from multiple sources
   (e.g., Google Play and third-party markets) and labeled by different analysts or tools.
   Adware and Riskware are notoriously difficult to distinguish: both monetize user data,
   both may display ads, and the boundary depends on researcher judgment.

2. **Different APKs with the same feature vector** — Two distinct APKs that happen to make
   the same 64 system calls in the same frequencies during the analysis window. At 64 non-zero
   features out of 470, this collision probability is extremely low, making hypothesis 1 more
   likely.

3. **Data pipeline error** — The same record was entered twice with different labels due to
   a labeling error in the dataset construction pipeline.

**Which hypotheses can we rule out?**

We cannot access the original APK files (not included in this dataset), so we cannot compare
SHA-256 hashes to distinguish hypotheses 1 and 3 from hypothesis 2. However, the Adware vs
Riskware boundary ambiguity is well-documented in Android malware research (see Yerima et al.,
2019; Mariconti et al., 2016). The ambiguity is genuine regardless of hypothesis.

**Decision options:**
- **KEEP one** (by majority, keep Adware since it's the earlier row): Introduces a label that
  is ambiguous even to domain experts. Risks training the model on an unreliable ground truth.
- **DROP both**: Conservative. Loses 2 samples but removes ambiguity.
- **Label as Riskware** (or Adware): Arbitrary without additional evidence.

### 2.3 Conflict Group 3 — Adware vs Riskware (rows 776 and 8784)

| Attribute | Row 776 | Row 8784 |
|-----------|---------|---------|
| Assigned class | 1 (Adware) | 4 (Riskware) |
| Non-zero features | **105** | **105** |
| Feature vector | Identical to Row 8784 | Identical to Row 776 |

**Analysis:** Same situation as Group 2, but with a richer 105-feature behavioral profile.
With 105 matching non-zero feature counts, this is almost certainly the same APK analyzed
twice with different family labels assigned. The probability of two distinct APKs producing
identical syscall/binder frequency profiles across 105 non-zero features is astronomically low.

**This is almost certainly the same APK labeled as both Adware and Riskware.**

### 2.4 Recommendations for Label-Conflict Groups 2 and 3

> ⚠️ **DECISION: DROP all 4 rows (rows 5, 776, 7261, 8784) — both members of each conflicting pair.**

**Rationale:**
- Keeping one member from each pair requires choosing a label without valid evidence.
- Neither Adware nor Riskware is demonstrably "more correct" for these specific samples.
- The Adware/Riskware boundary is a known ambiguity in Android malware taxonomy.
- 4 rows is 0.034% of the dataset — the loss is negligible.
- In federated learning, if one copy lands in a client's training set and the other in the
  global test set, it creates an artificial "memorization" advantage — another reason to
  remove both.

**UNRESOLVED QUESTION (requiring your input):**

> If you have access to the original APK SHA-256 hashes (from the CICMalDroid metadata), we
> could verify whether rows 5/7261 and 776/8784 are the same APKs. If they ARE the same APK,
> keeping one with either label is arbitrary. If they are DIFFERENT APKs, the Adware label
> (earlier row, and statistically Adware is the more commonly assigned primary category) may
> be preferable. Without hashes, DROP BOTH is the safest decision.

---

## 3. PERFECTLY CORRELATED FEATURE PAIRS (r = 1.0)

### 3.1 Complete List of All 35 Pairs

The pairs form cliques (groups of mutually correlated features). The table below shows all 35
pairs, organized by clique with non-zero sample count as a measure of overall prevalence.

#### Clique A — Thread creation aliases (2 features, 11,559 non-zeros each)

| Keep | Drop | Reason |
|------|------|--------|
| `clone` | `CREATE_THREAD_____` | `clone` is the actual Linux syscall name; the upper-case binder abstraction is a redundant wrapper label |

#### Clique B — Process execution aliases (2 features, 2,161 non-zeros each)

| Keep | Drop | Reason |
|------|------|--------|
| `execve` | `EXECUTE_____` | Same as above — `execve` is the actual syscall |

#### Clique C — Process creation near-aliases (2 features, 2,136 / 2,135 non-zeros)

| Keep | Drop | r | Reason |
|------|------|---|--------|
| `fork` | `CREATE_PROCESS\`_____` | 0.999997 | Near-perfect (r≈1.0); `fork` is the actual syscall |

#### Clique D — Thread termination near-aliases (2 features, 9,636 each)

| Keep | Drop | r | Reason |
|------|------|---|--------|
| `exit` | `TERMINATE_THREAD` | 0.999993 | `exit` is the actual syscall |

#### Clique E — Privilege/capability setting (2 features, 11,400 each)

| Keep | Drop | Reason |
|------|------|--------|
| `capset` | `setgroups32` | `capset` is broader (sets process capability masks); both always co-occur. Keep the more semantically specific one. Either is valid — recommend `capset` as it is used in root-privilege escalation detection literature. |

#### Clique F — UID/GID privilege group (4 features, 1 non-zero each)

All four features (`chown32`, `getresgid32`, `getresuid32`, `mount`) always appear together
(1 sample). Forming a 4-clique:

| Keep | Drop (3) | Reason |
|------|----------|--------|
| `mount` | `chown32`, `getresgid32`, `getresuid32` | `mount` is the most malware-relevant of the four (mounting filesystems is a core rootkit action). Keep 1, drop 3. |

#### Clique G — Backup restore group (3 features, 2 non-zeros each)

| Keep | Drop (2) | Reason |
|------|----------|--------|
| `beginRestoreSession` | `endRestoreSession`, `restorePackage` | All three always appear together. Keep first alphabetically to represent the "backup abuse" behavior. |

#### Clique H — Sync operations (3 features, 1 non-zero each)

| Keep | Drop (2) | Reason |
|------|----------|--------|
| `cancelSync` | `isSyncActive`, `removeStatusChangeListener` | All three always appear in the same 1 sample. Keep `cancelSync` as the most operationally meaningful. |

#### Clique I — Timer operations (3 features, 1 non-zero each)

| Keep | Drop (2) | Reason |
|------|----------|--------|
| `timer_create` | `fstatfs64`, `timer_settime` | `timer_create` is most descriptive. |

#### Clique J — Bluetooth state (3 features, 29 non-zeros each)

| Keep | Drop (2) | Reason |
|------|----------|--------|
| `getState` | `isBluetoothA2dpOn`, `startWatchingRoutes` | `getState` is a generic state query; `isBluetoothA2dpOn` is more specific. Both equally valid. Recommend `isBluetoothA2dpOn` for specificity. |

**Revised:** Keep `isBluetoothA2dpOn`, Drop `getState`, `startWatchingRoutes`.

#### Clique K — Media session metadata (3 features, 1 non-zero each)

| Keep | Drop (2) | Reason |
|------|----------|--------|
| `setMetadata` | `setPlaybackState`, `setTransportControlInfo` | All three are always together. |

#### Individual pairs (no clique expansion):

| Keep | Drop | r | Non-zeros | Reason |
|------|------|---|-----------|--------|
| `sched_getparam` | `sched_getscheduler` | 1.0 | 354 | Scheduling params always accompany scheduler query; keep the parameter-level call |
| `ALTER_PHONE_STATE___` | `updateServiceLocation` | 1.0 | 9 | Binder abstraction for the same operation |
| `setgid32` | `setuid32` | 1.0 | 2 | Both are UID/GID setting calls; always co-occur. Keep `setgid32`. |
| `getFlashlightEnabled` | `setFlashlightEnabled` | 1.0 | 2 | Always together. Keep the getter. |
| `getBoolean` | `getLong` | 1.0 | 2 | SharedPreferences access; always together. Keep `getBoolean`. |
| `sched_get_priority_max` | `sched_get_priority_min` | 1.0 | 9 | Always appear together. Keep `sched_get_priority_max`. |
| `getWifiApConfiguration` | `setWifiApConfiguration` | 1.0 | 1 | Always together. Keep getter. |
| `getAppWidgetInfo` | `hasNamedWallpaper` | 1.0 | 1 | Always together. Keep `getAppWidgetInfo`. |
| `getIsSyncable` | `isPackageAvailable` | 1.0 | 1 | Always together. Keep `getIsSyncable`. |

### 3.2 Proposed Survival Rule

**Rule 1 (Syscall alias cliques):** When a binder-level abstraction name (ALL_CAPS pattern with
underscores, e.g., `CREATE_THREAD_____`) is perfectly correlated with its underlying Linux syscall
name (e.g., `clone`), **keep the syscall name and drop the abstraction**. Reasoning: the syscall
name is universally documented, more general, and used consistently across Android versions.

**Rule 2 (Co-occurrence cliques):** When multiple features always appear together (r=1.0 because
they co-occur in the same 1–30 samples), **keep the most semantically informative member** and
drop the rest. The kept feature name should best describe the behavior the group encodes.

**Rule 3 (Near-perfect pairs, r≈1.0):** Apply Rule 1 (syscall name wins).

### 3.3 Verification: Does Dropping Remove Unique Information?

No unique information is lost because:
1. For r = 1.0, Feature B = k × Feature A (perfect linear relationship, k > 0). Any model
   that could learn from Feature B can learn identically from Feature A — they encode the same
   variance.
2. For r ≈ 1.0 (cliques C and D), the tiny deviation is due to 1 sample difference in non-zero
   count (2,136 vs 2,135). This is a likely data collection artifact (hash collision or single
   sample where the low-level call was recorded slightly differently). Dropping does not remove
   meaningful unique signal.
3. PCA or downstream regularization would effectively zero out one of two perfectly correlated
   features anyway — explicit removal is cleaner and more interpretable.

### 3.4 Summary of Feature Reduction

Counting clique members to drop per clique:

| Clique | Features | Keep | Drop |
|--------|----------|------|------|
| A (thread) | 2 | 1 | 1 |
| B (exec) | 2 | 1 | 1 |
| C (fork, near) | 2 | 1 | 1 |
| D (exit, near) | 2 | 1 | 1 |
| E (capset) | 2 | 1 | 1 |
| F (UID/GID) | 4 | 1 | 3 |
| G (restore) | 3 | 1 | 2 |
| H (sync) | 3 | 1 | 2 |
| I (timer) | 3 | 1 | 2 |
| J (bluetooth) | 3 | 1 | 2 |
| K (media) | 3 | 1 | 2 |
| Individual pairs (9 pairs) | 18 | 9 | 9 |
| **Total** | **47** | **20** | **27** |

> **Net reduction: 470 → 443 features** (drop 27 features from the 35-pair set,
> because 12 pairs form multi-member cliques where dropping 1 per clique removes fewer
> than 1-per-pair).

---

## 4. HIGH-SPARSITY FEATURES ANALYSIS

### 4.1 Summary Statistics

| Metric | Value |
|--------|-------|
| Features with >99% zeros | **247** of 470 (52.6%) |
| Features with 100% zeros | 0 |
| Minimum sparsity in group | 99.01% (feature is non-zero in ~115 rows) |
| Maximum sparsity | 99.99% (non-zero in ~1 row) |
| Mean sparsity | 99.82% (non-zero in ~21 rows on average) |

### 4.2 Class-Discriminative Potential

Analysis of the 247 high-sparsity features reveals significant variation in discriminative value:

**Highly discriminative (non-zeros in ≤2 classes):** 89 of 247 features

Examples with strong discrimination:
| Feature | NZ Count | Class Concentration |
|---------|----------|---------------------|
| `getUserIcon` | 100 | Banking: 40, SMS: 60 |
| `getUsers` | 100 | Banking: 40, SMS: 60 |
| `setIconVisibility` | 100 | Banking: 40, SMS: 60 |
| `statusBarVisibilityChanged` | 100 | Banking: 40, SMS: 60 |
| `getDeviceOwner` | 99 | Banking: 39, SMS: 60 |
| `setRingerMode` | 108 | Banking: 70, SMS: 5, Riskware: 31 |
| `onGetSentenceSuggestionsMultiple` | 92 | SMS: 75, Adware: 5 |
| `getMobileDataEnabled` | 108 | Riskware: 91 |
| `getSearchableInfo` | 87 | Riskware: 26, Benign: 60 |
| `getpgid` | 114 | Riskware: 80, Benign: 15 |
| `isCameraSoundForced` | 91 | Benign: 58, Adware: 10 |

**Notable pattern:** Four features (`getUserIcon`, `getUsers`, `setIconVisibility`,
`statusBarVisibilityChanged`) always appear together (all Banking/SMS only) with identical
counts — these are likely also perfectly correlated. They may represent a device-owner or
multi-user API abuse pattern specific to Banking and SMS malware. Even though they appear
in <1% of samples, their class specificity is high.

**Broadly spread (non-zeros in all 5 classes):** 17 of 247 — lower discriminative power
per sample, but still contribute variance.

### 4.3 Options Analysis

**Option A — Retain all 247 high-sparsity features:**
- Pros: Preserves rare but highly class-specific signals (e.g., features appearing in only
  Banking samples). Some tree-based models (Random Forest, XGBoost) handle sparse features well.
- Cons: 247 extra features where >99% of rows are zero. For gradient-based methods (neural
  networks / federated learning with FedAvg), extremely sparse features can dominate gradient
  updates for the rare samples containing them, causing instability.
- For FL specifically: rare features create non-IID pressure even in IID partition scenarios —
  a client that happens to have none of the rare-feature samples will learn a different local
  model than one with several.

**Option B — Remove all 247 high-sparsity features:**
- Pros: Cleaner, faster training. Reduces the feature space from 470 to 223 (after also
  removing perfect-correlation drops).
- Cons: Loses 89 potentially discriminative features (the ≤2-class concentrated ones).
  `getUserIcon`/`getUsers` etc. are genuinely useful Banking/SMS discriminators despite sparsity.
- Risk: May measurably reduce classification accuracy, especially for Banking vs SMS distinction.

**Option C — Compare both using validation performance:**
- Train two model variants:
  - Model F (Full): 470 features → ~443 after correlation drops
  - Model R (Reduced): 223 features (drop high-sparsity + correlation drops)
- Compare validation F1-macro and per-class recall on the same validation set.
- Select the better approach before training the final model.
- **Computationally feasible:** Both variants run on the same 11K-row dataset in minutes.

### 4.4 Recommendation

> ⚠️ **DECISION: Option C — compare both approaches, with strong expectation that retaining
> high-sparsity features or using a threshold-based subset will outperform full removal.**

**Specific guidance:**
- Do NOT remove all 247 at once — the 89 class-concentrated features are too valuable.
- Consider a **tiered threshold**: keep features with ≥ 5 non-zero samples (removes the
  least informative tail while retaining the discriminative ones).
- Apply threshold: Features with <5 non-zero rows across the entire dataset would be dropped.
  Inspect how many this affects.
- This is the **primary unresolved decision** — awaiting your approval to run the comparison.

---

## 5. PREPROCESSING ORDER (Leakage-Safe)

The following order prevents any form of data leakage where test-set information influences
training-set transformations.

```
Step 1: QUALITY FILTERING (whole-dataset, no fit involved)
  1a. Remove zero-feature rows (35 rows)
  1b. Remove exact duplicate rows (72 rows)
      Note: Duplicates are identified from the full dataset to ensure no
      cross-split duplicates remain. This is standard practice — it does not
      involve fitting a statistical model, so no leakage occurs.
  1c. Remove conflict-pair rows (4 rows: 5, 776, 7261, 8784 — excluding those
      already caught by zero-feature removal)
  Total removed: ≤111 rows (some overlap between steps)
  Remaining: ~11,487 rows

Step 2: FEATURE REDUCTION (whole-dataset, no fit involved — structural only)
  2a. Drop 1 feature from each of the 35 perfectly-correlated pairs
      (27 features dropped; 443 features remain)
  2b. [PENDING APPROVAL] Apply sparsity threshold — drop features with
      fewer than N non-zero samples (N to be determined from Option C comparison)

Step 3: STRATIFIED SPLIT (no fit involved)
  3a. Split into Train (70%) / Validation (10%) / Test (20%) using
      stratified sampling on the Class column.
  3b. Verify class proportions in each split match original distribution ±1%.

Step 4: LABEL ENCODING (fit on train labels only — trivial, no leakage risk)
  4a. Map Class {1,2,3,4,5} → {0,1,2,3,4}
  4b. Apply same mapping to val and test (no statistical fit needed)

Step 5: FEATURE SCALING (CRITICAL — fit ONLY on training data)
  5a. Fit StandardScaler on X_train ONLY
  5b. Transform X_train, X_val, X_test using training scaler
  → This is the primary leakage prevention point. Fitting the scaler on
    the full dataset would leak test-set distribution into training.

Step 6: FL PARTITIONING (train set only)
  6a. Partition X_train / y_train into N client shards
  6b. IID: random stratified split into N equal parts
  6c. Non-IID: Dirichlet(alpha) allocation (Hsu et al., 2019)
  → Validation and test sets are NOT partitioned.
```

**Why this order matters:**
- Steps 1–2 are structural (non-statistical) and can use the full dataset without leakage.
- Step 3 must come BEFORE Step 5 — the scaler must be fit on training rows only.
- Step 4 is order-invariant (no statistical fit).
- Step 6 must come AFTER Step 3 — clients only see training data.

---

## 6. SPLITTING STRATEGY

### 6.1 Proposed Split Ratios

| Split | Ratio | Approx. Samples (from ~11,487) | Purpose |
|-------|-------|-------------------------------|---------|
| **Train** | **70%** | **~8,041** | Centralized baseline training + FL client data |
| **Validation** | **10%** | **~1,149** | Hyperparameter tuning (model architecture, learning rate, etc.) |
| **Test** | **20%** | **~2,297** | Final evaluation — held out until all experiments complete |

### 6.2 Stratification Verification

With stratified sampling, expected per-class distribution in each split:

| Class | Category | Total (~11,487) | Train (~8,041) | Val (~1,149) | Test (~2,297) |
|-------|----------|-----------------|----------------|--------------|---------------|
| 1 | Adware | ~1,249 | ~874 | ~125 | ~250 |
| 2 | Banking | ~2,086 | ~1,460 | ~209 | ~417 |
| 3 | SMS | ~3,869 | ~2,708 | ~387 | ~774 |
| 4 | Riskware | ~2,528 | ~1,770 | ~253 | ~506 |
| 5 | Benign | ~1,755 | ~1,229 | ~175 | ~350 |

> **Note:** Actual counts will vary by ±1 due to integer rounding in stratified sampling.
> The smallest class (Adware) has ~874 training samples — sufficient for learning.

### 6.3 Confirmation

> ✅ **70/10/20 stratified split is confirmed as appropriate for this dataset.**

**Rationale:**
- 70% training (~8,041) is sufficient for FL partitioning across 5–20 clients with
  adequate per-client sample sizes (400–1,600 samples/client).
- 20% test (~2,297) is large enough to produce statistically reliable macro-F1 estimates.
  For 5 classes, 2,297 samples gives sufficient representation of even the minority class
  (Adware ~250 test samples → reliable per-class metrics).
- 10% validation is sufficient for hyperparameter selection given the relatively small
  hyperparameter search space expected for a neural network on this tabular dataset.

---

## 7. FEDERATED LEARNING PARTITIONING

### 7.1 Confirmed Architecture

> ✅ **Only the training set is partitioned among clients.**
> ✅ **Validation and test sets remain fully centralized throughout all experiments.**
> ✅ **The identical test set is used for BOTH centralized and federated evaluation.**

This architecture is mandatory for a valid comparison:
- **Centralized baseline:** Train on full training set, evaluate on centralized test set.
- **Federated baseline (IID):** Each client trains on their IID shard, FedAvg aggregates,
  evaluate on the SAME centralized test set.
- **Federated (Non-IID):** Same as above but with Dirichlet-partitioned shards.

Any deviation from using the identical test set would make the centralized vs federated
comparison invalid.

### 7.2 IID Partitioning

Partition `X_train` / `y_train` into N equal-sized shards using **stratified random split**,
maintaining class proportions within each client shard. This ensures IID conditions.

### 7.3 Non-IID Partitioning

Apply **Dirichlet(α) allocation** (Hsu et al., "Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification", 2019):
- Sample label proportions for each client from Dirichlet(α × ones(K)), where K=5 classes.
- Lower α → stronger class heterogeneity (more non-IID).
- Recommended α values for experiments: **α = 1.0** (mild), **α = 0.5** (moderate),
  **α = 0.1** (strong).
- With ~8,041 training samples and 5 clients, even α=0.1 will give each client ≥50 samples,
  which is workable.

### 7.4 Client Count

For the capstone, recommend: **5 clients** (represents hospitals, organizations, or device
clusters). Optionally also test **10 clients** for scaling experiments.

---

## 8. RECOMMENDED DECISIONS (Summary)

| # | Decision | Status | Recommendation |
|---|----------|--------|----------------|
| 1 | Zero-feature rows (35) | ✅ **RESOLVED** | DROP all 35 |
| 2 | Label-conflict rows (4, after zero-drop) | ✅ **RESOLVED** | DROP all 4 (rows 5, 776, 7261, 8784) |
| 3 | Perfectly correlated features (35 pairs) | ✅ **RESOLVED** | Drop 27 features per clique rules in Section 3 |
| 4 | High-sparsity features (247) | ⚠️ **UNRESOLVED** | Run Option C comparison after your approval |
| 5 | Preprocessing order | ✅ **RESOLVED** | Follow 6-step order in Section 5 |
| 6 | Train/Val/Test split | ✅ **RESOLVED** | 70/10/20, stratified |
| 7 | FL partitioning | ✅ **RESOLVED** | Train only; IID + Non-IID (α=0.1, 0.5, 1.0) |

---

## 9. UNRESOLVED DECISIONS (Awaiting Your Approval)

### Unresolved #1 — High-Sparsity Feature Handling

**Question:** Should we retain or remove the 247 features with >99% zero values?

**Options:**
- **A:** Retain all → 443 features after correlation drops
- **B:** Remove all → 223 features
- **C:** Run validation comparison, then decide → **Recommended**

**To run Option C, I need your approval to:**
1. Run a lightweight cross-validated experiment (logistic regression or single-layer MLP)
   comparing both feature sets on the validation set.
2. Report validation macro-F1 for both variants.
3. Select the feature set with higher performance as the canonical preprocessing pipeline.

> **This is the only remaining open question before the pipeline is fully specified.**

### Unresolved #2 — APK Hash Verification (Optional)

If you have access to original CICMalDroid 2020 APK metadata (SHA-256 hashes), we could
verify whether conflict rows 5/7261 and 776/8784 are the same APK files labeled differently.
This is informational only — the DROP decision stands regardless.

---

## 10. EXACT RECOMMENDED PREPROCESSING PIPELINE

```python
# ============================================================
# CICMalDroid 2020 — Recommended Preprocessing Pipeline
# (NOT implemented yet — awaiting approval)
# ============================================================

# --- Input ---
# df: raw DataFrame, 11,598 rows × 471 columns (470 features + Class)

# Step 1a: Remove zero-feature rows (35 rows)
zero_mask = (df[feature_cols] == 0).all(axis=1)
df = df[~zero_mask].copy()
# Expected: 11,598 - 35 = 11,563 rows

# Step 1b: Remove exact duplicate rows (keep first occurrence)
df = df.drop_duplicates(keep='first').reset_index(drop=True)
# Expected: ~11,491 rows (removes ~72 exact dups, some were already gone via step 1a)

# Step 1c: Remove conflict-pair rows (Groups 2 & 3)
# Note: Some of these rows may already be gone after 1a/1b
conflict_indices = [5, 776, 7261, 8784]
df = df.drop(index=[i for i in conflict_indices if i in df.index], errors='ignore')
df = df.reset_index(drop=True)
# Expected: ~11,487 rows

# Step 2a: Drop redundant correlated features (27 features to drop)
FEATURES_TO_DROP = [
    'CREATE_THREAD_____',   # r=1.0 with 'clone'
    'EXECUTE_____',         # r=1.0 with 'execve'
    "CREATE_PROCESS`_____", # r≈1.0 with 'fork'
    'TERMINATE_THREAD',     # r≈1.0 with 'exit'
    'setgroups32',          # r=1.0 with 'capset'
    'chown32',              # Clique F (keep 'mount')
    'getresgid32',          # Clique F
    'getresuid32',          # Clique F
    'endRestoreSession',    # Clique G (keep 'beginRestoreSession')
    'restorePackage',       # Clique G
    'isSyncActive',         # Clique H (keep 'cancelSync')
    'removeStatusChangeListener',  # Clique H
    'fstatfs64',            # Clique I (keep 'timer_create')
    'timer_settime',        # Clique I
    'getState',             # Clique J (keep 'isBluetoothA2dpOn')
    'startWatchingRoutes',  # Clique J
    'setPlaybackState',     # Clique K (keep 'setMetadata')
    'setTransportControlInfo',  # Clique K
    'sched_getscheduler',   # r=1.0 with 'sched_getparam'
    'updateServiceLocation',    # r=1.0 with 'ALTER_PHONE_STATE___'
    'setuid32',             # r=1.0 with 'setgid32'
    'setFlashlightEnabled', # r=1.0 with 'getFlashlightEnabled'
    'getLong',              # r=1.0 with 'getBoolean'
    'sched_get_priority_min',   # r=1.0 with 'sched_get_priority_max'
    'setWifiApConfiguration',   # r=1.0 with 'getWifiApConfiguration'
    'hasNamedWallpaper',    # r=1.0 with 'getAppWidgetInfo'
    'isPackageAvailable',   # r=1.0 with 'getIsSyncable'
]
df = df.drop(columns=FEATURES_TO_DROP, errors='ignore')
# Expected: 443 features remaining (470 - 27)

# Step 2b: [PENDING APPROVAL] Sparsity threshold — TBD after Option C comparison
# Placeholder: either keep all 443 or apply threshold

# Step 3: Stratified 70/10/20 split
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Class'])
y = df['Class']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.125, random_state=42, stratify=y_trainval
    # 0.125 of 80% = 10% of total
)

# Step 4: Label encoding {1,2,3,4,5} → {0,1,2,3,4}
label_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
y_train = y_train.map(label_map)
y_val   = y_val.map(label_map)
y_test  = y_test.map(label_map)

# Step 5: StandardScaler — fit on training data ONLY
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_val_scaled   = scaler.transform(X_val)          # transform only
X_test_scaled  = scaler.transform(X_test)         # transform only

# Step 6: FL Partitioning (applied to training set only)
# IID:
def iid_partition(X, y, n_clients, random_state=42):
    # Stratified split into n_clients equal shards
    ...

# Non-IID:
def dirichlet_partition(X, y, n_clients, alpha, random_state=42):
    # Dirichlet(alpha) allocation
    ...
```

---

*End of Preprocessing Decision Report*

**Status: Research-review complete. Awaiting your approval on Unresolved #1 (high-sparsity
feature comparison) before implementing the preprocessing pipeline.**
