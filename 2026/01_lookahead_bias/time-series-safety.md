# Time-Series Safety Rules

**Critical rules for any temporal data work: backtesting, feature engineering, signal generation, and analysis.**

## 1. Temporal Operations

| NEVER | ALWAYS |
|-------|--------|
| `shift(-1)`, `shift(-N)` | `shift(1)`, `shift(N)` |
| `rolling(center=True)` | `rolling()` (default `center=False`) |
| `bfill()`, `fillna(method='bfill')` | `ffill()` or constant fill |
| `diff(-1)` | `diff(1)` |
| `pct_change(fill_method='bfill')` | `pct_change(fill_method=None)` |
| `reindex(method='bfill')` | `reindex(method='ffill')` |

## 2. Missing Values: NEVER Use Backward Fill

**Rule:** ALWAYS use forward fill (`ffill`), NEVER backward fill (`bfill`).

| Method | Direction | Safe? | Why |
|--------|-----------|-------|-----|
| `ffill()` | Past → Present | ✓ Safe | Uses last known value (available at time t) |
| `bfill()` | Future → Present | ✗ **LOOKAHEAD BIAS** | Uses next value (not available at time t) |

```python
# Timeline: [100, NaN, NaN, 103, 104]
#              t0   t1   t2   t3   t4

# ffill() → [100, 100, 100, 103, 104]
# At t1, we use 100 (past value) - this is what we'd actually know

# bfill() → [100, 103, 103, 103, 104]  ← WRONG!
# At t1, we use 103 (future value from t3) - impossible to know at t1
```

### All Forbidden Backward Fill Variants

```python
# NEVER use any of these:
df.bfill()
df.fillna(method='bfill')
df.fillna(method='backfill')
df['col'].bfill()
df.reindex(new_index, method='bfill')
df.reindex(new_index, method='backfill')
df.asfreq('1H', method='bfill')
df.interpolate(method='linear')  # uses both past AND future!

# ALWAYS use forward fill:
df.ffill()
df.fillna(method='ffill')
df.fillna(method='pad')
df['col'].ffill()
df.reindex(new_index, method='ffill')
df.asfreq('1H', method='ffill')
df.fillna(0)  # constant fill is also safe
df.fillna(df.mean())  # ONLY if mean computed on train set
```

### Why This Matters

In backtesting/ML, backward fill creates **artificially good results** that won't replicate in live trading:
- Your model appears to predict better because it secretly has future information
- Sharpe ratios will be inflated
- Live performance will disappoint

## 3. Train/Test Split

- **Chronological split only**: `train = data[data.index < cutoff]`
- **NEVER** `shuffle=True` for time series
- **NEVER** `train_test_split(shuffle=True)` with temporal data

## 4. Preprocessing Order

```python
# CORRECT
X_train, X_test = split_chronological(X)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# WRONG - leaks test statistics into training
X_scaled = scaler.fit_transform(X)  # before split
X_train, X_test = split(X_scaled)
```

## 5. OHLCV Execution Timing

### Timestamp Convention

| NEVER | ALWAYS | Why |
|-------|--------|-----|
| `open_time` as index | `close_time` as index | Data available only at bar close |

```python
# Bar 10:00-11:00 → timestamp should be 11:00 (close_time)
# At 10:00 you don't know OHLCV yet, only at 11:00

# WRONG - open_time as timestamp
df.index = df['open_time']  # data appears available at 10:00

# CORRECT - close_time as timestamp
df.index = df['close_time']  # data available at 11:00
```

### Signal vs Execution

```python
# CORRECT - signal on close[t], execute on open[t+1]
signal = close.shift(1) > ma.shift(1)  # use t-1 data
entry_price = open  # execute at t

# WRONG - lookahead bias
signal = close > ma  # uses close[t] which isn't known yet
entry_price = close  # can't execute at close you just computed signal from
```

## 6. Feature Engineering

**Before split, NEVER compute:**
- Global rankings: `.rank()`
- Global z-scores: `(x - x.mean()) / x.std()`
- Global quantiles: `.quantile()`
- Global min/max normalization

**Use sklearn Pipeline** to ensure preprocessing inside CV folds.

### Aggregations with groupby

| NEVER | ALWAYS |
|-------|--------|
| `groupby().transform('mean')` | `groupby().transform(lambda x: x.expanding().mean().shift(1))` |
| `df['col'].rank()` | `df['col'].expanding().rank()` |
| `cumsum() / df['col'].sum()` | `cumsum() / expanding().sum()` |

```python
# WRONG - mean includes future values
df.groupby('symbol')['returns'].transform('mean')

# CORRECT - expanding mean with shift
df.groupby('symbol')['returns'].transform(
    lambda x: x.expanding().mean().shift(1)
)
```

### Regime Classification (percentile-based)

| Contexte | Méthode |
|----------|---------|
| **Analyse/exploration** | `rank(pct=True)` OK |
| **Backtest avec filtre régime** | `expanding().rank(pct=True)` requis |
| **Live trading** | Seuils absolus du train (ex: vol < 0.8 → Low) |

```python
# ANALYSE (post-hoc) - OK
pct = df['vol'].rank(pct=True)
df['regime'] = pd.cut(pct, bins=[0, 0.33, 0.67, 1.0], labels=['Low', 'Med', 'High'])

# BACKTEST (filtre signaux) - expanding requis
pct = df['vol'].expanding().rank(pct=True)

# LIVE - seuils absolus définis sur train
VOL_THRESHOLDS = train['vol'].quantile([0.33, 0.67]).values  # [0.8, 1.2]
df['regime'] = pd.cut(df['vol'], bins=[0, *VOL_THRESHOLDS, np.inf], labels=['Low', 'Med', 'High'])
```

### sklearn Preprocessing (must be AFTER split)

| NEVER before split | Why |
|--------------------|-----|
| `TargetEncoder().fit(X, y)` | Target leakage - use `TargetEncoder(cv=5)` |
| `PCA().fit(X)` | Covariance from test leaks |
| `SelectKBest().fit(X, y)` | Feature selection sees test |
| `KNNImputer().fit(X)` | Neighbors from future |

```python
# CORRECT - all preprocessing in Pipeline for CV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(k=10)),
    ('model', RandomForestClassifier())
])
```

## 7. Merge & Join Operations

### merge_asof (for bar aggregation, event alignment)

| NEVER | ALWAYS |
|-------|--------|
| `direction='forward'` | `direction='backward'` (default) |
| `direction='nearest'` (risky) | Explicit `direction='backward'` |

```python
# CORRECT - merge with past data only
pd.merge_asof(
    df_bars, df_features,
    on='timestamp',
    direction='backward'  # uses last available feature <= current bar
)

# WRONG - lookahead bias
pd.merge_asof(df_bars, df_features, on='timestamp', direction='forward')
```

### Regular merge/join

```python
# CORRECT - left join preserves bar timestamps
bars.merge(features, on=['timestamp', 'symbol'], how='left')

# CORRECT - join on index (left DataFrame's index preserved)
bars.join(features, how='left')

# RISKY - right join can pull in future timestamps
bars.merge(features, how='right')  # features timestamps may be ahead
```

### Key rules for joins:
- **Sort both DataFrames by timestamp** before merge_asof
- **Use `how='left'`** to preserve your bar timestamps
- **Validate alignment** after merge: no future data in features

## 8. Interpolation & Resampling

### interpolate() - most methods use future data!

| NEVER | ALWAYS |
|-------|--------|
| `interpolate(method='linear')` | `ffill()` or `interpolate(limit_direction='forward')` |
| `interpolate(method='nearest')` | `ffill()` |
| `interpolate(method='cubic')` | `ffill()` |
| `interpolate()` default | Explicit `limit_direction='forward'` |

```python
# WRONG - linear uses both past AND future values
df['price'].interpolate(method='linear')

# CORRECT - only use past values
df['price'].ffill()
df['price'].interpolate(method='linear', limit_direction='forward')
```

### resample() label parameter

| Context | label | Why |
|---------|-------|-----|
| Feature engineering | `'right'` | Label after data is complete |
| Display/reporting | `'left'` | OK for visualization |

```python
# WRONG for features - label='left' assigns value before bar closes
df.resample('1h', label='left').mean()

# CORRECT for features - label at end of period
df.resample('1h', label='right', closed='right').mean()
```

## 9. Signal Processing (NumPy/SciPy)

### NumPy

| NEVER | ALWAYS |
|-------|--------|
| `np.convolve(mode='same')` | `np.convolve(mode='valid')` |
| `np.convolve(mode='full')` | `np.convolve(mode='valid')` |
| `np.correlate(mode='same')` | `np.correlate(mode='valid')` |
| `np.interp(x, xp, fp)` | `np.searchsorted()` + ffill |
| `(x - x.mean()) / x.std()` | `expanding().mean/std()` |

```python
# WRONG - 'same' centers the convolution (uses future)
np.convolve(signal, kernel, mode='same')

# CORRECT - 'valid' only outputs where full overlap (truncates start)
np.convolve(signal, kernel, mode='valid')

# CORRECT - manual causal: pad at start, trim end
padded = np.pad(signal, (len(kernel)-1, 0), mode='edge')
result = np.convolve(padded, kernel, mode='valid')

# WRONG - np.interp uses linear interpolation (future)
np.interp(x_new, x_old, y_old)

# CORRECT - forward-fill manually
indices = np.searchsorted(x_old, x_new, side='right') - 1
result = y_old[np.clip(indices, 0, len(y_old)-1)]
```

### SciPy signal filters

| NEVER | ALWAYS |
|-------|--------|
| `filtfilt()` | `lfilter()` |
| `sosfiltfilt()` | `sosfilt()` |
| `savgol_filter()` | `rolling().apply()` custom |
| `gaussian_filter1d()` | `rolling().mean()` |

```python
# WRONG - filtfilt applies filter forward then backward (non-causal)
from scipy.signal import filtfilt
filtered = filtfilt(b, a, signal)  # uses future data!

# CORRECT - lfilter is causal (only past data)
from scipy.signal import lfilter
filtered = lfilter(b, a, signal)
```

### SciPy interpolate

| NEVER | ALWAYS |
|-------|--------|
| `interp1d(kind='linear')` | `interp1d(kind='previous')` |
| `interp1d(kind='cubic')` | `interp1d(kind='previous')` |
| `CubicSpline()` | `interp1d(kind='previous')` |

### SciPy stats

| NEVER | ALWAYS |
|-------|--------|
| `stats.zscore(x)` | `(x - x.expanding().mean()) / x.expanding().std()` |

## 10. Technical Indicators

### Warmup Period

| NEVER | ALWAYS |
|-------|--------|
| Use indicator before warmup complete | Skip first N bars (N = indicator period) |
| RSI(14) on first 14 bars | Start using RSI from bar 15+ |
| Bollinger on subset | Calculate on full history, use after warmup |

### OHLCV Current Bar

| NEVER | ALWAYS |
|-------|--------|
| `high` for same-bar decision | `high.shift(1)` |
| `low` for same-bar decision | `low.shift(1)` |
| `volume` for same-bar signal | `volume.shift(1)` |
| `close` for same-bar signal | `close.shift(1)` |

```python
# WRONG - uses current bar's high/low (not known until bar closes)
breakout = close > high.rolling(20).max()

# CORRECT - use previous bar's data
breakout = close.shift(1) > high.shift(1).rolling(20).max()
```

### Pivot Points & VWAP

| NEVER | ALWAYS |
|-------|--------|
| Pivots with current OHLC | `pivot = (prev_high + prev_low + prev_close) / 3` |
| VWAP without session reset | Reset VWAP at each session open |
| Cumulative VWAP across days | Intraday VWAP only |

## 11. Safe Patterns (OK to use)

- `shift(1)`, `shift(N)` with N > 0
- `rolling(window)` without `center=True`
- `ffill()` for missing data
- `expanding()` - uses only past data
- `ewm()` - exponential weighted, backward-looking
- `cumsum()`, `cumprod()`, `cummax()`, `cummin()`
- `merge_asof(direction='backward')` - default, safe
- `merge(how='left')` - preserves left DataFrame timestamps
- `interpolate(limit_direction='forward')` - only past values
- `resample(label='right')` - label after period ends
- `np.convolve(mode='valid')` - no future data
- `scipy.signal.lfilter()` - causal filter
- `interp1d(kind='previous')` - forward-fill interpolation
- `expanding().rank()` - temporal rank
- `groupby().transform(lambda x: x.expanding().mean().shift(1))` - safe groupby mean

## 12. Quick Checklist

Before writing any ML code, verify:
- [ ] All shifts are positive (backward-looking)
- [ ] **NEVER `bfill()`** - always `ffill()` for missing values
- [ ] No shuffle in splits
- [ ] Scalers/encoders fit on train only
- [ ] Signal computed before execution bar
- [ ] merge_asof uses `direction='backward'`
- [ ] Joins use `how='left'` to preserve bar timestamps
- [ ] interpolate uses `ffill()` or `limit_direction='forward'`
- [ ] resample uses `label='right'` for features
- [ ] np.convolve uses `mode='valid'`
- [ ] scipy filters use `lfilter` not `filtfilt`
- [ ] scipy.interpolate uses `kind='previous'`
- [ ] groupby aggregations use `expanding().shift(1)`
- [ ] TargetEncoder uses `cv=5`
- [ ] PCA/SelectKBest inside Pipeline
- [ ] Indicators have warmup period respected
- [ ] OHLCV uses `.shift(1)` for same-bar decisions
- [ ] Timestamp uses `close_time` not `open_time`

---

# Audit Configuration (for /bias-check)

## 13. Severity Levels

| Level | Icon | Description |
|-------|------|-------------|
| CRITICAL | 🔴 | Guaranteed leakage, always wrong |
| HIGH | 🟠 | Almost always leakage, rare exceptions |
| MEDIUM | 🟡 | Context-dependent, review required |

## 14. Detection Patterns (Search Keywords)

Search for these keywords in `src/`, then analyze context to determine severity.

### 🔴 CRITICAL: Lookahead Bias
| Search for | Issue |
|------------|-------|
| `shift(` with negative value | Future data access |
| `rolling(` with `center=True` | Centered window uses future |
| `bfill`, `backfill` | Backward fill propagates future |
| `merge_asof` with `direction='forward'` | Joins future data |
| `filtfilt`, `sosfiltfilt` | Non-causal filters |
| `interpolate(` without `limit_direction='forward'` | Default uses future |

### 🔴 CRITICAL: Train/Test Contamination
| Search for | Issue |
|------------|-------|
| `shuffle=True` or `shuffle = True` | Breaks temporal order |
| `KFold(`, `StratifiedKFold(` | Not temporal-aware |
| `fit_transform(` before split | Leaks test stats |
| `train_test_split(` without `shuffle=False` | Default shuffles |

### 🔴 CRITICAL: Preprocessing Before Split
| Search for | Issue |
|------------|-------|
| `.rank()` on full dataset | Global ranking |
| `.quantile(` on full dataset | Global percentiles |
| `zscore(` | Global normalization |
| `StandardScaler().fit(X)` before split | Leaks test stats |

### 🟡 MEDIUM: Context-Dependent
| Search for | When OK |
|------------|---------|
| `.cumsum()` | OK for cumulative returns |
| `transform('mean')` | OK if cross-sectional (across symbols) |
| `.expanding()` | Usually safe, verify start point |

## 15. Report Format

```
# Bias Check Report

## Summary
- Files scanned: N
- 🔴 CRITICAL: X
- 🟡 MEDIUM: Y

## Findings
### CRITICAL
- file.py:123 - `.bfill()` - Backward fill

### MEDIUM (Review)
- features.py:89 - `.cumsum()` - Verify intentional

## Conclusion
✅ Clean / ⚠️ N issues found
```

## 16. Exemptions

**Exempt directories** (intentionally use future data):
- `src/labeling/` - Target/label generation
- `src/analysis/` - IC analysis (forward returns for signal validation)
- `tests/` - Test fixtures
- `notebooks/` - Exploratory analysis

**Within exempted dirs, still flag** train/test split contamination.
