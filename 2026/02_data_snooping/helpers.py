"""
Helpers pour la démo Data Snooping en Trading Algorithmique.
Toute la logique (calcul, backtest, plots) est ici.
Le notebook n'appelle que ces fonctions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import vectorbt as vbt
from pathlib import Path
from itertools import product
import warnings


# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

def setup():
    """Configure plotting, warnings et vectorbt."""
    warnings.filterwarnings('ignore')
    plt.style.use('dark_background')
    sns.set_theme(style='darkgrid', rc={'figure.figsize': (14, 6)})
    vbt.settings.portfolio['freq'] = '4h'
    print('Setup OK')


# ─────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────

def load_data(path='data/btc_usdt_h4.csv'):
    """Charge les données BTC/USDT H4 depuis le cache CSV (ou Binance si absent)."""
    DATA_PATH = Path(path)

    if DATA_PATH.exists():
        print('Chargement depuis le cache local...')
        df = pd.read_csv(DATA_PATH, index_col='timestamp', parse_dates=True)
    else:
        print('Téléchargement depuis Binance...')
        from binance.client import Client
        client = Client('', '')
        klines = client.get_historical_klines(
            'BTCUSDT',
            Client.KLINE_INTERVAL_4HOUR,
            '2020-01-01',
            '2026-01-01'
        )
        cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        df = pd.DataFrame(klines, columns=cols)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH)
        print(f'Sauvegarde en {DATA_PATH}')

    print(f'Période : {df.index[0]} → {df.index[-1]}')
    print(f'Nombre de bougies H4 : {len(df):,}')
    return df


def split_train_test(df, ratio=0.7):
    """Split train/test et affiche les infos."""
    split_idx = int(len(df) * ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    print(f'Train : {train.index[0]} → {train.index[-1]}  ({len(train):,} bougies)')
    print(f'Test  : {test.index[0]} → {test.index[-1]}  ({len(test):,} bougies)')
    return train, test


# ─────────────────────────────────────────────
# Grid Backtest
# ─────────────────────────────────────────────

def print_grid_info(fast_windows, slow_windows, tp_values, sl_values):
    """Affiche les infos de la grille."""
    ma_pairs = [(f, s) for f in fast_windows for s in slow_windows if f < s]
    n_tpsl = len(tp_values) * len(sl_values)
    total = len(ma_pairs) * n_tpsl
    print(f'Fast MA windows  : {len(fast_windows)} valeurs ({fast_windows[0]} à {fast_windows[-1]})')
    print(f'Slow MA windows  : {len(slow_windows)} valeurs ({slow_windows[0]} à {slow_windows[-1]})')
    print(f'MA pairs (f < s) : {len(ma_pairs)}')
    print(f'TP values        : {len(tp_values)} valeurs')
    print(f'SL values        : {len(sl_values)} valeurs')
    print(f'TP/SL combos     : {n_tpsl}')
    print(f'Total combos     : {total:,}')
    return total


def run_grid_backtest(close_train, fast_windows, slow_windows, tp_values, sl_values):
    """
    Lance le backtest sur toute la grille MA x TP/SL.
    Retourne (pf, tp_sl_combos, n_ma_pairs).
    """
    # Paires MA valides
    ma_pairs = [(f, s) for f in fast_windows for s in slow_windows if f < s]
    pair_index = pd.MultiIndex.from_tuples(ma_pairs, names=['fast_window', 'slow_window'])

    # Calcul de toutes les MAs
    all_windows = sorted(set(fast_windows + slow_windows))
    ma_df = vbt.MA.run(close_train, window=all_windows).ma

    # Crossover vectorise (numpy 3D)
    fast_vals = ma_df[fast_windows].values
    slow_vals = ma_df[slow_windows].values
    fast_3d = fast_vals[:, :, np.newaxis]
    slow_3d = slow_vals[:, np.newaxis, :]

    above = fast_3d > slow_3d
    above_prev = np.roll(above, 1, axis=0)
    above_prev[0] = False
    crossed_above = above & ~above_prev
    crossed_below = ~above & above_prev

    # Reshape et filtre fast < slow
    valid_mask = np.array([[f < s for s in slow_windows] for f in fast_windows])
    entries_data = crossed_above.reshape(len(close_train), -1)[:, valid_mask.flatten()]
    exits_data = crossed_below.reshape(len(close_train), -1)[:, valid_mask.flatten()]

    entries = pd.DataFrame(entries_data, index=close_train.index, columns=pair_index)
    exits = pd.DataFrame(exits_data, index=close_train.index, columns=pair_index)
    n_ma = entries.shape[1]
    print(f'Paires MA générées : {n_ma}')

    # Tiler pour chaque combo TP/SL
    tp_sl_combos = list(product(tp_values, sl_values))
    n_tpsl = len(tp_sl_combos)

    entries_tiled = entries.vbt.tile(n_tpsl, keys=pd.Index(range(n_tpsl), name='tp_sl_id'))
    exits_tiled = exits.vbt.tile(n_tpsl, keys=pd.Index(range(n_tpsl), name='tp_sl_id'))

    tp_per_col = np.array([tp / 100 for tp, sl in tp_sl_combos for _ in range(n_ma)])
    sl_per_col = np.array([sl / 100 for tp, sl in tp_sl_combos for _ in range(n_ma)])

    print(f'Colonnes totales : {entries_tiled.shape[1]:,}')

    # Backtest
    pf = vbt.Portfolio.from_signals(
        close_train, entries_tiled, exits_tiled,
        tp_stop=tp_per_col,
        sl_stop=sl_per_col,
        init_cash=10_000,
        fees=0.001,
        freq='4h'
    )
    print(f'Backtest terminé — {entries_tiled.shape[1]:,} stratégies évaluées')
    return pf, tp_sl_combos, n_ma


def get_best_is_results(pf):
    """Extrait les métriques de la meilleure stratégie IS. Retourne un dict."""
    sharpes = pf.sharpe_ratio()
    returns = pf.total_return()
    max_dds = pf.max_drawdown()

    best_col = sharpes.idxmax()
    return {
        'sharpes': sharpes,
        'returns': returns,
        'max_dds': max_dds,
        'best_col': best_col,
        'best_sharpe': sharpes.loc[best_col],
        'best_return': returns.loc[best_col],
        'best_dd': max_dds.loc[best_col],
    }


def print_best_is(results, label=''):
    """Affiche les métriques de la meilleure stratégie IS."""
    print('=' * 60)
    print(f'{label} — Meilleure stratégie IN-SAMPLE')
    print('=' * 60)
    print(f'Colonne          : {results["best_col"]}')
    print(f'Sharpe Ratio     : {results["best_sharpe"]:.3f}')
    print(f'Total Return     : {results["best_return"]:.2%}')
    print(f'Max Drawdown     : {results["best_dd"]:.2%}')
    print('=' * 60)


def extract_best_params(best_col, tp_sl_combos):
    """Extrait fast_w, slow_w, tp, sl depuis la colonne du best."""
    if isinstance(best_col, tuple):
        tp_sl_id = best_col[0]
        fast_w = best_col[1]
        slow_w = best_col[2]
    else:
        tp_sl_id = best_col
        fast_w, slow_w = None, None

    best_tp, best_sl = tp_sl_combos[tp_sl_id]
    print(f'Meilleure config :')
    print(f'  Fast MA  = {fast_w}')
    print(f'  Slow MA  = {slow_w}')
    print(f'  TP       = {best_tp}%')
    print(f'  SL       = {best_sl}%')
    return fast_w, slow_w, best_tp, best_sl


# ─────────────────────────────────────────────
# Out-of-Sample
# ─────────────────────────────────────────────

def run_oos_backtest(close_test, fast_w, slow_w, tp, sl):
    """Lance le backtest OOS sur une config unique. Retourne le portfolio."""
    fast_ma = vbt.MA.run(close_test, window=fast_w)
    slow_ma = vbt.MA.run(close_test, window=slow_w)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(
        close_test, entries, exits,
        tp_stop=tp / 100,
        sl_stop=sl / 100,
        init_cash=10_000,
        fees=0.001,
        freq='4h'
    )
    return pf


def print_is_vs_oos(is_results, pf_oos, label=''):
    """Affiche la comparaison IS vs OOS."""
    oos_sharpe = pf_oos.sharpe_ratio()
    oos_return = pf_oos.total_return()
    oos_dd = pf_oos.max_drawdown()

    print('=' * 60)
    print(f'{label} — Comparaison IS vs OOS')
    print('=' * 60)
    print(f'{"Métrique":<20} {"In-Sample":>12} {"Out-of-Sample":>14}')
    print('-' * 60)
    print(f'{"Sharpe Ratio":<20} {is_results["best_sharpe"]:>12.3f} {oos_sharpe:>14.3f}')
    print(f'{"Total Return":<20} {is_results["best_return"]:>12.2%} {oos_return:>14.2%}')
    print(f'{"Max Drawdown":<20} {is_results["best_dd"]:>12.2%} {oos_dd:>14.2%}')
    print('=' * 60)

    return oos_sharpe, oos_return, oos_dd


# ─────────────────────────────────────────────
# Sensitivity Analysis
# ─────────────────────────────────────────────

def run_sensitivity(close_train, fast_w, slow_w, best_tp, best_sl):
    """
    Test de sensibilité ±10% sur chaque paramètre (un à la fois).
    Retourne un dict {param_name: [(perturbation%, sharpe), ...]}.
    """
    perturb = np.arange(-10, 11, 1)

    fast_range = sorted(set(max(2, round(fast_w * (1 + p / 100))) for p in perturb))
    slow_range = sorted(set(round(slow_w * (1 + p / 100)) for p in perturb))
    all_w = sorted(set(fast_range + slow_range))
    mas = vbt.MA.run(close_train, window=all_w).ma

    def backtest_one(fw, sw, tp, sl):
        cross = mas[fw] > mas[sw]
        e = cross & ~cross.shift(1, fill_value=False)
        x = ~cross & cross.shift(1, fill_value=False)
        pf = vbt.Portfolio.from_signals(
            close_train, e, x,
            tp_stop=tp / 100, sl_stop=sl / 100,
            init_cash=10_000, fees=0.001, freq='4h')
        return pf.sharpe_ratio()

    sensitivity = {
        'Fast MA':     [(p, backtest_one(max(2, round(fast_w * (1 + p / 100))), slow_w, best_tp, best_sl)) for p in perturb],
        'Slow MA':     [(p, backtest_one(fast_w, round(slow_w * (1 + p / 100)), best_tp, best_sl)) for p in perturb],
        'Take Profit': [(p, backtest_one(fast_w, slow_w, best_tp * (1 + p / 100), best_sl)) for p in perturb],
        'Stop Loss':   [(p, backtest_one(fast_w, slow_w, best_tp, best_sl * (1 + p / 100))) for p in perturb],
    }
    return sensitivity


def print_sensitivity_stats(sensitivity, best_sharpe, fast_w, slow_w, best_tp, best_sl):
    """Affiche les stats de sensibilité."""
    print(f'Paramètres originaux : Fast={fast_w}, Slow={slow_w}, TP={best_tp}%, SL={best_sl}%')
    print(f'Sharpe original : {best_sharpe:.3f}\n')
    for name, data in sensitivity.items():
        sharpes = [s for _, s in data]
        print(f'  {name:12s} : min={min(sharpes):.3f}, max={max(sharpes):.3f}, écart={max(sharpes) - min(sharpes):.3f}')


# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

def summary_table(is_lg, oos_lg, n_combos_lg, is_sm, oos_sm, n_combos_sm):
    """Crée le DataFrame de synthèse."""
    oos_sharpe_lg, oos_return_lg, oos_dd_lg = oos_lg
    oos_sharpe_sm, oos_return_sm, oos_dd_sm = oos_sm

    summary = pd.DataFrame({
        'Métrique': [
            'Nombre de combos', 'Best Sharpe IS', 'Sharpe OOS', 'Dégradation Sharpe',
            'Total Return IS', 'Total Return OOS', 'Max DD IS', 'Max DD OOS'
        ],
        'Grande Grille': [
            f'{n_combos_lg:,}',
            f'{is_lg["best_sharpe"]:.3f}',
            f'{oos_sharpe_lg:.3f}',
            f'{((oos_sharpe_lg - is_lg["best_sharpe"]) / abs(is_lg["best_sharpe"]) * 100):.1f}%',
            f'{is_lg["best_return"]:.2%}',
            f'{oos_return_lg:.2%}',
            f'{is_lg["best_dd"]:.2%}',
            f'{oos_dd_lg:.2%}'
        ],
        'Petite Grille': [
            f'{n_combos_sm}',
            f'{is_sm["best_sharpe"]:.3f}',
            f'{oos_sharpe_sm:.3f}',
            f'{((oos_sharpe_sm - is_sm["best_sharpe"]) / abs(is_sm["best_sharpe"]) * 100):.1f}%',
            f'{is_sm["best_return"]:.2%}',
            f'{oos_return_sm:.2%}',
            f'{is_sm["best_dd"]:.2%}',
            f'{oos_dd_sm:.2%}'
        ]
    })
    return summary.style.set_properties(**{
        'text-align': 'center'
    }).set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]}
    ])


# ═════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════

def _style_ax(ax):
    """Style commun dark pour un axe."""
    ax.set_facecolor('#0a0a0a')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#555555')


def plot_train_test_split(train, test):
    """Plot du split train/test."""
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor('#0a0a0a')
    _style_ax(ax)
    ax.plot(train.index, train['close'], color='#00d4ff', label='Train (70%)', linewidth=0.8)
    ax.plot(test.index, test['close'], color='#ff6b6b', label='Test (30%)', linewidth=0.8)
    ax.axvline(test.index[0], color='white', linestyle='--', alpha=0.5, label='Split')
    ax.set_title('BTC/USDT H4 — Split Train / Test', fontsize=14, color='white')
    ax.set_ylabel('Prix (USDT)', color='white')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.tight_layout()
    plt.show()


def plot_equity_is_oos(pf_is, pf_oos, sharpe_is, sharpe_oos, title_prefix=''):
    """Plot equity curves IS vs OOS côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor('#0a0a0a')

    eq_is = pf_is.value()
    _style_ax(axes[0])
    axes[0].plot(eq_is.index, eq_is.values, color='#00d4ff', linewidth=1)
    axes[0].axhline(10_000, color='white', linestyle='--', alpha=0.3)
    axes[0].set_title(f'{title_prefix} In-Sample (Sharpe: {sharpe_is:.2f})', fontsize=12, color='white')
    axes[0].set_ylabel('Portfolio Value ($)', color='white')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    eq_oos = pf_oos.value()
    _style_ax(axes[1])
    axes[1].plot(eq_oos.index, eq_oos.values, color='#ff6b6b', linewidth=1)
    axes[1].axhline(10_000, color='white', linestyle='--', alpha=0.3)
    axes[1].set_title(f'{title_prefix} Out-of-Sample (Sharpe: {sharpe_oos:.2f})', fontsize=12, color='white')
    axes[1].set_ylabel('Portfolio Value ($)', color='white')
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    plt.tight_layout()
    plt.show()


def plot_comparison_2x2(pf_is_lg, pf_oos_lg, sharpe_is_lg, oos_sharpe_lg,
                        pf_is_sm, pf_oos_sm, sharpe_is_sm, oos_sharpe_sm):
    """Plot 2x2 : Grande Grille IS/OOS vs Petite Grille IS/OOS."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0a')

    configs = [
        (axes[0, 0], pf_is_lg, '#00d4ff', f'Grande Grille IS (Sharpe: {sharpe_is_lg:.2f})'),
        (axes[0, 1], pf_oos_lg, '#ff6b6b', f'Grande Grille OOS (Sharpe: {oos_sharpe_lg:.2f})'),
        (axes[1, 0], pf_is_sm, '#00d4ff', f'Petite Grille IS (Sharpe: {sharpe_is_sm:.2f})'),
        (axes[1, 1], pf_oos_sm, '#ff6b6b', f'Petite Grille OOS (Sharpe: {oos_sharpe_sm:.2f})'),
    ]

    for ax, pf, color, title in configs:
        eq = pf.value()
        _style_ax(ax)
        ax.plot(eq.index, eq.values, color=color, linewidth=2)
        ax.axhline(10_000, color='white', linestyle='--', alpha=0.3)
        ax.set_title(title, fontsize=11, color='white')
        ax.set_ylabel('Portfolio ($)', color='white')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    plt.suptitle('Data Snooping : Grande Grille vs Petite Grille', fontsize=14, y=1.02, color='white')
    plt.tight_layout()
    plt.show()


def plot_sharpe_degradation_bar(sharpe_is_lg, oos_sharpe_lg, sharpe_is_sm, oos_sharpe_sm):
    """Bar chart de la dégradation IS → OOS."""
    labels = ['Grande Grille\n(~100,000 combos)', 'Petite Grille\n(36 combos)']
    is_sharpes = [sharpe_is_lg, sharpe_is_sm]
    oos_sharpes = [oos_sharpe_lg, oos_sharpe_sm]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0a0a0a')
    _style_ax(ax)
    bars1 = ax.bar(x - width / 2, is_sharpes, width, label='In-Sample', color='#00d4ff', alpha=0.8)
    bars2 = ax.bar(x + width / 2, oos_sharpes, width, label='Out-of-Sample', color='#ff6b6b', alpha=0.8)

    ax.set_ylabel('Sharpe Ratio', fontsize=12, color='white')
    ax.set_title('Dégradation du Sharpe Ratio : IS vs OOS', fontsize=14, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, color='white')
    ax.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')
    ax.axhline(0, color='white', linestyle='-', alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10, color='#00d4ff')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.02 if h >= 0 else h - 0.08,
                f'{h:.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=10, color='#ff6b6b')

    plt.tight_layout()
    plt.show()


def plot_sharpe_distribution(sharpes, best_sharpe, title=''):
    """Distribution annotée des Sharpe IS avec seuils."""
    sharpes_clean = sharpes.replace([np.inf, -np.inf], np.nan).dropna()
    total = len(sharpes_clean)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a0a')
    _style_ax(ax)

    ax.hist(sharpes_clean.values, bins=120, color='#00d4ff', alpha=0.7, edgecolor='none')
    ax.axvline(best_sharpe, color='#ff6b6b', linestyle='--', linewidth=2,
               label=f'Best: {best_sharpe:.2f}')

    thresholds = [1.0, 1.5, 2.0]
    for th in thresholds:
        count = (sharpes_clean > th).sum()
        pct = count / total * 100
        ax.axvline(th, color='#ffd700', linestyle=':', linewidth=1.5, alpha=0.8)
        ax.annotate(f'Sharpe > {th}\n{count:,} strats ({pct:.1f}%)',
                    xy=(th, ax.get_ylim()[1] * 0.85),
                    xytext=(th + 0.08, ax.get_ylim()[1] * 0.85),
                    fontsize=9, color='#ffd700', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#ffd700', lw=1.2))

    ax.set_title(f'{title}Distribution des Sharpe IS — {total:,} stratégies', fontsize=14, color='white')
    ax.set_xlabel('Sharpe Ratio', color='white')
    ax.set_ylabel('Nombre de stratégies', color='white')
    ax.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')
    plt.tight_layout()
    plt.show()

    print(f'\nSur {total:,} stratégies testées :')
    for th in thresholds:
        count = (sharpes_clean > th).sum()
        pct = count / total * 100
        print(f'  Sharpe > {th:.1f} : {count:,} stratégies ({pct:.1f}%)')


def plot_distributions_comparison(sharpes_lg, best_sharpe_lg, oos_sharpe_lg,
                                  sharpes_sm, best_sharpe_sm, oos_sharpe_sm):
    """Comparaison des distributions Grande vs Petite Grille."""
    sharpes_clean_lg = sharpes_lg.replace([np.inf, -np.inf], np.nan).dropna()
    sharpes_clean_sm = sharpes_sm.replace([np.inf, -np.inf], np.nan).dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor('#0a0a0a')

    _style_ax(ax1)
    ax1.hist(sharpes_clean_lg.values, bins=30, color='#00d4ff', alpha=0.7, edgecolor='none')
    ax1.axvline(best_sharpe_lg, color='#ff6b6b', linestyle='--', linewidth=2,
                label=f'Best: {best_sharpe_lg:.2f} (OOS: {oos_sharpe_lg:.2f})')
    ax1.set_title(f'Grande Grille — {len(sharpes_clean_lg):,} stratégies', fontsize=13, color='white')
    ax1.set_ylabel('Nombre de stratégies', color='white')
    ax1.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')

    _style_ax(ax2)
    ax2.hist(sharpes_clean_sm.values, bins=15, color='#00ff88', alpha=0.7, edgecolor='none')
    ax2.axvline(best_sharpe_sm, color='#ff6b6b', linestyle='--', linewidth=2,
                label=f'Best: {best_sharpe_sm:.2f} (OOS: {oos_sharpe_sm:.2f})')
    ax2.set_title(f'Petite Grille — {len(sharpes_clean_sm)} stratégies', fontsize=13, color='white')
    ax2.set_xlabel('Sharpe Ratio', fontsize=12, color='white')
    ax2.set_ylabel('Nombre de stratégies', color='white')
    ax2.legend(fontsize=11, facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')

    plt.suptitle('Distribution des Sharpe IS : plus la grille est grande, plus le "best" est un outlier',
                 fontsize=14, color='white', y=1.02)
    plt.tight_layout()
    plt.show()

    print(f'Grande Grille : {len(sharpes_clean_lg):,} stratégies → best IS = {best_sharpe_lg:.2f}, écart au médiane = {best_sharpe_lg - sharpes_clean_lg.median():.2f}')
    print(f'Petite Grille : {len(sharpes_clean_sm)} stratégies  → best IS = {best_sharpe_sm:.2f}, écart au médiane = {best_sharpe_sm - sharpes_clean_sm.median():.2f}')


def plot_sensitivity(sensitivity, best_sharpe, fast_w, slow_w, best_tp, best_sl):
    """Plot des courbes de sensibilité ±10%."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#0a0a0a')

    for ax, (name, data) in zip(axes.flat, sensitivity.items()):
        pcts, sharpes = zip(*data)
        _style_ax(ax)
        ax.plot(pcts, sharpes, color='#ffd700', linewidth=2, marker='o', markersize=3)
        ax.axhline(best_sharpe, color='#ff6b6b', linestyle='--', alpha=0.7, label=f'Best: {best_sharpe:.2f}')
        ax.axvline(0, color='white', linestyle=':', alpha=0.3)
        ax.set_title(name, fontsize=12, color='white')
        ax.set_xlabel('Perturbation (%)', color='white')
        ax.set_ylabel('Sharpe Ratio', color='white')
        ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#333333', labelcolor='white')

    plt.suptitle(f'Sensibilité ±10% — Paramètres originaux : Fast={fast_w}, Slow={slow_w}, TP={best_tp}%, SL={best_sl}%',
                 fontsize=13, color='white')
    plt.tight_layout()
    plt.show()
