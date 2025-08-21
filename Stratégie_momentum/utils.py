"""
Module pour récupérer les données de cryptomonnaies depuis Binance
et traitement des indicateurs techniques
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union
from binance.client import Client
from datetime import datetime
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import column
from bokeh.models import DatetimeTickFormatter, Span
import warnings
warnings.filterwarnings('ignore')

# Constants
KLINE_COLUMNS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]
PRICE_VOLUME_COLS = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
DROP_COLS = ['ignore', 'timestamp']

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_date(date_input: Union[str, datetime]) -> str:
    """Convert date input to Binance API format."""
    if isinstance(date_input, datetime):
        return date_input.strftime("%d %b %Y")
    return datetime.strptime(date_input, "%Y-%m-%d").strftime("%d %b %Y")


def fetch_binance_data(
    symbol: str, 
    start_date: Union[str, datetime], 
    end_date: Union[str, datetime], 
    interval: str = Client.KLINE_INTERVAL_4HOUR
) -> Optional[pd.DataFrame]:
    """Fetch candlestick data for a cryptocurrency from Binance.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        start_date: Start date in 'YYYY-MM-DD' format or datetime object
        end_date: End date in 'YYYY-MM-DD' format or datetime object  
        interval: Time interval (default: 4 hours)
        
    Returns:
        DataFrame with OHLCV data or None if error
    """
    client = Client()
    
    try:
        start_str = _format_date(start_date)
        end_str = _format_date(end_date)
        
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        
        if not klines:
            logger.warning(f"No data retrieved for {symbol}")
            return pd.DataFrame()
        
        df = pd.DataFrame(klines, columns=KLINE_COLUMNS)
        df = df.drop(columns=DROP_COLS)
        
        # Optimize type conversions
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        df[PRICE_VOLUME_COLS] = df[PRICE_VOLUME_COLS].astype(float)
        df['symbol'] = symbol
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None


def fetch_multiple_cryptos(
    crypto_list: List[str], 
    start_date: Union[str, datetime], 
    end_date: Union[str, datetime], 
    interval: str = Client.KLINE_INTERVAL_4HOUR
) -> pd.DataFrame:
    """Fetch data for multiple cryptocurrencies and combine into single DataFrame.
    
    Args:
        crypto_list: List of cryptocurrency symbols
        start_date: Start date in 'YYYY-MM-DD' format or datetime object
        end_date: End date in 'YYYY-MM-DD' format or datetime object
        interval: Time interval (default: 4 hours)
        
    Returns:
        Combined DataFrame with all cryptocurrency data
    """
    logger.info(f"Fetching data for {len(crypto_list)} cryptocurrencies from {start_date} to {end_date}")
    
    all_crypto_data = []
    successful_fetches = 0
    
    for symbol in crypto_list:
        logger.info(f"Fetching data for {symbol}")
        df = fetch_binance_data(symbol, start_date, end_date, interval)
        
        if df is not None and not df.empty:
            all_crypto_data.append(df)
            successful_fetches += 1
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        else:
            logger.warning(f"Failed to fetch data for {symbol}")
    
    logger.info(f"Data fetch completed for {successful_fetches}/{len(crypto_list)} cryptocurrencies")
    
    if not all_crypto_data:
        logger.error("No data could be retrieved")
        return pd.DataFrame()
    
    crypto_df = pd.concat(all_crypto_data, ignore_index=True)
    crypto_df = crypto_df.sort_values(['close_time', 'symbol']).set_index('close_time')
    
    return crypto_df


def _validate_columns(df: pd.DataFrame, columns: List[str]) -> None:
    """Validate that required columns exist in DataFrame."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def generate_clean_signals(
    df: pd.DataFrame,
    short_tstat_col: str,
    long_tstat_col: str,
    long_sma_col: str,
    short_up_level: float,
    short_down_level: float,
    long_min_threshold: float,
    max_holding_periods: Optional[int] = None,
    output_col: str = "signal",
) -> pd.Series:
    """
    Génère des signaux de trading propres avec tous les filtres appliqués en une seule passe.

    Combine la détection de franchissement, le filtre de momentum,
    l'alternance naturelle des signaux et la gestion de la limite de temps.

    Règles de signal :
        ACHAT (+1) : Toutes les conditions doivent être réunies :
            - Le t-stat court franchit à la hausse short_up_level
            - Le t-stat long >= long_min_threshold
            - Le t-stat long >= sa SMA (momentum en amélioration)
            - Pas déjà en position

        VENTE (-1) : Une seule condition suffit pour sortir :
            - Le t-stat court franchit à la baisse short_down_level
            - Limite de temps atteinte (si précisée)

    Args:
        df: DataFrame contenant les colonnes de t-stat (doit être trié par temps)
        short_tstat_col: Nom de la colonne du t-stat court terme
        long_tstat_col: Nom de la colonne du t-stat long terme
        long_sma_col: Nom de la colonne de la SMA du t-stat long
        short_up_level: Seuil de franchissement haussier pour le t-stat court (ex: +2.0)
        short_down_level: Seuil de franchissement baissier pour le t-stat court (ex: -2.0)
        long_min_threshold: Seuil minimum pour le t-stat long
        max_holding_periods: Nombre maximal de périodes en position (optionnel)
        output_col: Nom de la colonne de sortie (défaut : "signal")

    Returns:
        Série des signaux propres avec alternance naturelle
    """
    # Vérification que toutes les colonnes requises sont présentes dans le DataFrame
    required_cols = [short_tstat_col, long_tstat_col, long_sma_col]
    _validate_columns(df, required_cols)
    
    # EXTRACTION ET NETTOYAGE DES DONNÉES
    # Conversion en valeurs numériques avec gestion des erreurs
    # Les valeurs non numériques (NaN, string, etc.) sont remplacées par 0
    short_tstat = pd.to_numeric(df[short_tstat_col], errors="coerce").fillna(0)
    long_tstat = pd.to_numeric(df[long_tstat_col], errors="coerce").fillna(0)
    long_sma = pd.to_numeric(df[long_sma_col], errors="coerce").fillna(0)
    
    # INITIALISATION DES VARIABLES DE CONTRÔLE
    # signals: Série qui contiendra les signaux de trading (1=achat, -1=vente, 0=neutre)
    signals = pd.Series(0, index=df.index, dtype=int)
    
    # in_position: Booléen indiquant si nous sommes actuellement en position (True) ou non (False)
    # Garantit l'alternance des signaux (pas d'achat multiple sans vente entre)
    in_position = False
    
    # position_start: Index de la barre où nous sommes entrés en position
    # Utilisé pour calculer la durée de détention et appliquer la limite de temps
    position_start = None
    
    # prev_short: Valeur du t-stat court à la période précédente
    # Nécessaire pour détecter les franchissements de seuils (comparaison avant/après)
    prev_short = None
    
    # BOUCLE PRINCIPALE - Parcours de chaque barre de données
    for i, (idx, short_val, long_val, sma_val) in enumerate(zip(df.index, short_tstat, long_tstat, long_sma)):
        # signal: Signal pour cette barre spécifique (0 par défaut)
        signal = 0
        
        # On ne peut détecter un franchissement qu'à partir de la 2ème barre
        # (besoin de comparer avec la valeur précédente)
        if prev_short is not None:
            
            # ========== CONDITIONS D'ACHAT (ENTRÉE EN POSITION) ==========
            # On ne peut acheter que si on n'est pas déjà en position
            if not in_position:
                # CONDITION 1: Franchissement haussier du t-stat court
                # Le t-stat court était <= au seuil haut et passe maintenant au-dessus
                # Indique un momentum court terme devenant fortement positif
                cross_up = (prev_short <= short_up_level) and (short_val > short_up_level)
                
                # CONDITION 2: Le t-stat long terme est suffisamment élevé
                # Filtre les périodes de tendance baissière forte
                # Si le t-stat long est trop négatif, on évite d'acheter même si le court terme est positif
                long_valid = long_val >= long_min_threshold
                
                # CONDITION 3: Le momentum long terme s'améliore
                # Le t-stat long doit être au-dessus de sa moyenne mobile
                # Cela indique que la tendance long terme est en train de s'accélérer
                momentum_good = long_val >= sma_val
                
                # DÉCISION D'ACHAT: Toutes les conditions doivent être vraies
                if cross_up and long_valid and momentum_good:
                    signal = 1  # Signal d'achat
                    in_position = True  # On marque qu'on est maintenant en position
                    position_start = i  # On enregistre l'index d'entrée pour le suivi temporel
            
            # ========== CONDITIONS DE VENTE (SORTIE DE POSITION) ==========
            # On ne peut vendre que si on est déjà en position
            else:  # Déjà en position
                
                # CONDITION DE SORTIE 1: Franchissement baissier du t-stat court
                # Le t-stat court était >= au seuil bas et passe maintenant en-dessous
                # Indique que le momentum court terme devient fortement négatif
                # Note: Pour la sortie, on n'applique PAS les filtres long terme
                # On veut sortir rapidement si le court terme se dégrade
                traditional_sell = (prev_short >= short_down_level) and (short_val < short_down_level)
                
                # CONDITION DE SORTIE 2: Limite de temps atteinte
                # Protection contre les positions qui restent ouvertes trop longtemps
                # Même si le signal court terme n'est pas négatif, on sort après N périodes
                # Cela permet de libérer du capital et éviter les retournements tardifs
                time_limit_reached = (max_holding_periods is not None and  # Limite définie
                                    position_start is not None and  # Position ouverte
                                    i - position_start >= max_holding_periods)  # Durée dépassée
                
                # DÉCISION DE VENTE: Une seule condition suffit (OR logique)
                # Approche plus agressive pour la sortie que pour l'entrée
                # Protège le capital en sortant rapidement en cas de signal négatif
                if traditional_sell or time_limit_reached:
                    signal = -1  # Signal de vente
                    in_position = False  # On n'est plus en position
                    position_start = None  # Reset de l'index d'entrée
        
        # Enregistrement du signal pour cette barre
        signals.iloc[i] = signal
        
        # Mise à jour de la valeur précédente pour la prochaine itération
        # Nécessaire pour détecter les franchissements à la barre suivante
        prev_short = short_val
    
    # Ajout de la colonne de signaux au DataFrame original
    # Permet d'avoir les signaux directement dans les données
    df[output_col] = signals
    
    # Retour de la série de signaux pour utilisation ultérieure
    return signals


def calculate_rolling_ols(df: pd.DataFrame, price_col: str = 'close', window: int = 20, suffix: str = '') -> pd.Series:
    """
    Calcule une régression linéaire mobile pour détecter les tendances.
    
    Args:
        df: DataFrame avec les données de prix
        price_col: colonne des prix à analyser (défaut: 'close')
        window: fenêtre pour la régression mobile (défaut: 20)
        suffix: suffixe pour le nom de la série retournée
        
    Returns:
        Series des t-statistics de la pente de régression
    """
    df_copy = df.copy()
    df_copy['time_index'] = range(len(df_copy))
    
    y = df_copy[price_col]
    X = sm.add_constant(df_copy['time_index'])
    
    rolling_ols = RollingOLS(y, X, window=window)
    rolling_results = rolling_ols.fit()
    
    # Extraire coefficients et t-stats
    trend_tstat = (rolling_results.tvalues.iloc[:, 1] 
                  if hasattr(rolling_results.tvalues, 'iloc') 
                  else rolling_results.tvalues['time_index'])
    
    return pd.Series(trend_tstat, index=df.index, name=f'trend_tstat{suffix}')


def calculate_tstat_sma(df: pd.DataFrame, tstat_col: str, sma_window: int = 30) -> pd.Series:
    """
    Calcule une moyenne mobile simple (SMA) sur les t-statistiques.
    Utilisé pour filtrer les périodes de faible momentum.
    
    Args:
        df: DataFrame avec la colonne de t-statistiques
        tstat_col: Nom de la colonne contenant les t-stats
        sma_window: Fenêtre pour la moyenne mobile (défaut: 30 périodes)
        
    Returns:
        Series avec les valeurs de la SMA
    """
    tstat_series = pd.to_numeric(df[tstat_col], errors='coerce')
    sma = tstat_series.rolling(window=sma_window, min_periods=sma_window).mean()
    
    return sma


def process_crypto_data(
    crypto_df: pd.DataFrame, 
    symbol: str, 
    max_holding_periods: int = None,
    short_window: int = 20,
    long_window: int = 200,
    sma_window: int = 100,
    short_up_level: float = 2.0,
    short_down_level: float = -2.0,
    long_min_threshold: float = -5.0
) -> pd.DataFrame:
    """
    Traite les données d'une cryptomonnaie: calcule les indicateurs et génère les signaux.
    Utilise une SMA sur le t-stat long terme pour filtrer les périodes de faible momentum.
    
    Args:
        crypto_df: DataFrame complet avec toutes les cryptos
        symbol: Symbole de la crypto à traiter (ex: 'BTCUSDT')
        max_holding_periods: Limite de temps maximale pour les positions (optionnel)
        short_window: Fenêtre pour la régression court terme (défaut: 20)
        long_window: Fenêtre pour la régression long terme (défaut: 200)
        sma_window: Fenêtre pour la SMA du t-stat long terme (défaut: 100)
        short_up_level: Seuil haut pour les signaux courts (défaut: 2.0)
        short_down_level: Seuil bas pour les signaux courts (défaut: -2.0)
        long_min_threshold: Seuil minimum du t-stat long pour autoriser les signaux (défaut: -5.0)
        
    Returns:
        DataFrame avec les indicateurs et signaux pour cette crypto
    """
    # Filtrer et trier les données pour ce symbole
    data = crypto_df[crypto_df['symbol'] == symbol].copy().sort_index()
    
    # Calculer les indicateurs de tendance
    data['trend_tstat_short'] = calculate_rolling_ols(data, window=short_window, suffix='_short')
    data['trend_tstat_long'] = calculate_rolling_ols(data, window=long_window, suffix='_long')
    
    # Calculer la SMA sur le t-stat long terme pour filtrer le momentum
    # Quand le t-stat long est au-dessus de sa SMA = momentum croissant
    # Quand le t-stat long est en-dessous de sa SMA = momentum décroissant
    data['trend_tstat_long_sma'] = calculate_tstat_sma(data, 'trend_tstat_long', sma_window=sma_window)
    
    # Générer les signaux avec tous les filtres appliqués en une seule passe
    data['signal'] = generate_clean_signals(
        data, 
        short_tstat_col='trend_tstat_short',
        long_tstat_col='trend_tstat_long', 
        long_sma_col='trend_tstat_long_sma',
        short_up_level=short_up_level, 
        short_down_level=short_down_level, 
        long_min_threshold=long_min_threshold,
        max_holding_periods=max_holding_periods
    )
    
    return data


def process_multiple_cryptos(
    crypto_df: pd.DataFrame, 
    crypto_list: List[str] = None, 
    max_holding_periods: int = None,
    short_window: int = 20,
    long_window: int = 200,
    sma_window: int = 100,
    short_up_level: float = 2.0,
    short_down_level: float = -2.0,
    long_min_threshold: float = -5.0
) -> dict:
    """
    Traite plusieurs cryptomonnaies en parallèle.
    
    Args:
        crypto_df: DataFrame avec toutes les données
        crypto_list: Liste des symboles à traiter (None = tous)
        max_holding_periods: Limite de temps maximale pour les positions (optionnel)
        short_window: Fenêtre pour la régression court terme (défaut: 20)
        long_window: Fenêtre pour la régression long terme (défaut: 200)
        sma_window: Fenêtre pour la SMA du t-stat long terme (défaut: 100)
        short_up_level: Seuil haut pour les signaux courts (défaut: 2.0)
        short_down_level: Seuil bas pour les signaux courts (défaut: -2.0)
        long_min_threshold: Seuil minimum du t-stat long pour autoriser les signaux (défaut: -5.0)
        
    Returns:
        Dictionnaire {symbol: DataFrame_processed}
    """
    if crypto_list is None:
        crypto_list = crypto_df['symbol'].unique()
    
    cryptos_processed = {}
    
    for symbol in crypto_list:
        print(f"Traitement de {symbol}...")
        cryptos_processed[symbol] = process_crypto_data(
            crypto_df, symbol, max_holding_periods,
            short_window=short_window,
            long_window=long_window,
            sma_window=sma_window,
            short_up_level=short_up_level,
            short_down_level=short_down_level,
            long_min_threshold=long_min_threshold
        ).dropna()
    
    time_limit_msg = f" avec limite de {max_holding_periods} périodes" if max_holding_periods else ""
    return cryptos_processed

def _plot_colored_line_segments(plot_obj, x_data, y_data, condition, color_true='green', color_false='red', 
                               line_width=2, alpha=0.9, legend_label_base='', line_dash='solid'):
    """
    Trace une ligne avec des segments colorés selon une condition.
    
    Args:
        plot_obj: Objet Bokeh plot
        x_data: Données X (dates)
        y_data: Données Y (valeurs)
        condition: Série boolean pour la condition
        color_true: Couleur quand condition=True (défaut: vert)
        color_false: Couleur quand condition=False (défaut: rouge)
        line_width: Épaisseur de la ligne
        alpha: Transparence
        legend_label_base: Base pour les labels de légende
        line_dash: Style de ligne ('solid', 'dashed', 'dotted', etc.)
    """
    # Convertir en numpy arrays pour de meilleures performances
    x_vals = x_data.values if hasattr(x_data, 'values') else x_data
    y_vals = y_data.values if hasattr(y_data, 'values') else y_data
    cond_vals = condition.values if hasattr(condition, 'values') else condition
    
    # Créer les segments
    current_condition = None
    segment_x = []
    segment_y = []
    
    for i in range(len(x_vals)):
        if pd.isna(y_vals[i]) or pd.isna(cond_vals[i]):
            continue
            
        if current_condition is None:
            current_condition = cond_vals[i]
            segment_x = [x_vals[i]]
            segment_y = [y_vals[i]]
        elif current_condition == cond_vals[i]:
            segment_x.append(x_vals[i])
            segment_y.append(y_vals[i])
        else:
            # Condition change - plot current segment
            if len(segment_x) > 1:
                color = color_true if current_condition else color_false
                plot_obj.line(segment_x, segment_y, line_width=line_width, color=color, alpha=alpha, line_dash=line_dash)
            
            # Start new segment
            current_condition = cond_vals[i]
            segment_x = [x_vals[i]]
            segment_y = [y_vals[i]]
    
    # Plot the last segment
    if len(segment_x) > 1:
        color = color_true if current_condition else color_false
        plot_obj.line(segment_x, segment_y, line_width=line_width, color=color, alpha=alpha, line_dash=line_dash)
    

def plot_crypto(cryptos_with_indicators: dict, symbol: str, start_date: str = None, end_date: str = None, 
                width: int = 1500, height: int = 800, long_min_threshold: float = -5.0):
    """
    Visualisation interactive optimisée à 2 panneaux pour une crypto avec Bokeh.
    Affiche: Prix/Signaux et T-Statistics avec coloration dynamique selon les conditions.
    
    Args:
        cryptos_with_indicators: Dict des données traitées par crypto
        symbol: Symbole à afficher
        start_date: Date de début (format 'YYYY-MM-DD')
        end_date: Date de fin (format 'YYYY-MM-DD')
        width: Largeur du graphique
        height: Hauteur totale du graphique (défaut: 800 pour 2 panneaux)
        long_min_threshold: Seuil pour colorer la ligne long t-stat (défaut: -5.0)
    """
    if symbol not in cryptos_with_indicators:
        print(f"Erreur: {symbol} non disponible")
        return
    
    df = cryptos_with_indicators[symbol].copy()
    
    # Filtrer par dates
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    
    df['date'] = pd.to_datetime(df.index)
    
    # Hauteurs des panneaux (70% - 30%)
    height_p1 = int(height * 0.7)
    height_p2 = int(height * 0.3)
    
    # PANNEAU 1: Prix et signaux
    p1 = figure(x_axis_type="datetime", width=width, height=height_p1, 
                title=f"{symbol} - Prix et Signaux", 
                tools="pan,box_zoom,wheel_zoom,zoom_in,zoom_out,reset,save",
                active_drag="box_zoom")
    p1.line(df['date'], df['close'], line_width=2, color='white', alpha=0.8)
    
    # Signaux
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    if len(buy_signals) > 0:
        p1.triangle(buy_signals['date'], buy_signals['close'], size=10, color='lime', alpha=0.8)
    if len(sell_signals) > 0:
        p1.inverted_triangle(sell_signals['date'], sell_signals['close'], size=10, color='red', alpha=0.8)
    
    # PANNEAU 2: T-Statistics combinées avec filtre de momentum
    p2 = figure(x_axis_type="datetime", width=width, height=height_p2, 
                title="T-Statistics & Filtre de Momentum", x_range=p1.x_range, 
                tools="pan,box_zoom,wheel_zoom,zoom_in,zoom_out,reset,save",
                active_drag="box_zoom")
    
    # Afficher le t-stat court terme (pas de coloration dynamique)
    p2.line(df['date'], df['trend_tstat_short'], line_width=2, color='cyan', legend_label='T-Stat Court (20p)', alpha=0.9)
    
    # T-stat long terme avec coloration dynamique selon le seuil
    long_above_threshold = df['trend_tstat_long'] >= long_min_threshold
    _plot_colored_line_segments(p2, df['date'], df['trend_tstat_long'], long_above_threshold,
                               color_true='green', color_false='red', line_width=2, alpha=0.9)
    
    # Ajouter la SMA du t-stat long terme avec coloration dynamique selon le momentum
    if 'trend_tstat_long_sma' in df.columns:
        momentum_positive = df['trend_tstat_long'] >= df['trend_tstat_long_sma']
        _plot_colored_line_segments(p2, df['date'], df['trend_tstat_long_sma'], momentum_positive,
                                   color_true='green', color_false='red', line_width=2, alpha=0.8,
                                   line_dash='dashed')
        
    
    # Lignes de référence pour les signaux
    p2.add_layout(Span(location=2, dimension='width', line_color='lime', line_dash='dashed', line_alpha=0.7))
    p2.add_layout(Span(location=-2, dimension='width', line_color='red', line_dash='dashed', line_alpha=0.7))
    p2.add_layout(Span(location=0, dimension='width', line_color='gray', line_dash='solid', line_alpha=0.5))
    p2.add_layout(Span(location=-5, dimension='width', line_color='darkred', line_dash='dotted', line_alpha=0.5))
    
    # Style sombre pour tous les panneaux
    for p in [p1, p2]:
        p.background_fill_color = '#1e1e1e'
        p.border_fill_color = '#1e1e1e'
        p.xgrid.grid_line_color = '#333'
        p.ygrid.grid_line_color = '#333'
        p.legend.location = "top_left"
        p.legend.background_fill_color = '#2a2a2a'
        p.legend.background_fill_alpha = 0.8
        p.legend.label_text_color = '#cccccc'
        p.legend.border_line_color = '#444444'
    
    show(column(p1, p2))


def quick_analysis(cryptos_with_indicators: dict, figsize: tuple = (15, 16)):
    """
    Affiche un résumé rapide de tous les actifs avec matplotlib.
    Layout amélioré: 3 graphiques par ligne pour une meilleure lisibilité.
    
    Args:
        cryptos_with_indicators: Dict des données traitées par crypto
        figsize: Tuple defining figure size (width, height)
    """
    # Nouveau layout: 4 lignes x 3 colonnes = 12 emplacements pour 10 cryptos
    fig, axes = plt.subplots(4, 3, figsize=figsize)
    axes = axes.flatten()
    
    symbols = sorted(cryptos_with_indicators.keys())
    
    for i, symbol in enumerate(symbols):
        data = cryptos_with_indicators[symbol]
        
        # Prix en gris foncé pour meilleur contraste
        axes[i].plot(data.index, data['close'], color='darkgray', alpha=0.8, linewidth=1)
        axes[i].set_title(symbol, color='white', fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Marquer les signaux avec couleurs vives
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        if len(buy_signals) > 0:
            axes[i].scatter(buy_signals.index, buy_signals['close'], color='lime', s=25, alpha=0.9, zorder=5)
        if len(sell_signals) > 0:
            axes[i].scatter(sell_signals.index, sell_signals['close'], color='red', s=25, alpha=0.9, zorder=5)
    
    # Masquer les graphiques vides (positions 10 et 11)
    for i in range(len(symbols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
