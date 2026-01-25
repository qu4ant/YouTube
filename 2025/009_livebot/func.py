import talib as ta
import pandas as pd

#########################
#  Tech. Indic. + sig.  #
#########################

def add_technical_indicator(df):
    df['UP_BBLOG'], df['SMA_BBLOG'], df['LOW_BBLOG'] = ta.BBANDS(df['close'], timeperiod=20, nbdevup=1.5, nbdevdn=1.5, matype=0)
    df['rsi'] = ta.RSI(df['close'], timeperiod=14)
    return df

def add_signals(df):
    df['signal'] = ((df['close'] < df['LOW_BBLOG']) & (df['rsi'] < 50)).astype(int)
    return df

#########################
# Gestion du risque      #
#########################

def generate_TP_SL(entry_price, SL_pct=0.01, risk_reward_ratio=2, decimal_precision=2):
    SL = round(entry_price * (1 - SL_pct), decimal_precision)  # SL basé sur le pourcentage de risque
    TP = round(entry_price * (1 + (SL_pct * risk_reward_ratio)), decimal_precision)  # TP basé sur le ratio RR
    print(f'ACHAT -> Prix d\'entrée:[{entry_price}] SL: {SL} TP: {TP}')
    return SL, TP

#########################
# Système de gestion des ordres #
#########################

def place_market_order(client, symbol, side='BUY', quantity=0):
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )

        print(f"Ordre marché placé: {order}")

    except Exception as e:
        print(f"Erreur lors du placement de l'ordre: {str(e)}")
        return None
    
    # Retourne la quantité reelement exécutée
    executed_qty = float(order['fills'][0]['qty'])
    commission = float(order['fills'][0]['commission'])
    total_qty = executed_qty - commission
    
    return round(total_qty, 3)

def place_oco_order(client, symbol, side='SELL', quantity=0, TP=0, SL=0, limit_pct=0.005):
    try:
        order = client.create_oco_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price= TP,  # Prix TP
            stopPrice= SL,  # Prix déclencheur SL
            stopLimitPrice= round((SL - (SL*limit_pct)), 2),  # Prix limite SL
            stopLimitTimeInForce='GTC'  # Bon jusqu'à annulation
        )

        print(f"Ordre OCO placé: {order}")

    except Exception as e:
        print(f"Erreur lors du placement de l'ordre OCO: {str(e)}")
        return None
    
def has_open_orders(client, symbol=None):
    if symbol:
        open_orders = client.get_open_orders(symbol=symbol)
    else:
        open_orders = client.get_open_orders()
    return len(open_orders) > 0
