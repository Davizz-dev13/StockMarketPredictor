import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# --- PARÁMETROS ORIGINALES ---
RISK_FREE = 0.03
MC_PATHS = 1000000
MC_STEPS = 252
ML_FORWARD_DAYS = 5
WEIGHTS = {'mc': 0.4, 'mom': 0.25, 'val': 0.15, 'ml': 0.2}
DATA_YEARS = 5

# --- BLOQUE DE FUNCIONES ORIGINALES ---
def descargar_datos_una_vez(ticker, period_years=DATA_YEARS, auto_adjust=False):
    period = f"{period_years}y"
    return yf.download(ticker, period=period, interval='1d', auto_adjust=auto_adjust, progress=False)

def normalizar_columnas(data):
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data.columns = data.columns.droplevel(1)
        except Exception:
            pass
    if 'Adj Close' not in data.columns:
        if 'Close' in data.columns:
            data['Adj Close'] = data['Close']
    return data

def descargar_datos(ticker, period_years=DATA_YEARS):
    data = descargar_datos_una_vez(ticker, period_years, auto_adjust=False)
    if data is None or data.empty:
        data = descargar_datos_una_vez(ticker, period_years, auto_adjust=True)
    if data is None or data.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker} con yfinance.")
    data = normalizar_columnas(data)
    required = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    presentes = [c for c in required if c in data.columns]
    if not presentes:
        raise ValueError(f"No hay columnas válidas en los datos de {ticker}.")
    data = data[presentes].dropna()
    return data

def obtener_info_ticker(ticker):
    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    return info

def calcular_volatilidad_anual(adj_close):
    returns = np.log(adj_close / adj_close.shift(1)).dropna()
    if returns.empty:
        return 0.3, returns
    vol_daily = returns.std()
    vol_annual = vol_daily * np.sqrt(252)
    return float(vol_annual), returns

def indicador_momentum(adj_close):
    ret20 = adj_close.pct_change(20).iloc[-1] if len(adj_close) > 20 else adj_close.pct_change().iloc[-1]
    ret60 = adj_close.pct_change(60).iloc[-1] if len(adj_close) > 60 else ret20
    rolling_ret20 = adj_close.pct_change(20).dropna()
    if len(rolling_ret20) > 10:
        z = (rolling_ret20.iloc[-1] - rolling_ret20.mean()) / rolling_ret20.std()
    else:
        z = 0.0
    mom_score = np.tanh((ret20 + 0.5 * ret60) * 10 + z)
    return float(mom_score), float(ret20), float(ret60)

def puntuacion_valoracion(info):
    pe = info.get('trailingPE') or info.get('forwardPE') or None
    pb = info.get('priceToBook') or None
    if pe is None and pb is None:
        return 0.0, {'pe': pe, 'pb': pb}
    score_pe = 0.0
    score_pb = 0.0
    if pe is not None and isinstance(pe, (int, float)) and pe > 0:
        score_pe = -np.tanh(np.log1p(pe) / 3.0)
    if pb is not None and isinstance(pb, (int, float)) and pb > 0:
        score_pb = -np.tanh(np.log1p(pb) / 2.0)
    if pe is not None and pb is not None:
        val_score = 0.6 * score_pe + 0.4 * score_pb
    else:
        val_score = score_pe if pe is not None else score_pb
    return float(val_score), {'pe': pe, 'pb': pb}

def entrenamiento_ml_features(adj_close, vol_daily, forward_days=ML_FORWARD_DAYS):
    prices = adj_close.copy()
    df = pd.DataFrame({'price': prices})
    df['ret_1'] = prices.pct_change(1)
    df['ret_5'] = prices.pct_change(5)
    df['ret_10'] = prices.pct_change(10)
    df['sma_10'] = prices.rolling(10).mean()
    df['sma_50'] = prices.rolling(50).mean()
    df['sma_diff'] = (df['sma_10'] - df['sma_50']) / df['sma_50']
    df['vol_20'] = df['price'].pct_change().rolling(20).std()
    df = df.dropna().copy()
    df['target'] = np.log(prices.shift(-forward_days) / prices)
    df = df.dropna()
    if df.empty or len(df) < 30:
        return None, None, None
    X = df[['ret_1', 'ret_5', 'ret_10', 'sma_diff', 'vol_20']].values
    y = df['target'].values
    if SKLEARN_AVAILABLE:
        model = LinearRegression()
        model.fit(X, y)
        coef = model.coef_
        intercept = model.intercept_
        predict_fun = lambda x: model.predict(x.reshape(1, -1))[0]
    else:
        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        intercept = beta[0]
        coef = beta[1:]
        def predict_fun(x):
            xd = np.hstack(([1.0], x))
            return float(np.dot(xd, np.hstack(([intercept], coef))))
    return predict_fun, coef, intercept

def combinar_expectativas(mc_mean_return, mom_score, val_score, ml_pred):
    if ml_pred is None:
        ml_ann = 0.0
    else:
        ml_ann = ml_pred * (252.0 / ML_FORWARD_DAYS)
    mom_component = np.tanh(mom_score) * 0.15
    val_component = np.tanh(val_score) * 0.10
    ml_component = np.tanh(ml_ann) * 0.12
    combined = (WEIGHTS['mc'] * mc_mean_return
                + WEIGHTS['mom'] * mom_component
                + WEIGHTS['val'] * val_component
                + WEIGHTS['ml'] * ml_component)
    return float(combined), {'mom_comp': mom_component, 'val_comp': val_component, 'ml_ann': ml_ann}

def monte_carlo_gbm(S0, mu, sigma, T=1.0, N=MC_STEPS, n_paths=MC_PATHS, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    Z = np.random.randn(n_paths, N)
    S = np.zeros((n_paths, N + 1))
    S[:, 0] = S0
    for t in range(1, N + 1):
        exp_term = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        S[:, t] = S[:, t - 1] * np.exp(np.clip(exp_term, -0.5, 0.5))
    return S

def calcular_sistema_señales(prob_exito, retorno_esperado, vol, ml_pred, mom_score):
    score = (prob_exito - 40) * 1.5 
    score = np.clip(score, 0, 45)
    score_retorno = np.clip(retorno_esperado * 1.3, -10, 40)
    score += score_retorno
    if vol > 0.60:
        score -= (vol - 0.60) * 30
    if ml_pred is not None and ml_pred > 0 and mom_score > 0:
        score += 15
    score = np.clip(score, 0, 100)
    if score >= 85: return score, "🚀 COMPRA FUERTE", "\033[1;32m" 
    elif score >= 70: return score, "✅ COMPRA", "\033[0;32m"       
    elif score >= 50: return score, "⚖️ NEUTRAL", "\033[0;33m" 
    elif score >= 30: return score, "⚠️ EVITAR", "\033[0;31m"    
    else: return score, "❌ VENTA", "\033[1;31m"

def main():
    ticker = input("Introduce la abreviatura del stock (ej. ASTS): ").strip().upper()
    try:
        df = descargar_datos(ticker)
    except Exception as e:
        print("Error al descargar datos:", e)
        sys.exit(1)

    info = obtener_info_ticker(ticker)
    adj = df['Adj Close']
    S0 = float(adj.iloc[-1])
    vol_ann, returns_daily = calcular_volatilidad_anual(adj)
    vol_ann = float(vol_ann) if not np.isnan(vol_ann) else 0.3

    hist_ann_return = float((adj.pct_change().dropna().mean()) * 252.0)
    mc_mean_return = hist_ann_return
    mom_score, ret20, ret60 = indicador_momentum(adj)
    val_score, val_raw = puntuacion_valoracion(info)

    # FIX: Se añade vol_ann como argumento requerido
    predict_fun, coef, intercept = entrenamiento_ml_features(adj, vol_ann)
    
    ml_pred = None
    if predict_fun is not None:
        latest_ret1 = adj.pct_change(1).iloc[-1]
        latest_ret5 = adj.pct_change(5).iloc[-1] if len(adj) > 5 else latest_ret1
        latest_ret10 = adj.pct_change(10).iloc[-1] if len(adj) > 10 else latest_ret1
        sma_10 = adj.rolling(10).mean().iloc[-1]
        sma_50 = adj.rolling(50).mean().iloc[-1] if len(adj) > 50 else sma_10
        sma_diff = (sma_10 - sma_50) / sma_50 if sma_50 != 0 else 0.0
        vol20 = adj.pct_change().rolling(20).std().iloc[-1]
        x_latest = np.array([latest_ret1, latest_ret5, latest_ret10, sma_diff, vol20], dtype=float)
        ml_pred = float(predict_fun(x_latest))

    combined_expected_return, comps = combinar_expectativas(mc_mean_return, mom_score, val_score, ml_pred)
    mu_for_mc = RISK_FREE + combined_expected_return
    sigma_for_mc = vol_ann

    S = monte_carlo_gbm(S0, mu_for_mc, sigma_for_mc, T=1.0, N=MC_STEPS, n_paths=MC_PATHS)
    final_prices = S[:, -1]
    
    precio_objetivo_mediano = np.median(final_prices)
    conf_inf, conf_sup = np.percentile(final_prices, [15, 85])
    prob_exito = np.mean(final_prices > S0) * 100
    retorno_est_pct = ((precio_objetivo_mediano - S0) / S0) * 100
    perdida_base_p15 = abs((conf_inf / S0 - 1) * 100)

    # --- CONFIGURACIÓN GRÁFICA SOLICITADA ---
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Gráfico 1: Trayectorias
    t_axis = np.linspace(0, 1, MC_STEPS + 1)
    for i in range(min(100, S.shape[0])):
        ax1.plot(t_axis, S[i, :], linewidth=0.7, alpha=0.06, color='cyan')
    
    ax1.axhline(conf_sup, color='lime', linestyle=':', alpha=0.6, label=f"Sup 70%: {conf_sup:.2f}$")
    ax1.axhline(conf_inf, color='red', linestyle=':', alpha=0.6, label=f"Inf 70%: {conf_inf:.2f}$")

    # Líneas verticales trimestrales
    for quarter in [0.25, 0.5, 0.75]:
        ax1.axvline(quarter, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    ax1.set_title(f"Escenarios MC para {ticker} (Rango Operativo 70%)", fontsize=13)
    ax1.set_xlabel("Horizonte Temporal (1 Año)")
    ax1.set_ylabel("Precio Proyectado ($)")
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.15, linestyle='--')

    # Gráfico 2: Histograma
    final_returns = (final_prices / S0) - 1
    ax2.hist(final_returns, bins=80, alpha=0.8, edgecolor='black', color='royalblue', range=(-1, 4))
    ax2.axvline(np.mean(final_returns), color='red', linestyle='--', linewidth=2, label=f"Media MC: {np.mean(final_returns)*100:.2f}%")
    ax2.axvline(conf_inf/S0 - 1, color='orange', linestyle=':', label="Intervalo de confianza")
    ax2.axvline(conf_sup/S0 - 1, color='orange', linestyle=':')
    
    ax2.set_title(f"Distribución retornos finales — {ticker}")
    ax2.set_xlabel("Retorno final (%)")
    ax2.set_ylabel("Frecuencia")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    # --- REPORTE FINAL ---
    nota_final, señal, color = calcular_sistema_señales(prob_exito, retorno_est_pct, vol_ann, ml_pred, mom_score)
    reset = "\033[0m"
    print(f"\n" + "═"*50)
    print(f" 📊  David Quant Stats:           {ticker}")
    print(f" " + "═"*50)
    print(f" Precio Actual:                   {S0:.2f}$")
    print(f" Objetivo Mediano:                {precio_objetivo_mediano:.2f}$ ({retorno_est_pct:+.2f}%)")
    print(f" Volatilidad Anual:               {vol_ann*100:.1f}%")
    print(f" Prob. de Éxito (>0%):            {prob_exito:.1f}%")
    print(f" Pérdida Base (P15):              {perdida_base_p15:.2f}%")
    print(f" " + "─"*50)
    print(f" PUNTUACIÓN:                      {color}{nota_final:.1f} / 100{reset}")
    print(f" SEÑAL TÉCNICA:                   {color}{señal}{reset}")
    print(f" " + "─"*50)
    print(f" Intervalo de confianza(70%):    [{conf_inf:.2f}$ - {conf_sup:.2f}$]")
    print(f" ML Pred (5d):                    {((ml_pred*100 if ml_pred else 0)):+.2f}%")
    print(f" Momentum Score:                  {mom_score:+.2f}")
    print("═"*50 + "\n")

if __name__ == "__main__":
    main()