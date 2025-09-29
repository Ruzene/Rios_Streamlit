import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.ticker as mtick
from pathlib import Path
import os

st.set_page_config(page_title="Otimização de Portfólio", layout="wide")

# ==== CACHE DIÁRIO DE PREÇOS (com Parquet + atualização incremental) ====
DATA_DIR = Path("data_cache")
DATA_DIR.mkdir(exist_ok=True)
PARQUET_PATH = DATA_DIR / "ibov_prices.parquet"

def _today_midnight_sp():
    # retorna Timestamp NAIVE (sem timezone), para bater com o índice do Yahoo
    return pd.Timestamp.today().normalize()

@st.cache_data(ttl=60*60*24, show_spinner=False)
def _fetch_from_yf_cached(tickers, start, end, chunk_size=20):
    """
    Baixa fechamento de vários tickers (formato Yahoo .SA), lida com:
      - DataFrame MultiIndex (1º nível = campo como 'Close', 2º = ticker)
      - Fallback para 'Adj Close' se 'Close' não existir
      - Ticker único
      - Colunas 100% NaN (remove)
      - Índice com/sem timezone (normaliza p/ naïve)
      - Baixa em lotes para evitar falhas silenciosas
    Retorna DF wide: datas x [TICKER_RAIZ]
    """
    tickers = [str(t).upper().replace(".SA", "").strip() for t in tickers]
    all_chunks = []

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i:i+chunk_size]
        batch_sa = [f"{t}.SA" for t in batch]

        df_raw = yf.download(
            batch_sa,
            start=start, end=end,
            progress=False, auto_adjust=False, threads=True
        )

        if df_raw is None or df_raw.empty:
            continue

        # Caso múltiplos tickers: MultiIndex
        if isinstance(df_raw.columns, pd.MultiIndex):
            # yfinance geralmente usa nível 0 = campo, nível 1 = ticker
            lvl0 = df_raw.columns.get_level_values(0)

            if 'Close' in set(lvl0):
                df_close = df_raw.loc[:, 'Close']
            elif 'Adj Close' in set(lvl0):
                df_close = df_raw.loc[:, 'Adj Close']
            else:
                # fallback: pega o 1º nível disponível
                first_level_value = list(dict.fromkeys(lvl0))[0]
                df_close = df_raw.loc[:, first_level_value]

            # Normaliza nomes das colunas (tickers), removendo ".SA"
            df_close.columns = [str(c).upper().replace(".SA", "").strip() for c in df_close.columns]

        else:
            # Um único ticker: colunas simples
            if 'Close' in df_raw.columns:
                s = df_raw['Close']
            elif 'Adj Close' in df_raw.columns:
                s = df_raw['Adj Close']
            else:
                cols_lower = {c.lower(): c for c in df_raw.columns}
                key = cols_lower.get('close') or cols_lower.get('adj close')
                if not key:
                    continue
                s = df_raw[key]

            name = batch[0] if len(batch) == 1 else 'TICKER'
            df_close = s.to_frame(name=name.upper().strip())

        # Índice -> DatetimeIndex naïve
        if not isinstance(df_close.index, pd.DatetimeIndex):
            df_close.index = pd.to_datetime(df_close.index)
        if df_close.index.tz is not None:
            df_close.index = df_close.index.tz_convert(None)

        # Remove colunas 100% NaN
        df_close = df_close.dropna(how='all', axis=1)

        if not df_close.empty:
            all_chunks.append(df_close)

    if not all_chunks:
        return pd.DataFrame()

    # Concatena por colunas, alinhando datas, e remove duplicatas
    df = pd.concat(all_chunks, axis=1).sort_index()
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def _load_parquet_if_exists():
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    return None

def _save_parquet(df):
    df.sort_index().to_parquet(PARQUET_PATH)

def get_prices(tickers, meses):
    if not tickers:
        return pd.DataFrame()

    # Sanitiza entradas (sem .SA)
    tickers = [str(t).upper().replace(".SA", "").strip() for t in tickers]
    tickers = list(dict.fromkeys(tickers))
    end = _today_midnight_sp()
    start = (end - pd.DateOffset(months=int(meses) + 1)).normalize()

    df_hist = _load_parquet_if_exists()

    if df_hist is None:
        # 1ª vez: baixa tudo
        df_all = _fetch_from_yf_cached(tickers, start, end)
        if df_all.empty:
            return pd.DataFrame(columns=tickers)
        _save_parquet(df_all)
        # Retorna só os tickers pedidos (os que vieram)
        have = [t for t in tickers if t in df_all.columns]
        return df_all.loc[df_all.index >= start, have].copy()

    # Garante alinhamento e ordenação
    df_hist = df_hist.sort_index()

    # Tickers faltantes
    faltantes = [t for t in tickers if t not in df_hist.columns]
    if faltantes:
        df_new = _fetch_from_yf_cached(faltantes, start, end)
        if not df_new.empty:
            for c in df_new.columns:
                if c not in df_hist.columns:
                    df_hist[c] = df_new[c]
                else:
                    df_hist[c] = df_hist[c].combine_first(df_new[c])

    # Datas novas até hoje
    last_dt = df_hist.index.max()
    if pd.isna(last_dt) or last_dt < end:
        fetch_start = (last_dt + pd.Timedelta(days=1)) if pd.notna(last_dt) else start
        df_upd = _fetch_from_yf_cached(tickers, fetch_start, end)
        if not df_upd.empty:
            for c in df_upd.columns:
                if c not in df_hist.columns:
                    df_hist[c] = df_upd[c]
                else:
                    df_hist[c] = df_hist[c].combine_first(df_upd[c])

    # Salva consolidado e recorta janela
    _save_parquet(df_hist)

    # Garante que só retornamos os tickers que realmente existem
    have = [t for t in tickers if t in df_hist.columns]
    df_out = df_hist.loc[df_hist.index >= start, have].copy()

    # Aviso amigável se algum ticker não veio
    missing = [t for t in tickers if t not in have]
    if missing:
        st.warning(f"Os seguintes tickers não retornaram dados do Yahoo e foram ignorados: {', '.join(missing)}")

    return df_out


# Criar 2 colunas: uma para imagem, outra para o título
col1, col2 = st.columns([1, 6])

#with col1:
#    st.image("C:/Users/johng/Downloads/logo_rios.jpg", width=250)

with col2:
    st.markdown(
        """
        <h1 style="padding-top: 20px;">Simulador de Portfólio - Markowitz</h1>
        """,
        unsafe_allow_html=True
    )

# ===============================
# Período e taxa livre de risco
# ===============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    risk_free_rate = st.number_input("Taxa Livre de Risco (% a.a.)", min_value=0.0, max_value=20.0, value=8.0, step=0.1)/100

with col2:
    periodo_meses = st.number_input("Período de Análise (meses - até 60)", min_value=1, max_value=60, value=12, step=1)

with col3:
    st.write("Faixa de correlação entre ativos")
    corr_min = st.number_input("Correlação mínima", min_value=-1.0, max_value=1.0, value=-1.0, step=0.01)
    corr_max = st.number_input("Correlação máxima", min_value=-1.0, max_value=1.0, value=1.0, step=0.01)

with col4:
    risco_max_user = st.number_input(
        "Risco Máximo Aceito (% a.a.)", 
        min_value=1.0, max_value=100.0, value=20.0, step=0.5
    ) / 100

# Lista de ativos de exemplo
ativos_ibov = [
    'ABEV3', 'ALPA4', 'ARZZ3', 'ASAI3', 'AZUL4', 'B3SA3', 'BBSE3',
    'BBDC3', 'BBDC4', 'BRAP4', 'BBAS3', 'BRKM5', 'BRFS3', 'BPAC11',
    'CRFB3', 'CCRO3', 'CMIG4', 'CIEL3', 'COGN3', 'CPLE6', 'CSAN3',
    'CPFE3', 'CMIN3', 'CVCB3', 'CYRE3', 'DXCO3', 'ELET3', 'ELET6',
    'EMBR3', 'ENGI11', 'ENEV3', 'EGIE3', 'EQTL3', 'EZTC3', 'FLRY3',
    'GGBR4', 'GOAU4', 'NTCO3', 'HAPV3', 'HYPE3', 'IGTI11', 'IRBR3',
    'ITSA4', 'ITUB4', 'JBSS3', 'KLBN11', 'RENT3', 'LREN3', 'LWSA3',
    'MGLU3', 'MRFG3', 'BEEF3', 'MRVE3', 'MULT3', 'PCAR3', 'PETR3',
    'PETR4', 'RECV3', 'PRIO3', 'PETZ3', 'RADL3', 'RAIZ4', 'RDOR3',
    'RAIL3', 'SBSP3', 'SANB11', 'SMTO3', 'CSNA3', 'SLCE3', 'SUZB3',
    'TAEE11', 'VIVT3', 'TIMS3', 'TOTS3', 'TRPL4', 'UGPA3', 'USIM5',
    'VALE3', 'VAMO3', 'VBBR3', 'WEGE3', 'YDUQ3']


# ===============================
# Funções
# ===============================

def portfolio_performance(weights, log_returns):
    mean_returns = log_returns.mean() * 252
    cov_matrix  = log_returns.cov()  * 252
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    return ret, vol

def negative_sharpe_ratio(weights, log_returns, risk_free_rate):
    ret, vol = portfolio_performance(weights, log_returns)
    return -(ret - risk_free_rate) / vol

def portfolio_volatility(weights, log_returns):
    return portfolio_performance(weights, log_returns)[1]

def print_portfolio(weights, name, log_returns):
    # Retorno anual de cada ativo (%)
    ativos_retorno = (log_returns.mean() * 252 * 100).round(2)

    # DataFrame com Peso (%) e Retorno (%) ponderado
    df_w = pd.DataFrame({
        'Ativo': log_returns.columns,
        'Peso (%)': (weights*100).round(2),
        'Retorno (%)': (weights * ativos_retorno).round(2)
    })

    # Retorno total do portfólio
    port_ret = (weights @ (log_returns.mean() * 252)) * 100  # % anual
    ret_total_row = pd.DataFrame({
        'Ativo': ['Retorno do Portfólio'],
        'Peso (%)': [''],
        'Retorno (%)': [round(port_ret, 2)]
    })

    df_final = pd.concat([df_w, ret_total_row], ignore_index=True)

    ret, vol = portfolio_performance(weights, log_returns)
    sharpe = (ret - risk_free_rate)/vol

    st.write(f"### {name}")
    st.dataframe(df_final)
    st.write(f"Volatilidade Anual: {vol:.2%} | Sharpe Ratio: {sharpe:.2f}")

def nonzero_df(w, ativos, mean_returns, tol=1e-6):
    w = np.asarray(w)
    mask = np.abs(w) > tol
    if not mask.any():
        return pd.DataFrame({'Ativo': [], 'Peso (%)': [], 'Contrib. Retorno (%)': []})
    ativos_arr = np.asarray(ativos)[mask]
    w_sel = w[mask]
    mu_sel = mean_returns.values[mask]
    df = pd.DataFrame({
        'Ativo': ativos_arr,
        'Peso (%)': (w_sel*100).round(2),
        'Contrib. Retorno (%)': (w_sel*mu_sel*100).round(2)
    })
    return df.sort_values('Peso (%)', ascending=False).reset_index(drop=True)

def risk_parity_weights(cov_matrix, bounds, x0):
    """
    Minimiza a diferença entre as contribuições de risco (RC_i) e a média (var/N),
    sob sum(w)=1 e bounds informados (respeita K-limit, min/max e short se permitido).
    """
    n = cov_matrix.shape[0]
    C = cov_matrix.values

    def obj(w):
        var = float(w @ (C @ w))
        if var <= 0:
            return 1e6
        mrc = C @ w               # Marginal Risk Contribution
        rc = w * mrc              # Risk Contribution por ativo
        target = var / n
        return float(((rc - target)**2).sum())

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    res = minimize(
        obj, x0, method='SLSQP',
        bounds=bounds, constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    return res.x if res.success else None

def risk_contrib_table(w, cov_matrix, ativos, mean_returns=None, tol=1e-6):
    """
    Tabela com:
      - Peso (%)
      - RC_i (%) = contribuição de risco do ativo (w_i * (Σ w)_i / (w' Σ w))
      - Budget alvo (%) = 100/N_ativos_positivos
      - Desvio (p.p.) = RC_i(%) - Budget alvo(%)
      - (opcional) Contrib. Retorno (%)
    Mostra apenas ativos com peso > tol.
    """
    w = np.asarray(w, dtype=float)
    mask = w > tol
    if mask.sum() == 0:
        return pd.DataFrame({'Ativo': [], 'Peso (%)': [], 'RC_i (%)': [], 'Budget alvo (%)': [], 'Desvio (p.p.)': []})

    C = cov_matrix.values
    mrc = C @ w                     # marginal risk contribution
    rc = w * mrc                    # risk contribution
    total_var = float(w @ mrc)      # w' Σ w
    rc_pct = rc / total_var         # fração do risco total

    ativos_sel = np.asarray(ativos)[mask]
    w_sel = w[mask]
    rc_pct_sel = rc_pct[mask]
    N_act = int(mask.sum())
    budget = np.full(N_act, 1.0 / N_act)

    data = {
        'Ativo': ativos_sel,
        'Peso (%)': (w_sel * 100).round(2),
        'RC_i (%)': (rc_pct_sel * 100).round(2),
        'Budget alvo (%)': (budget * 100).round(2),
        'Desvio (p.p.)': ((rc_pct_sel - budget) * 100).round(2),
    }
    if mean_returns is not None:
        mu_sel = mean_returns.values[mask]
        data['Contrib. Retorno (%)'] = (w_sel * mu_sel * 100).round(2)

    df = pd.DataFrame(data)
    return df.sort_values('RC_i (%)', ascending=False).reset_index(drop=True)



# ===============================
# Escolha do tipo de otimização
# ===============================
modo = st.radio(
    "Escolha o modo de otimização:",
    ["Otimização Ibov", "Selecionar Ativos Manualmente"]
)

ativos_selecionados = []
peso_min = {}
peso_max = {}
user_weights = None  # carteira do usuário

# -------------------------------
# Modo Otimização Ibov
# -------------------------------
if modo == "Otimização Ibov":
    st.info("Todos os ativos do Ibovespa serão considerados. Defina limites globais e, se desejar, limite o nº de papéis (K).")
    ativos_selecionados = ativos_ibov.copy()
    n_ibov = len(ativos_selecionados)

    # Short opcional
    allow_short_ibov = st.checkbox("Permitir venda a descoberto (short) nos ativos", value=False)

    col1, col2, col3 = st.columns([1,1,1.2])
    with col1:
        # min = 0% quando não houver short (para permitir zerar ativo)
        min_global = st.number_input(
            "Min. alocação por papel (%)",
            min_value=(-100.0 if allow_short_ibov else 0.0),
            max_value=100.0,
            value=(0.0 if not allow_short_ibov else -5.0),
            step=0.5
        )
    with col2:
        max_global = st.number_input(
            "Max. alocação por papel (%)",
            min_value=(-100.0 if allow_short_ibov else 0.0),
            max_value=100.0,
            value=10.0,  # exemplo mais realista p/ Ibov
            step=0.5
        )
    with col3:
        use_k_limit = st.checkbox("Limitar nº de ativos (K)", value=True)
        k_limit = st.number_input(
            "Qtd. máx. de ativos no portfólio (K)",
            min_value=2, max_value=n_ibov, value=min(15, n_ibov), step=1, disabled=not use_k_limit
        )

    # Validações de viabilidade das cotas
    if min_global > max_global:
        st.warning("⚠️ O mínimo não pode ser maior que o máximo. Ajustando automaticamente.")
        min_global, max_global = max_global, min_global

    # Se não permitir short: checa soma mínima e máxima possíveis
    if not allow_short_ibov:
        soma_min = n_ibov * min_global
        soma_max = n_ibov * max_global
        if soma_min > 100.0:
            st.warning(
                f"⚠️ Com {n_ibov} ativos e mínimo {min_global:.2f}%, "
                f"a soma mínima seria {soma_min:.1f}% (>100%). Ajustei min para 0%."
            )
            min_global = 0.0
        if soma_max < 100.0:
            novo_max = max_global
            if n_ibov > 0:
                novo_max = max(max_global, 100.0 / n_ibov)
            st.warning(
                f"⚠️ Com {n_ibov} ativos e máximo {max_global:.2f}%, "
                f"a soma máxima seria {soma_max:.1f}% (<100%). Ajustei max para {novo_max:.2f}%."
            )
            max_global = novo_max

    # Preenche dicionários de limites
    for ativo in ativos_selecionados:
        peso_min[ativo] = min_global
        peso_max[ativo] = max_global

    # Guarda preferências para o bloco de otimização
    st.session_state["use_k_limit_ibov"] = use_k_limit
    st.session_state["k_limit_ibov"] = int(k_limit) if use_k_limit else None
    st.session_state["allow_short_ibov"] = allow_short_ibov

# -------------------------------
# Modo Selecionar Ativos Manualmente
# -------------------------------
elif modo == "Selecionar Ativos Manualmente":
    st.info(
        "Escolha os ativos e **informe o peso atual de cada um** (permite percentuais negativos).\n\n"
        "Atenção: A soma deve ser **100%**. Ex.: PETR4 = 120%, VALE3 = -20%."
    )

    ativos_selecionados = st.multiselect("Selecione os ativos:", ativos_ibov, key="ativos_manuais")

    # Pesos informados pelo usuário (em %)
    user_weights_manual = {}
    total_peso_manual = 0.0

    if ativos_selecionados:
        st.write("Informe o **peso atual** de cada ativo (%). Intervalo permitido: -100% a +100%.")
        cols = st.columns(3)  # só para organizar melhor na tela
        for i, ativo in enumerate(ativos_selecionados):
            with cols[i % 3]:
                w = st.number_input(
                    f"Peso {ativo} (%)",
                    min_value=-100.0, max_value=100.0, value=0.0, step=1.0,
                    key=f"peso_manual_{ativo}"
                )
            user_weights_manual[ativo] = w
            total_peso_manual += w

        st.write(f"**Soma dos pesos:** {total_peso_manual:.2f}%")

        # Guarda na sessão para usar no bloco do botão
        st.session_state["user_weights_manual"] = user_weights_manual
        st.session_state["total_peso_manual"] = total_peso_manual


# ===============================
# Botão para rodar a otimização (Ibov | Manual)
# ===============================
can_run = True
helper_msg = None

if modo == "Otimização Ibov":
    if len(ativos_ibov) < 2:
        can_run = False
        helper_msg = "Lista do Ibov insuficiente (menos de 2 ativos)."
else:  # Selecionar Ativos Manualmente
    if len(ativos_selecionados) < 2:
        can_run = False
        helper_msg = "Selecione pelo menos 2 ativos."
    else:
        total_ok = abs(st.session_state.get("total_peso_manual", 0.0) - 100.0) <= 0.01
        if not total_ok:
            can_run = False
            helper_msg = (
                f"A soma dos pesos está em {st.session_state.get('total_peso_manual', 0.0):.2f}%. "
                "Ela deve ser **100%** para liberar a otimização."
            )

if helper_msg:
    st.warning(f"⚠️ {helper_msg}")

if st.button("Rodar Otimização", disabled=not can_run):
    # -------------------------------
    # Escolha de universo (Ibov vs Manual)
    # -------------------------------
    if modo == "Otimização Ibov":
        ativos_selecionados = ativos_ibov.copy()
    # no modo Manual, ativos_selecionados já veio da UI

    # -------------------------------
    # Download de dados
    # -------------------------------
    st.info("Carregando dados...")
    try:
        dados = get_prices(ativos_selecionados, periodo_meses)

        cols_ok = list(dados.columns)
        missing = [t for t in ativos_selecionados if t not in cols_ok]
        if missing:
            st.warning("Sem dados para: " + ", ".join(missing))

        # Use apenas os que vieram
        ativos_selecionados = cols_ok.copy()
        if len(ativos_selecionados) < 2:
            st.error("Dados insuficientes para otimização (menos de 2 ativos com dados).")
            st.stop()

        log_returns = np.log(dados / dados.shift(1)).dropna()

        # Benchmark: BOVA11 (negociável)
        try:
            bova = get_prices(["BOVA11"], periodo_meses)
            bova_log = np.log(bova / bova.shift(1)).dropna()
            common_idx = log_returns.index.intersection(bova_log.index)
            bova_log = bova_log.loc[common_idx]
        except Exception as e:
            bova_log = None
            st.warning(f"Não foi possível carregar BOVA11: {e}")

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()

    # -------------------------------
    # Filtro de correlação (opcional)
    # -------------------------------
    if (corr_min != -1.0 or corr_max != 1.0):
        corr_matrix = log_returns.corr()
        ativos_permitidos = []
        for i, a1 in enumerate(ativos_selecionados):
            allow = True
            for j, a2 in enumerate(ativos_selecionados):
                if i != j and not (corr_min <= corr_matrix.loc[a1, a2] <= corr_max):
                    allow = False
                    break
            if allow:
                ativos_permitidos.append(a1)
        if len(ativos_permitidos) < 2:
            st.warning("Após o filtro de correlação, restaram menos de 2 ativos. Ignorando filtro.")
            ativos_permitidos = ativos_selecionados.copy()
        ativos_selecionados = ativos_permitidos
        log_returns = log_returns[ativos_selecionados]

    # -------------------------------
    # Estatísticas anualizadas
    # -------------------------------
    mean_returns = log_returns.mean() * 252
    cov_matrix   = log_returns.cov()  * 252
    num_assets   = len(ativos_selecionados)

    # -------------------------------
    # Bounds e restrições por modo
    # -------------------------------
    if modo == "Otimização Ibov":
        bounds = tuple((peso_min[a]/100, peso_max[a]/100) for a in ativos_selecionados)
    else:  # Selecionar Ativos Manualmente
        # permite short amplo por padrão; ajuste se quiser travar
        bounds = tuple((-1.0, 1.0) for _ in ativos_selecionados)

    active_bounds = bounds  # guarda os bounds em vigor
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    x0 = np.full(num_assets, 1/num_assets)

    # -------------------------------
    # Funções auxiliares (fecham sobre mean_returns/cov_matrix)
    # -------------------------------
    def portfolio_performance(w):
        r = float(w @ mean_returns.values)
        v = float(np.sqrt(w @ (cov_matrix.values @ w)))
        return r, v

    def neg_sharpe(w):
        r, v = portfolio_performance(w)
        return -(r - risk_free_rate) / (v if v > 0 else 1e-12)

    def vol_only(w):
        return portfolio_performance(w)[1]

    # -------------------------------
    # Otimizações principais
    # -------------------------------
    # Máx. Sharpe
    res_sharpe = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res_sharpe.success:
        st.error(f"Máx. Sharpe falhou: {res_sharpe.message}")
        st.stop()
    weights_sharpe = res_sharpe.x
    ret_sharpe, vol_sharpe = portfolio_performance(weights_sharpe)
    sharpe_value = (ret_sharpe - risk_free_rate) / (vol_sharpe if vol_sharpe > 0 else np.nan)

    # Mínima Variância
    res_minvar = minimize(vol_only, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res_minvar.success:
        st.error(f"Mínima Variância falhou: {res_minvar.message}")
        st.stop()
    weights_minvar = res_minvar.x
    ret_minvar, vol_minvar = portfolio_performance(weights_minvar)

    # Equal-Weighted
    weights_equal = np.full(num_assets, 1/num_assets)
    ret_equal, vol_equal = portfolio_performance(weights_equal)


    # Máx. Retorno com risco ≤ risco_max_user (constraint correta)
    def port_ret(w): 
        return float(w @ mean_returns.values)
    
    def port_vol(w):
        return float(np.sqrt(w @ (cov_matrix.values @ w)))
    
    constraints_risk = [
        {'type': 'eq',  'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq','fun': lambda w: risco_max_user - port_vol(w)}  # vol ≤ risco_max_user
    ]
    
    res_maxret = minimize(lambda w: -port_ret(w), x0, method='SLSQP',
                          bounds=active_bounds, constraints=constraints_risk)
    weights_maxret = res_maxret.x if res_maxret.success else None
    if weights_maxret is not None:
        ret_maxret, vol_maxret = portfolio_performance(weights_maxret)


    # Limite K (apenas para Ibov)
    if modo == "Otimização Ibov" and st.session_state.get("use_k_limit_ibov", False):
        k = st.session_state.get("k_limit_ibov", None)
        if k is not None and 2 <= k < num_assets:
            idx_keep = np.argsort(-np.abs(weights_sharpe))[:k]
            bounds_k = []
            for i, a in enumerate(ativos_selecionados):
                if i in idx_keep:
                    bounds_k.append((peso_min[a]/100, peso_max[a]/100))
                else:
                    bounds_k.append((0.0, 0.0))
            bounds_k = tuple(bounds_k)
            active_bounds = bounds_k  # atualiza bounds ativos

            res_sharpe_k = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds_k, constraints=constraints)
            if res_sharpe_k.success:
                weights_sharpe = res_sharpe_k.x
                ret_sharpe, vol_sharpe = portfolio_performance(weights_sharpe)
                sharpe_value = (ret_sharpe - risk_free_rate) / (vol_sharpe if vol_sharpe > 0 else np.nan)

            res_minvar_k = minimize(vol_only, x0, method='SLSQP', bounds=bounds_k, constraints=constraints)
            if res_minvar_k.success:
                weights_minvar = res_minvar_k.x
                ret_minvar, vol_minvar = portfolio_performance(weights_minvar)

            # EW entre os K escolhidos
            weights_equal = np.zeros(num_assets)
            weights_equal[idx_keep] = 1.0 / k
            ret_equal, vol_equal = portfolio_performance(weights_equal)

            # Recalcula Máx. Retorno com restrição de risco (≤ risco_max_user), agora com K
            def _port_ret(w):
                return float(w @ mean_returns.values)
            
            def _port_vol(w):
                return float(np.sqrt(w @ (cov_matrix.values @ w)))
            
            constraints_risk_k = [
                {'type': 'eq',  'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq','fun': lambda w: risco_max_user - _port_vol(w)},  # vol ≤ risco_max_user
            ]
            
            res_maxret_k = minimize(lambda w: -_port_ret(w), x0, method='SLSQP',
                                    bounds=bounds_k, constraints=constraints_risk_k)
            
            weights_maxret = res_maxret_k.x if res_maxret_k.success else None
            if weights_maxret is not None:
                ret_maxret, vol_maxret = portfolio_performance(weights_maxret)


    # -------------------------------
    # Risk Parity (sempre long-only; respeita K-limit via bounds = 0)
    # -------------------------------
    # Clipa os bounds ativos para [0, hi] (long-only) e mantém zero onde K-limit travou
    rp_bounds = tuple((max(0.0, lo), max(0.0, hi)) for (lo, hi) in active_bounds)
    
    # x0 factível para RP: zera onde ub==0 e normaliza
    mask_pos = np.array([ub > 0 for (lo, ub) in rp_bounds], dtype=bool)
    x0_rp = np.clip(x0, 0.0, 1.0) * mask_pos
    if x0_rp.sum() == 0 and mask_pos.any():
        x0_rp = np.full(num_assets, 1.0 / mask_pos.sum()) * mask_pos
    elif x0_rp.sum() > 0:
        x0_rp = x0_rp / x0_rp.sum()
    
    weights_rp = risk_parity_weights(cov_matrix, rp_bounds, x0_rp) if mask_pos.any() else None
    if weights_rp is not None:
        ret_rp, vol_rp = portfolio_performance(weights_rp)
    else:
        ret_rp = vol_rp = np.nan

    # -------------------------------
    # Nuvem + Fronteira + CML (usando active_bounds)
    # -------------------------------
    lb = np.array([lo for (lo, hi) in active_bounds], dtype=float)
    ub = np.array([hi for (lo, hi) in active_bounds], dtype=float)
    
    def sample_feasible(lb, ub):
        """
        Amostra um w viável sob lb ≤ w ≤ ub e sum(w)=1.
        Não-uniforme (adequado p/ visual), mas respeita bounds e K-limit.
        """
        cap = ub - lb
        rem = 1.0 - lb.sum()
        if rem < 0:  # inviável (não deveria acontecer após validações)
            return lb / max(lb.sum(), 1e-12)
        if rem <= 1e-12:
            return lb.copy()
        r = np.random.random(len(lb)) * cap
        s = r.sum()
        if s <= 1e-12:
            return lb.copy()
        return lb + r * (rem / s)
    
    num_portfolios = 600
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        w = sample_feasible(lb, ub)
        r, v = portfolio_performance(w)
        results[0, i] = v
        results[1, i] = r
        results[2, i] = (r - risk_free_rate) / (v if v > 0 else 1e-12)
    
    # Faixa de retornos factível pelos MESMOS bounds
    res_rmax = minimize(lambda w: -float(w @ mean_returns.values), x0,
                        method='SLSQP', bounds=active_bounds,
                        constraints={'type':'eq','fun': lambda w: np.sum(w)-1})
    res_rmin = minimize(lambda w:  float(w @ mean_returns.values), x0,
                        method='SLSQP', bounds=active_bounds,
                        constraints={'type':'eq','fun': lambda w: np.sum(w)-1})
    
    ret_min = float(res_rmin.fun) if res_rmin.success else results[1,:].min()
    ret_max = float(-res_rmax.fun) if res_rmax.success else results[1,:].max()
    
    returns_range = np.linspace(ret_min, ret_max, 120)
    frontier_vol, frontier_ret = [], []
    for target_ret in returns_range:
        cons_frontier = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: float(x @ mean_returns.values) - target_ret}
        )
        res_f = minimize(lambda w: np.sqrt(w @ (cov_matrix.values @ w)),
                         x0, method='SLSQP', bounds=active_bounds, constraints=cons_frontier)
        if res_f.success:
            frontier_vol.append(res_f.fun)
            frontier_ret.append(target_ret)


    sigma_vals = np.linspace(0, max(results[0, :]) * 1.1, 100)
    cml_vals = risk_free_rate + (sharpe_value if not np.isnan(sharpe_value) else 0.0) * sigma_vals

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(results[0, :]*100, results[1, :]*100, c=results[2, :],
                    cmap='viridis', s=10, alpha=0.3)
    fig.colorbar(sc, ax=ax, label='Sharpe Ratio')

    if frontier_vol:
        ax.plot(np.array(frontier_vol)*100, np.array(frontier_ret)*100, 'b--', label='Fronteira Eficiente')

    if not np.isnan(sharpe_value):
        ax.plot(sigma_vals*100, cml_vals*100, 'r--', label='CML')

    # Pontos especiais
    ax.scatter(vol_sharpe*100, ret_sharpe*100, c='r', s=80, label='Máx. Sharpe', zorder=3)
    ax.scatter(vol_minvar*100, ret_minvar*100, c='orange', s=80, label='Mín. Variância', zorder=3)
    ax.scatter(vol_equal*100,  ret_equal*100,  c='g', s=80, label='Equal-Weighted', zorder=3)
    if weights_maxret is not None:
        ax.scatter(vol_maxret*100, ret_maxret*100, c='purple', s=80,
                   label=f'Máx. Retorno (Risco ≤ {risco_max_user*100:.1f}%)', zorder=3)

    # Ponto do Risk Parity (Long-only)
    if weights_rp is not None and not np.isnan(vol_rp):
        ax.scatter(vol_rp*100, ret_rp*100, c='magenta', s=90, marker='s', label='Risk Parity (Long-only)', zorder=3)

    ax.set_xlabel("Volatilidade Anual (%)")
    ax.set_ylabel("Retorno Esperado Anual (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)

    # -------------------------------
    # Tabelas (em ABAS, largura total) + séries acumuladas
    # -------------------------------
    
    def show_port_table(title, w, use_rc_table=False):
        st.subheader(title)
        if w is None:
            st.info("Não encontrado/viável com as restrições atuais.")
            return
    
        if use_rc_table:
            # Tabela específica do Risk Parity com RC_i%, budget e desvio
            df = risk_contrib_table(w, cov_matrix, ativos_selecionados, mean_returns)
            # formatação amigável (sem índice)
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True
            )
        else:
            # Tabela padrão: apenas ativos com peso > 0
            df = nonzero_df(w, ativos_selecionados, mean_returns)
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True
            )
    
        # métricas embaixo da tabela
        r, v = portfolio_performance(w)
        sh = (r - risk_free_rate)/(v if v>0 else np.nan)
        m1, m2 = st.columns(2)
        m1.metric("Volatilidade Anual", f"{v:.2%}")
        m2.metric("Sharpe", f"{sh:.2f}")
    
    
    tabs = st.tabs([
        "Máx. Sharpe",
        "Mín. Variância",
        "Equal-Weighted",
        f"Máx. Retorno (≤ {risco_max_user:.0%})",
        "Risk Parity (Long-only)"
    ])
    
    with tabs[0]:
        show_port_table("Portfólio Máx. Sharpe", weights_sharpe)
    
    with tabs[1]:
        show_port_table("Portfólio Mínima Variância", weights_minvar)
    
    with tabs[2]:
        show_port_table("Portfólio Equal-Weighted", weights_equal)
    
    with tabs[3]:
        show_port_table(f"Máx. Retorno (Risco ≤ {risco_max_user:.0%})", weights_maxret)
    
    with tabs[4]:
        show_port_table("Risk Parity (Long-only)", weights_rp, use_rc_table=True)
    
    
    # -------------------------------
    # Séries acumuladas (mantém o BOVA11)
    # -------------------------------
    ret_sharpe_daily = (log_returns @ weights_sharpe)
    ret_minvar_daily = (log_returns @ weights_minvar)
    ret_equal_daily  = (log_returns @ weights_equal)
    ret_maxrisk_daily = (log_returns @ weights_maxret) if weights_maxret is not None else None
    rp_daily = (log_returns @ weights_rp) if weights_rp is not None else None
    
    sharpe_acum = (1 + ret_sharpe_daily).cumprod()
    minvar_acum = (1 + ret_minvar_daily).cumprod()
    equal_acum  = (1 + ret_equal_daily).cumprod()
    maxrisk_acum = (1 + ret_maxrisk_daily).cumprod() if ret_maxrisk_daily is not None else None
    rp_acum = (1 + rp_daily).cumprod() if rp_daily is not None else None
    
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(sharpe_acum.index, sharpe_acum, label="Máx. Sharpe")
    ax2.plot(minvar_acum.index, minvar_acum, label="Mínima Variância")
    ax2.plot(equal_acum.index, equal_acum, label="Equal-Weighted")
    if maxrisk_acum is not None:
        ax2.plot(maxrisk_acum.index, maxrisk_acum, label="Máx. Retorno (≤ Risco)")
    if rp_acum is not None:
        ax2.plot(rp_acum.index, rp_acum, label="Risk Parity (Long-only)")
    if bova_log is not None and "BOVA11" in bova_log.columns:
        bova_acum = (1 + bova_log["BOVA11"]).cumprod()
        ax2.plot(bova_acum.index, bova_acum, label="BOVA11", linestyle='--')
    
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{(y-1)*100:.0f}%"))
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Retorno Acumulado (%)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig2)
