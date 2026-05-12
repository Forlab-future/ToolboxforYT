import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
import io
import os
import zipfile
from datetime import datetime, date as _date
from scipy.optimize import (
    least_squares, differential_evolution,
    minimize, dual_annealing, shgo,
)

st.set_page_config(page_title="Yoon Team 전용 전기화학 데이터 정리", layout="wide")
st.title("📊 Yoon Team 전용 전기화학 데이터 정리")

today = datetime.today().strftime("%Y%m%d")

current_col = "Current Density (A/cm²)"
voltage_col = "Voltage (V)"

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# ══════════════════════════════════════════════════════════════════════════════
# EIS 피팅 탭 함수
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 숫자 포맷 (소수점 5자리 미만이면 전체 표시, 이상이면 과학적 표기)
# ══════════════════════════════════════════════════════════════════════════════
def fmt_val(v):
    """5자리 이하 소수점은 그대로, 이상은 과학적 표기"""
    if v == 0:
        return "0"
    abs_v = abs(v)
    if abs_v >= 1e-4:
        # 소수점 몇 자리 필요한지 확인
        formatted = f"{v:.10f}".rstrip("0").rstrip(".")
        return formatted
    else:
        return f"{v:.6e}"

# ══════════════════════════════════════════════════════════════════════════════
# 파서
# ══════════════════════════════════════════════════════════════════════════════
def parse_z_file_fit(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    data_start = None
    for idx, line in enumerate(content):
        if "End Comments" in line:
            data_start = idx + 1
            break
    if data_start is None:
        return None
    rows = []
    for line in content[data_start:]:
        parts = re.split(r'\s+', line.strip())
        if len(parts) >= 6:
            try:
                rows.append((float(parts[0]), float(parts[4]), float(parts[5])))
            except ValueError:
                continue
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["Freq", "Zr", "Zi"])


# ══════════════════════════════════════════════════════════════════════════════
# 회로 모델: L — Rs — (R1‖CPE1) — ... — (Rn‖CPEn)
# ══════════════════════════════════════════════════════════════════════════════
def z_cpe(omega, Q, n):
    return 1.0 / (Q * (1j * omega) ** n)

def z_parallel_cpe(R, Q, n, omega):
    zc = z_cpe(omega, Q, n)
    return (R * zc) / (R + zc)

def circuit_impedance(freq, params, num_rc):
    omega = 2 * np.pi * np.array(freq, dtype=float)
    Z = 1j * omega * params[0] + params[1]
    for i in range(num_rc):
        b = 2 + i * 3
        Z += z_parallel_cpe(params[b], params[b+1], params[b+2], omega)
    return Z

def residuals_fn(params, freq, zr, zi, num_rc):
    try:
        Z = circuit_impedance(freq, params, num_rc)
        w = 1.0 / (zr**2 + zi**2 + 1e-12)
        return np.concatenate([(Z.real - zr)*np.sqrt(w), (Z.imag - zi)*np.sqrt(w)])
    except Exception:
        return np.ones(2 * len(freq)) * 1e10

def chi2_fn(params, freq, zr, zi, num_rc):
    return float(np.sum(residuals_fn(params, freq, zr, zi, num_rc)**2))


# ══════════════════════════════════════════════════════════════════════════════
# 알고리즘 정의
# ══════════════════════════════════════════════════════════════════════════════
ALGORITHMS = {
    "TRF": {
        "label": "🚀 TRF (Trust Region Reflective)",
        "desc":  "빠른 로컬 최적화. 초기값이 좋을 때 효과적. 가장 일반적인 선택.",
        "scope": "local",
    },
    "LM": {
        "label": "⚡ Levenberg-Marquardt",
        "desc":  "경계 조건 없는 빠른 로컬 최적화. 잡음이 적은 데이터에 강함.",
        "scope": "local",
    },
    "Nelder-Mead": {
        "label": "🔺 Nelder-Mead (Simplex)",
        "desc":  "기울기 불필요. 노이즈에 강하나 느림. 비정형 목적함수에 유용.",
        "scope": "local",
    },
    "L-BFGS-B": {
        "label": "📐 L-BFGS-B",
        "desc":  "경계 조건 지원 준뉴턴법. 파라미터 많을 때 효율적.",
        "scope": "local",
    },
    "DE": {
        "label": "🌐 Differential Evolution",
        "desc":  "진화 알고리즘 기반 전역 최적화. 초기값 무관, 느림.",
        "scope": "global",
    },
    "DA": {
        "label": "🌡️ Dual Annealing",
        "desc":  "모의 담금질 + 로컬 탐색 결합. 깊은 전역 최솟값 탐색.",
        "scope": "global",
    },
    "SHGO": {
        "label": "🔬 SHGO (Simplicial Homology)",
        "desc":  "위상수학 기반 전역 최적화. 다봉 함수에 강함.",
        "scope": "global",
    },
    "DE+TRF": {
        "label": "🏆 DE → TRF (전역 후 정밀)",
        "desc":  "전역(DE)으로 초기값 탐색 후 TRF로 정밀 수렴. 가장 정확.",
        "scope": "hybrid",
    },
    "DA+TRF": {
        "label": "🏆 DA → 시TRF (어닐링 후 정밀)",
        "desc":  "이중 어닐링으로 초기값 탐색 후 TRF로 정밀 수렴.",
        "scope": "hybrid",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 플랏
# ══════════════════════════════════════════════════════════════════════════════
def plot_nyquist(df, Z_fit=None, xmin=0.0, xmax=0.0, ymin=0.0, ymax=0.0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Zr"], y=-df["Zi"], mode="markers", name="측정값",
        marker=dict(color="#4361ee", size=6, opacity=0.85),
    ))
    if Z_fit is not None:
        fig.add_trace(go.Scatter(
            x=Z_fit.real, y=-Z_fit.imag, mode="lines", name="피팅",
            line=dict(color="#e63946", width=2.5),
        ))
    fig.update_layout(
        title=dict(text="나이키스트 플랏", font=dict(size=13, color="#1a1a2e"), x=0.02),
        xaxis=dict(title="Z' (Ω)", range=[xmin, xmax] if xmin != xmax else None,
                   showgrid=True, gridcolor="#ebebeb",
                   zeroline=True, zerolinecolor="#333", zerolinewidth=2.5),
        yaxis=dict(title="-Z'' (Ω)", range=[ymin, ymax] if ymin != ymax else None,
                   showgrid=True, gridcolor="#ebebeb",
                   zeroline=True, zerolinecolor="#333", zerolinewidth=2.5),
        plot_bgcolor="white", paper_bgcolor="white",
        height=360, margin=dict(l=55, r=10, t=40, b=50),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=11)),
    )
    return fig

def plot_bode(df, Z_fit=None, fmin=0.1, fmax=100000.0, ymin=0.0, ymax=0.0):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Freq"], y=-df["Zi"], mode="markers", name="측정값",
        marker=dict(color="#4361ee", size=6, opacity=0.85),
    ))
    if Z_fit is not None:
        fig.add_trace(go.Scatter(
            x=df["Freq"], y=-Z_fit.imag, mode="lines", name="피팅",
            line=dict(color="#e63946", width=2.5),
        ))
    fig.update_layout(
        title=dict(text="보데 플랏", font=dict(size=13, color="#1a1a2e"), x=0.02),
        xaxis=dict(title="Frequency (Hz)", type="log",
                   range=[np.log10(max(fmin, 1e-9)), np.log10(max(fmax, 1e-9))],
                   showgrid=True, gridcolor="#ebebeb",
                   zeroline=True, zerolinecolor="#333", zerolinewidth=2.5),
        yaxis=dict(title="-Z'' (Ω)", range=[ymin, ymax] if ymin != ymax else None,
                   showgrid=True, gridcolor="#ebebeb",
                   zeroline=True, zerolinecolor="#333", zerolinewidth=2.5),
        plot_bgcolor="white", paper_bgcolor="white",
        height=360, margin=dict(l=55, r=10, t=40, b=50),
        legend=dict(x=0.99, y=0.99, xanchor="right",
                    bgcolor="rgba(255,255,255,0.85)", font=dict(size=11)),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 피팅 실행 함수
# ══════════════════════════════════════════════════════════════════════════════
def run_fitting(algo_key, p0, lo_b, hi_b, freq_arr, zr_arr, zi_arr, num_rc, progress_cb=None):
    """
    파라미터 수가 많을수록 maxiter를 줄여 무한 대기 방지.
    progress_cb: 진행률(0~1)을 받는 콜백 함수 (선택)
    """
    bounds_pairs = list(zip(lo_b, hi_b))
    n_params = len(p0)

    # 파라미터 수에 따른 maxiter 자동 조정 (많을수록 줄임)
    de_maxiter  = max(50,  300 - n_params * 15)   # 5arc=17→65, 3arc=11→135
    da_maxiter  = max(500, 2000 - n_params * 80)
    nfev_max    = max(5000, 30000 - n_params * 1000)

    def _trf(x0):
        r = least_squares(residuals_fn, x0=x0, bounds=(lo_b, hi_b),
                          args=(freq_arr, zr_arr, zi_arr, num_rc),
                          method="trf", max_nfev=nfev_max,
                          ftol=1e-10, xtol=1e-10, gtol=1e-10)
        return r.x

    def _lm(x0):
        r = least_squares(residuals_fn, x0=x0,
                          args=(freq_arr, zr_arr, zi_arr, num_rc),
                          method="lm", max_nfev=nfev_max,
                          ftol=1e-10, xtol=1e-10, gtol=1e-10)
        return r.x

    def _nelder(x0):
        r = minimize(chi2_fn, x0=x0,
                     args=(freq_arr, zr_arr, zi_arr, num_rc),
                     method="Nelder-Mead",
                     options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8})
        return r.x

    def _lbfgsb(x0):
        r = minimize(chi2_fn, x0=x0, bounds=bounds_pairs,
                     args=(freq_arr, zr_arr, zi_arr, num_rc),
                     method="L-BFGS-B",
                     options={"maxiter": 20000, "ftol": 1e-12, "gtol": 1e-8})
        return r.x

    def _de():
        iters_done = [0]
        def cb(xk, convergence=0):
            iters_done[0] += 1
            if progress_cb:
                progress_cb(min(iters_done[0] / de_maxiter, 0.95))
        r = differential_evolution(chi2_fn, bounds=bounds_pairs,
                                   args=(freq_arr, zr_arr, zi_arr, num_rc),
                                   maxiter=de_maxiter, tol=1e-8, seed=42,
                                   workers=1, polish=True, callback=cb)
        if progress_cb: progress_cb(1.0)
        return r.x

    def _da():
        iters_done = [0]
        def cb(x, f, ctx):
            iters_done[0] += 1
            if progress_cb:
                progress_cb(min(iters_done[0] / da_maxiter, 0.95))
            return False
        r = dual_annealing(chi2_fn, bounds=bounds_pairs,
                           args=(freq_arr, zr_arr, zi_arr, num_rc),
                           maxiter=da_maxiter, seed=42,
                           callback=cb,
                           minimizer_kwargs={"method": "L-BFGS-B",
                                             "bounds": bounds_pairs})
        if progress_cb: progress_cb(1.0)
        return r.x

    def _shgo():
        r = shgo(chi2_fn, bounds=bounds_pairs,
                 args=(freq_arr, zr_arr, zi_arr, num_rc),
                 n=100, iters=2,
                 minimizer_kwargs={"method": "L-BFGS-B"})
        return r.x

    if algo_key == "TRF":
        if progress_cb: progress_cb(0.5)
        res = _trf(p0)
        if progress_cb: progress_cb(1.0)
        return res
    elif algo_key == "LM":
        if progress_cb: progress_cb(0.5)
        res = _lm(p0)
        if progress_cb: progress_cb(1.0)
        return res
    elif algo_key == "Nelder-Mead":
        if progress_cb: progress_cb(0.5)
        res = _nelder(p0)
        if progress_cb: progress_cb(1.0)
        return res
    elif algo_key == "L-BFGS-B":
        if progress_cb: progress_cb(0.5)
        res = _lbfgsb(p0)
        if progress_cb: progress_cb(1.0)
        return res
    elif algo_key == "DE":
        return _de()
    elif algo_key == "DA":
        return _da()
    elif algo_key == "SHGO":
        if progress_cb: progress_cb(0.3)
        res = _shgo()
        if progress_cb: progress_cb(1.0)
        return res
    elif algo_key == "DE+TRF":
        x_de = _de()
        if progress_cb: progress_cb(0.97)
        res = _trf(x_de)
        if progress_cb: progress_cb(1.0)
        return res
    elif algo_key == "DA+TRF":
        x_da = _da()
        if progress_cb: progress_cb(0.97)
        res = _trf(x_da)
        if progress_cb: progress_cb(1.0)
        return res
    else:
        if progress_cb: progress_cb(0.5)
        res = _trf(p0)
        if progress_cb: progress_cb(1.0)
        return res


def eis_fitting_tab():
    # ══════════════════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════════════════

    uploaded_fit = st.file_uploader("📂 임피던스 파일 (.z / .txt)", type=["z", "txt"], key="eis_fit_uploader")

    if uploaded_fit is None:
        st.info("👆 .z 또는 .txt 파일을 업로드하면 시작됩니다.")
        return

    df = parse_z_file_fit(uploaded_fit)
    if df is None:
        st.error("❌ 파싱 실패. 파일 형식을 확인해 주세요.")
        return

    st.markdown(
        f'<span class="badge-ok">✅ {len(df)}개 포인트 | '
        f'{df["Freq"].min():.2g} ~ {df["Freq"].max():.2g} Hz</span>',
        unsafe_allow_html=True,
    )

    st.markdown(" ")

    col_L, col_R = st.columns([6, 4], gap="medium")

    # ── 왼쪽: 그래프 ──────────────────────────────────────────────────────────────
    with col_L:
        Z_fit_full  = st.session_state.get("fit_Z_fit_full", None)
        popt_graph  = st.session_state.get("fit_popt", None)
        num_rc_graph = (len(popt_graph) - 2) // 3 if popt_graph is not None else 0

        # 아크별 임피던스 계산
        arc_Z_list = []  # [(label, color, Z_arc_array), ...]
        if popt_graph is not None:
            freq_all = df["Freq"].values
            omega_all = 2 * np.pi * freq_all
            arc_colors = ["#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]
            Rs_val = popt_graph[1]
            # 각 아크를 독립적으로 계산하고, 나이키스트 x 오프셋은 별도 관리
            # 아크 i의 x_offset = Rs + R1 + R2 + ... + R(i-1)
            x_offset = Rs_val
            for i in range(num_rc_graph):
                base = 2 + i * 3
                R_i = popt_graph[base]
                Q_i = popt_graph[base+1]
                n_i = popt_graph[base+2]
                # 아크만 단독 계산 (0 기준)
                Z_arc_only = z_parallel_cpe(R_i, Q_i, n_i, omega_all)
                # x_offset은 나이키스트 플랏 시 실수부에 더함 (허수부는 그대로)
                arc_Z_list.append((f"아크 {i+1} (R{i+1}‖CPE{i+1})",
                                   arc_colors[i % len(arc_colors)],
                                   Z_arc_only, x_offset))
                x_offset += R_i

        # ── 축 범위 설정 (나이키스트 + 보데 공통) ────────────────────────────
        with st.expander("⚙️ 축 범위 설정", expanded=False):
            st.caption("나이키스트")
            a1, a2, a3, a4 = st.columns(4)
            ny_xmin = a1.number_input("X min", value=0.0, format="%.4f", key="fit_ny_xmin")
            ny_xmax = a2.number_input("X max", value=0.0, format="%.4f", key="fit_ny_xmax")
            ny_ymin = a3.number_input("Y min", value=0.0, format="%.4f", key="fit_ny_ymin")
            ny_ymax = a4.number_input("Y max", value=0.0, format="%.4f", key="fit_ny_ymax")
            st.caption("보데")
            b1, b2, b3, b4 = st.columns(4)
            bo_fmin = b1.number_input("Freq min", value=0.1,      format="%.4g", key="fit_bo_fmin")
            bo_fmax = b2.number_input("Freq max", value=100000.0, format="%.4g", key="fit_bo_fmax")
            bo_ymin = b3.number_input("Y min",    value=0.0,      format="%.4f", key="fit_bo_ymin")
            bo_ymax = b4.number_input("Y max",    value=0.0,      format="%.4f", key="fit_bo_ymax")

        # ── 나이키스트 + 보데 나란히 ─────────────────────────────────────────
        g_ny, g_bo = st.columns(2)

        # 주파수 순서 그대로 사용 (파일이 이미 고주파→저주파 순)
        freq_vals = df["Freq"].values

        with g_ny:
            fig_ny = plot_nyquist(df, Z_fit_full, ny_xmin, ny_xmax, ny_ymin, ny_ymax)
            for label, color, Z_arc, x_off in arc_Z_list:
                fig_ny.add_trace(go.Scatter(
                    x=Z_arc.real + x_off, y=-Z_arc.imag,
                    mode="lines", name=label,
                    line=dict(color=color, width=1.5, dash="dot"),
                ))
            st.plotly_chart(fig_ny, use_container_width=True)

        with g_bo:
            fig_bo = plot_bode(df, Z_fit_full, bo_fmin, bo_fmax, bo_ymin, bo_ymax)
            for label, color, Z_arc, x_off in arc_Z_list:
                fig_bo.add_trace(go.Scatter(
                    x=freq_vals, y=-Z_arc.imag,
                    mode="lines", name=label,
                    line=dict(color=color, width=1.5, dash="dot"),
                ))
            st.plotly_chart(fig_bo, use_container_width=True)

    # ── 오른쪽: 컨트롤 ────────────────────────────────────────────────────────────
    with col_R:

        # ── 회로 구성 ────────────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ⚡ 회로 구성")
        num_rc = st.slider("R+CPE 병렬 조합 개수", 1, 5, 3, key="fit_num_rc")
        circuit_str = "L — Rs — " + " — ".join([f"(R{i}‖CPE{i})" for i in range(1, num_rc+1)])
        st.markdown(f'<div class="circuit-box">{circuit_str}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 주파수 범위 ──────────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📡 피팅 주파수 범위")
        fc1, fc2 = st.columns(2)
        f_lo = fc1.number_input("최소 (Hz)", value=float(df["Freq"].min()), format="%.4g", key="fit_fit_flo")
        f_hi = fc2.number_input("최대 (Hz)", value=float(df["Freq"].max()), format="%.4g", key="fit_fit_fhi")
        df_fit = df[(df["Freq"] >= f_lo) & (df["Freq"] <= f_hi)].reset_index(drop=True)
        st.caption(f"사용 포인트: **{len(df_fit)}개**")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 알고리즘 선택 + 피팅 실행 ────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ▶ 피팅 알고리즘")

        scope_map = {"로컬 최적화": "local", "전역 최적화": "global", "하이브리드": "hybrid"}
        scope_sel = st.radio("종류", list(scope_map.keys()), horizontal=True, key="fit_algo_scope")
        scope_val = scope_map[scope_sel]

        filtered_algos = {k: v for k, v in ALGORITHMS.items() if v["scope"] == scope_val}
        algo_labels = [v["label"] for v in filtered_algos.values()]
        algo_keys   = list(filtered_algos.keys())

        sel_algo_label = st.selectbox("알고리즘", algo_labels, key="fit_algo_sel")
        sel_algo_key   = algo_keys[algo_labels.index(sel_algo_label)]

        st.markdown(
            f'<p class="algo-desc">ℹ️ {ALGORITHMS[sel_algo_key]["desc"]}</p>',
            unsafe_allow_html=True,
        )

        auto_init = st.checkbox("🤖 자동 초기값 추정", value=True, key="fit_auto_init",
                                help="고주파 Z' → Rs, 저주파 Z' 차이 → R 총합 자동 추정")

        rb_col, sp_col = st.columns([2, 3])
        with rb_col:
            run_btn = st.button("▶ 피팅 실행", type="primary", key="fit_run_fit")

        # status: 버튼 옆 텍스트
        status_placeholder = sp_col.empty()

        # 로딩바: 버튼 바로 아래 (카드 닫기 전)
        progress_placeholder = st.empty()

        # ── 결과 (파라미터 위에 표시) ───────────────────────────────────────────────
        st.markdown('</div>', unsafe_allow_html=True)
        if "fit_popt" in st.session_state:
            popt = st.session_state["fit_popt"]
            res_labels = st.session_state.get("fit_res_labels", [])

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 피팅 결과")
            st.caption(f"알고리즘: {st.session_state.get('fit_fit_algo','')}")

            m1, m2 = st.columns(2)
            m1.metric("Chi²",      f"{st.session_state['fit_chi2']:.3e}")
            m2.metric("Chi² 환원", f"{st.session_state['fit_chi2_red']:.3e}")
            st.markdown(" ")

            # ── 결과 테이블 ──────────────────────────────────────────────────
            def _fmt(v):
                """소수점 5자리 미만 → 지수 표기, 이상 → 전체 소수점"""
                if v == 0:
                    return "0"
                abs_v = abs(v)
                if abs_v < 1e-4:
                    return f"{v:.6e}"
                else:
                    # 소수점 10자리까지 확인 후 trailing zero 제거
                    return f"{v:.10f}".rstrip("0").rstrip(".")

            # 그룹별로 나눠서 테이블 표시
            groups = []
            cur_group = []
            for (sym, name, unit), val in zip(res_labels, popt):
                if sym in ("L", "Rs"):
                    cur_group.append((sym, name, unit, val))
                elif sym.startswith("R") and not sym.startswith("Rs"):
                    if cur_group:
                        groups.append(cur_group)
                    cur_group = [(sym, name, unit, val)]
                else:
                    cur_group.append((sym, name, unit, val))
            if cur_group:
                groups.append(cur_group)

            def _fmt_display(sym, v):
                """UI 표시용: L은 전체, 나머지는 소수점 4자리"""
                if sym == "L":
                    return _fmt(v)
                if abs(v) < 1e-4:
                    return f"{v:.6e}"
                return f"{v:.4f}"

            # L만 메트릭으로
            L_item = next(((s,n,u,v) for (s,n,u,v) in (groups[0] if groups else []) if s == "L"), None)
            if L_item:
                s, n, u, v = L_item
                st.metric(f"{s} ({u})", _fmt_display(s, v))
                st.markdown(" ")

            # Rs + 아크 모두 테이블
            # groups[0] = [L, Rs], groups[1..] = 아크들
            # Rs 테이블
            rs_items = [(s,n,u,v) for (s,n,u,v) in (groups[0] if groups else []) if s == "Rs"]
            if rs_items:
                st.markdown(
                    '<p style="font-size:0.75rem;font-weight:700;color:#4361ee;margin:8px 0 2px;">직렬 저항</p>',
                    unsafe_allow_html=True
                )
                tbl_rs = {
                    "파라미터": [f"{s} ({u})" if u else s for (s,n,u,v) in rs_items],
                    "이름":     [n for (s,n,u,v) in rs_items],
                    "피팅값":   [_fmt_display(s, v) for (s,n,u,v) in rs_items],
                }
                st.dataframe(pd.DataFrame(tbl_rs), use_container_width=True, hide_index=True)

            arc_groups = [g for g in groups if not all(s in ("L","Rs") for (s,n,u,v) in g)]

            # 아크별 테이블
            for g in arc_groups:
                arc_num = g[0][0][1:]  # R1 → "1"
                st.markdown(
                    f'<p style="font-size:0.75rem;font-weight:700;color:#4361ee;'
                    f'margin:8px 0 2px;">아크 {arc_num}</p>',
                    unsafe_allow_html=True
                )
                tbl_data = {
                    "파라미터": [f"{s} ({u})" if u else s for (s,n,u,v) in g],
                    "이름":     [n for (s,n,u,v) in g],
                    "피팅값":   [_fmt_display(s, v) for (s,n,u,v) in g],
                }
                st.dataframe(
                    pd.DataFrame(tbl_data),
                    use_container_width=True,
                    hide_index=True,
                )

            fn = st.session_state.get("fit_fit_filename", "result").replace(".z", "")

            # 회로 문자열 생성
            num_rc_saved = (len(popt) - 2) // 3
            circuit_str = "L — Rs — " + " — ".join([f"(R{i}|CPE{i})" for i in range(1, num_rc_saved + 1)])

            buf = io.StringIO()
            buf.write(f"파일명,{fn}\n")
            buf.write(f"회로 모델,{circuit_str}\n")
            buf.write(f"알고리즘,{st.session_state.get('fit_fit_algo', '')}\n")
            buf.write("\n")
            buf.write("=== 피팅 파라미터 ===\n")
            buf.write("심볼,이름,단위,피팅값\n")
            for (sym, name, unit), val in zip(res_labels, popt):
                buf.write(f"{sym},{name},{unit},{fmt_val(val)}\n")
            buf.write("\n")
            buf.write("=== 임피던스 데이터 (측정값 & 피팅값) ===\n")
            buf.write("Freq(Hz),실수부Z'_측정,허수부Z\"_측정,실수부Z'_피팅,허수부Z\"_피팅,neg허수부Z\"_측정,neg허수부Z\"_피팅\n")
            Z_fit_csv = st.session_state["fit_Z_fit_full"]
            for i, row in df.iterrows():
                freq_v = row["Freq"]
                zr_m = row["Zr"]
                zi_m = row["Zi"]
                zr_f = float(Z_fit_csv.real[i])
                zi_f = float(Z_fit_csv.imag[i])
                buf.write(
                    f"{fmt_val(freq_v)},{fmt_val(zr_m)},{fmt_val(zi_m)},"
                    f"{fmt_val(zr_f)},{fmt_val(zi_f)},"
                    f"{fmt_val(-zi_m)},{fmt_val(-zi_f)}\n"
                )
            st.markdown(" ")
            st.download_button("⬇️ 결과 CSV (파라미터 + 임피던스 데이터)",
                               data=buf.getvalue().encode("utf-8-sig"),
                               file_name=f"EIS_fit_{fn}.csv", mime="text/csv")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── 파라미터 입력 ─────────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔧 파라미터 설정")

        hc0, hc1, hc2, hc3 = st.columns([1.5, 1, 1, 1])
        for hc, txt in zip([hc1, hc2, hc3], ["초기값", "하한", "상한"]):
            hc.markdown(f'<p style="font-size:0.70rem;color:#aaa;text-align:center;margin:0">{txt}</p>',
                        unsafe_allow_html=True)

        p0, lo_b, hi_b = [], [], []

        # Rs 초기값: Z''이 양→음으로 바뀌는 교차점의 Z' (내삽)
        _zr = df["Zr"].values
        _zi = df["Zi"].values
        _rs_default = 0.30  # fallback
        for _k in range(len(_zi) - 1):
            if _zi[_k] > 0 and _zi[_k+1] <= 0:
                _t = _zi[_k] / (_zi[_k] - _zi[_k+1])
                _rs_default = float(_zr[_k] + _t * (_zr[_k+1] - _zr[_k]))
                break

        st.markdown('<p class="group-title">인덕턴스 &amp; 직렬 저항</p>', unsafe_allow_html=True)

        # L
        st.markdown('<p class="param-label">L [H] — 인덕턴스</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        v0_L  = c1.number_input("v", value=1e-7, format="%.2e", key="p0_L",  label_visibility="collapsed")
        vlo_L = c2.number_input("l", value=1e-12, format="%.2e", key="lo_L", label_visibility="collapsed")
        vhi_L = c3.number_input("h", value=1e-3,  format="%.2e", key="hi_L", label_visibility="collapsed")
        p0.append(v0_L); lo_b.append(vlo_L); hi_b.append(vhi_L)

        # Rs — 파일명 기반 key로 session_state 캐시 우회
        _fname_key = uploaded_fit.name.replace(".", "_").replace(" ", "_")
        st.markdown('<p class="param-label">Rs [Ω] — 직렬 저항</p>', unsafe_allow_html=True)
        st.caption(f"🔍 자동 추정 Rs ≈ {_rs_default:.4f} Ω (Z\'\' 부호 교차점)")
        c1, c2, c3 = st.columns(3)
        v0_Rs  = c1.number_input("v", value=_rs_default, format="%.4f", key=f"p0_Rs_{_fname_key}", label_visibility="collapsed")
        vlo_Rs = c2.number_input("l", value=1e-4,        format="%.2e", key=f"lo_Rs_{_fname_key}", label_visibility="collapsed")
        vhi_Rs = c3.number_input("h", value=10.0,        format="%.2e", key=f"hi_Rs_{_fname_key}", label_visibility="collapsed")
        p0.append(v0_Rs); lo_b.append(vlo_Rs); hi_b.append(vhi_Rs)

        R_defs = [0.02, 0.10, 0.30, 0.50, 1.00]
        Q_defs = [1e-3, 5e-3, 1e-2, 2e-2, 5e-2]
        n_defs = [0.80, 0.75, 0.60, 0.65, 0.70]

        for i in range(1, num_rc + 1):
            st.markdown(f'<hr><p class="group-title">아크 {i} — R{i}‖CPE{i}</p>', unsafe_allow_html=True)
            for sym, name, unit, default, lo, hi in [
                (f"R{i}", f"저항 {i}",      "Ω",    R_defs[i-1], 1e-4, 100.0),
                (f"Q{i}", f"CPE{i} 계수",   "S·sⁿ", Q_defs[i-1], 1e-9, 10.0 ),
                (f"n{i}", f"CPE{i} 지수",   "",     n_defs[i-1], 0.01, 1.00 ),
            ]:
                tag = f"{sym} [{unit}]" if unit else sym
                st.markdown(f'<p class="param-label">{tag} — {name}</p>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                v0  = c1.number_input("v", value=default, format="%.2e", key=f"p0_{sym}", label_visibility="collapsed")
                vlo = c2.number_input("l", value=lo,      format="%.2e", key=f"lo_{sym}", label_visibility="collapsed")
                vhi = c3.number_input("h", value=hi,      format="%.2e", key=f"hi_{sym}", label_visibility="collapsed")
                p0.append(v0); lo_b.append(vlo); hi_b.append(vhi)

        st.markdown('</div>', unsafe_allow_html=True)

        if run_btn:
            status_placeholder.markdown(
                f'<p style="color:#555;font-size:0.85rem;margin-top:8px">'
                f'⏳ {sel_algo_label} 최적화 진행 중...</p>',
                unsafe_allow_html=True
            )

            freq_arr = df_fit["Freq"].values
            zr_arr   = df_fit["Zr"].values
            zi_arr   = df_fit["Zi"].values
            n_params = 2 + num_rc * 3

            if len(freq_arr) < n_params:
                st.error(f"데이터 포인트({len(freq_arr)})가 파라미터 수({n_params})보다 적습니다.")
            else:
                if auto_init:
                    # Rs 추정: Z''이 음수→양수로 바뀌는 교차점의 Z'
                    # (유도성 고주파 영역 제외하고 실제 Rs 추정)
                    rs_est = float(zr_arr[np.argmax(freq_arr)])  # fallback
                    for _i in range(len(zi_arr) - 1):
                        if zi_arr[_i] > 0 and zi_arr[_i+1] <= 0:
                            # 양수→음수 교차 (고→저주파 방향)
                            _t = zi_arr[_i] / (zi_arr[_i] - zi_arr[_i+1])
                            rs_est = float(zr_arr[_i] + _t * (zr_arr[_i+1] - zr_arr[_i]))
                            break
                        elif zi_arr[_i] <= 0 and zi_arr[_i+1] > 0:
                            # 음수→양수 교차
                            _t = -zi_arr[_i] / (zi_arr[_i+1] - zi_arr[_i])
                            rs_est = float(zr_arr[_i] + _t * (zr_arr[_i+1] - zr_arr[_i]))
                            break

                    re_est  = float(zr_arr[np.argmin(freq_arr)])
                    r_total = max(re_est - rs_est, 0.01)
                    p0[1]   = rs_est
                    ws = [0.10, 0.25, 0.40, 0.15, 0.10][:num_rc]
                    s  = sum(ws)
                    for i in range(num_rc):
                        p0[2 + i*3] = r_total * ws[i] / s

                # LM은 경계 조건 불가 → 초기값 클리핑만
                if sel_algo_key == "LM":
                    p0 = [max(lo, min(hi, v)) for v, lo, hi in zip(p0, lo_b, hi_b)]

                try:
                        # 로딩바 (버튼 아래 미리 선언한 placeholder에 표시)
                        progress_placeholder.progress(0, text=f"⏳ {sel_algo_label} 실행 중...")

                        def _update_progress(v):
                            pct = int(v * 100)
                            progress_placeholder.progress(pct, text=f"⏳ {sel_algo_label} 실행 중... {pct}%")

                        popt = run_fitting(sel_algo_key, p0, lo_b, hi_b,
                                           freq_arr, zr_arr, zi_arr, num_rc,
                                           progress_cb=_update_progress)

                        Z_fit_full = circuit_impedance(df["Freq"].values, popt, num_rc)
                        chi2     = chi2_fn(popt, freq_arr, zr_arr, zi_arr, num_rc)
                        chi2_red = chi2 / max(1, 2*len(freq_arr) - n_params)

                        res_labels = [("L","인덕턴스","H"), ("Rs","직렬 저항","Ω")]
                        for i in range(1, num_rc+1):
                            res_labels += [
                                (f"R{i}", f"저항 {i}",    "Ω"),
                                (f"Q{i}", f"CPE{i} 계수", "S·sⁿ"),
                                (f"n{i}", f"CPE{i} 지수", ""),
                            ]

                        st.session_state.update({
                            "fit_popt": popt, "fit_Z_fit_full": Z_fit_full,
                            "fit_chi2": chi2, "fit_chi2_red": chi2_red,
                            "fit_res_labels": res_labels,
                            "fit_fit_algo": sel_algo_label,
                            "fit_fit_filename": uploaded_fit.name,
                        })
                        progress_placeholder.progress(100, text="✅ 완료!")
                        status_placeholder.empty()
                        st.rerun()

                except Exception as e:
                        status_placeholder.empty()
                        st.error(f"❌ 피팅 실패: {e}")

        st.markdown('</div>', unsafe_allow_html=True)


# ── 공통 파서 (장기 데이터용) ──────────────────────────────────────────────────
def parse_idf(file_bytes: bytes, downsample: int = 60) -> pd.DataFrame | None:
    text = file_bytes.decode("latin-1")
    technique_match = re.search(r'Technique=(\w+)', text)
    technique = technique_match.group(1) if technique_match else ""
    is_chronopot = (technique == "ChronoPotentiometry")

    idx = text.find("primary_data")
    if idx == -1:
        return None
    after = text[idx + len("primary_data"):]
    m = re.search(r"\r\n(\d+)\s*\r\n", after)
    if not m:
        return None
    data_start = idx + len("primary_data") + m.end()
    data_text = text[data_start:]

    rows = []
    for line in data_text.split("\r\n"):
        line = line.strip()
        if not line:
            break
        parts = line.split()
        if len(parts) == 3:
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                break
        else:
            break

    if not rows:
        return None

    raw_df = pd.DataFrame(rows, columns=["time_s", "col2", "col3"])
    if is_chronopot:
        raw_df["current_a"] = raw_df["col3"]
        raw_df["voltage_v"] = raw_df["col2"]
    else:
        raw_df["current_a"] = raw_df["col2"]
        raw_df["voltage_v"] = raw_df["col3"]

    df = raw_df[["time_s", "current_a", "voltage_v"]].copy()
    df = df.iloc[::downsample].reset_index(drop=True)
    return df


# ── IVP 전용 함수 ──────────────────────────────────────────────────────────────
def parse_idf_ivp(file_bytes: bytes) -> pd.DataFrame:
    text = file_bytes.decode("latin-1")
    lines = text.split("\r\n")
    start_idx, n_points = None, 0
    for i, line in enumerate(lines):
        if line.strip().startswith("primary_data"):
            n_points = int(lines[i + 1].strip())
            start_idx = i + 2
            break
    if start_idx is None:
        raise ValueError("primary_data 섹션을 찾을 수 없습니다.")
    rows = []
    for line in lines[start_idx: start_idx + n_points]:
        cols = line.split()
        if len(cols) >= 2:
            rows.append((-float(cols[0]), float(cols[1])))
    return pd.DataFrame(rows, columns=["I (A)", "E (V)"])


def detect_mode(i_arr):
    valid = i_arr[~np.isnan(i_arr)]
    if len(valid) == 0:
        return "FC"
    return "FC" if abs(np.max(valid)) >= abs(np.min(valid)) else "EC"


def interpolate_at(i_arr, e_arr, p_arr, target):
    abs_i = np.abs(i_arr)
    for k in range(len(abs_i) - 1):
        a, b = abs_i[k], abs_i[k + 1]
        if (a <= target <= b) or (b <= target <= a):
            t = (target - a) / (b - a) if b != a else 0
            return (e_arr[k] + t * (e_arr[k+1] - e_arr[k]),
                    p_arr[k] + t * (p_arr[k+1] - p_arr[k]))
    return None, None


def get_ocv(i_arr, e_arr):
    for k, iv in enumerate(i_arr):
        if iv == 0.0:
            return e_arr[k]
    for k in range(len(i_arr) - 1):
        a, b = i_arr[k], i_arr[k + 1]
        if (a <= 0 <= b) or (b <= 0 <= a):
            t = (0 - a) / (b - a) if b != a else 0
            return e_arr[k] + t * (e_arr[k + 1] - e_arr[k])
    abs_i = np.abs(i_arr)
    idx = np.argsort(abs_i)
    k0, k1 = idx[0], idx[1]
    a, b = i_arr[k0], i_arr[k1]
    if b != a:
        t = (0 - a) / (b - a)
        return e_arr[k0] + t * (e_arr[k1] - e_arr[k0])
    return e_arr[k0]


def process_file_ivp(file_bytes, area_cm2, area, area_unit, file_stem):
    df = parse_idf_ivp(file_bytes)
    df["I density (A/cm²)"]     = df["I (A)"] / area_cm2
    df["Power (W)"]              = df["I (A)"] * df["E (V)"]
    df["Power density (W/cm²)"] = df["Power (W)"] / area_cm2

    i_arr = df["I density (A/cm²)"].values
    e_arr = df["E (V)"].values
    p_arr = df["Power density (W/cm²)"].values

    mode    = detect_mode(i_arr)
    ppd     = float(np.max(np.abs(p_arr)))
    ocv     = get_ocv(i_arr, e_arr)
    targets = [0.5, 1.0, 2.0, 3.0]
    interp  = {t: interpolate_at(i_arr, e_arr, p_arr, t) for t in targets}

    csv_rows = [
        [file_stem, "", ""],
        ["Active Area", f"{area} {area_unit}", ""],
        ["OCV", f"{ocv:.6g}", ""],
        ["PPD", f"{ppd:.6g}", ""],
        ["", "Voltage", "Power"],
    ]
    for t in targets:
        v, p = interp[t]
        csv_rows.append([f"{t}A",
                         f"{v:.6g}" if v is not None else "N/A",
                         f"{p:.6g}" if p is not None else "N/A"])
    csv_rows += [["", "", ""],
                 [f"{file_stem} I", f"{file_stem} V", f"{file_stem} P"],
                 ["I density (A/cm²)", "E (V)", "Power density (W/cm²)"]]
    for _, row in df.iterrows():
        csv_rows.append([str(row["I density (A/cm²)"]),
                         str(row["E (V)"]),
                         str(row["Power density (W/cm²)"])])
    return csv_rows, ppd, ocv, interp, df, mode


def blocks_to_csv(all_blocks):
    if not all_blocks:
        return b""
    max_rows = max(len(b) for b in all_blocks)
    padded = [b + [["", "", ""]] * (max_rows - len(b)) for b in all_blocks]
    buf = io.StringIO()
    for r in range(max_rows):
        row_cells = []
        for b_idx, block in enumerate(padded):
            row_cells.extend(block[r])
            if b_idx < len(padded) - 1:
                row_cells.append("")
        buf.write(",".join(row_cells) + "\n")
    return buf.getvalue().encode("utf-8-sig")


def auto_x_range(plot_data):
    modes = {item["mode"] for item in plot_data}
    if "FC" in modes and "EC" in modes:
        return -4.0, 4.0
    elif "EC" in modes:
        return -4.0, 0.0
    else:
        return 0.0, 4.0


def make_ivp_chart(plot_data, x_min, x_max, y1_min, y1_max, y2_min, y2_max):
    fig = go.Figure()
    for i, item in enumerate(plot_data):
        color = COLORS[i % len(COLORS)]
        df, name, mode = item["df"], item["name"], item["mode"]
        x = df["I density (A/cm²)"].values
        v = df["E (V)"].values
        p = np.abs(df["Power density (W/cm²)"].values)
        label = f"{name} [{mode}]"
        fig.add_trace(go.Scatter(x=x, y=v, name=f"{label} — V",
                                 mode="lines", line=dict(color=color, width=2), yaxis="y1"))
        fig.add_trace(go.Scatter(x=x, y=p, name=f"{label} — P",
                                 mode="lines", line=dict(color=color, width=2, dash="dash"), yaxis="y2"))
    fig.update_layout(
        shapes=[
            dict(type="line", x0=0, x1=0, y0=y1_min, y1=y1_max,
                 xref="x", yref="y", line=dict(color="black", width=2.5)),
            dict(type="line", x0=x_min, x1=x_max, y0=0, y1=0,
                 xref="x", yref="y", line=dict(color="black", width=2.5)),
        ],
        xaxis=dict(title="Current Density (A/cm²)", range=[x_min, x_max],
                   showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Voltage (V)", range=[y1_min, y1_max],
                   showgrid=True, gridcolor="#e0e0e0"),
        yaxis2=dict(title="Power Density (W/cm²)", range=[y2_min, y2_max],
                    overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="v", x=1.08, y=1),
        plot_bgcolor="white", paper_bgcolor="white",
        height=550, margin=dict(l=60, r=160, t=40, b=60),
    )
    return fig


def render_settings(key_prefix: str):
    """시간 단위 / Active Area / 다운샘플 / 초기5분제거 / Organize 버튼을 탭 내부에 렌더링"""
    with st.expander("⚙️ 설정", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("🕐 시간 단위")
            time_unit = st.radio(
                "단위 선택",
                options=["초 (s)", "분 (min)", "시간 (h)", "일 (day)"],
                index=2,
                key=f"{key_prefix}_time_unit",
                horizontal=True,
            )
        with c2:
            st.caption("⚡ Active Area")
            active_area = st.number_input(
                "Active Area (cm²)",
                min_value=0.001, value=1.0, step=0.1, format="%.3f",
                key=f"{key_prefix}_active_area",
            )
        with c3:
            st.caption("📉 다운샘플링 배율")
            downsample = st.number_input(
                "N배 다운샘플링",
                min_value=1, max_value=1000, value=60, step=1,
                key=f"{key_prefix}_downsample",
                help="1=전체, 60=60배 축약"
            )
        with c4:
            st.caption("🗑️ 초기 5분 데이터 제거")
            remove_5min = st.toggle(
                "초기 5분 제거",
                value=False,
                key=f"{key_prefix}_remove5min",
                help="ON: 각 파일의 처음 300초(5분) 데이터를 제거합니다"
            )
    time_divisor = {"초 (s)": 1, "분 (min)": 60, "시간 (h)": 3600, "일 (day)": 86400}[time_unit]
    time_label   = {"초 (s)": "s", "분 (min)": "min", "시간 (h)": "h", "일 (day)": "day"}[time_unit]
    organize     = st.button("▶ Organize", type="primary", key=f"{key_prefix}_organize")
    return time_divisor, time_label, active_area, organize, downsample, remove_5min


# ── 메인 탭 배너 (상위) ────────────────────────────────────────────────────────
tab_longterm, tab_ivp, tab_eis = st.tabs(["📂 Long Term 정리", "⚡ IVP 정리", "🔬 EIS 정리"])


# ══════════════════════════════════════════════════════════════════════════════
# Long Term 정리
# ══════════════════════════════════════════════════════════════════════════════
with tab_longterm:
    tab_개별, tab_겹치기, tab_이어붙이기 = st.tabs(["📋 개별 정리", "📊 겹치기", "🔗 이어붙이기"])

    # ── 개별 정리 ──────────────────────────────────────────────────────────────
    with tab_개별:
            uploaded_files_ind = st.file_uploader(
                "IDF 파일 업로드 (복수 선택 가능)",
                type=["idf"], accept_multiple_files=True, key="uploader_individual",
            )
            time_divisor, time_label, active_area, organize, downsample, remove_5min = render_settings("ind")

            if not uploaded_files_ind:
                st.info("👆 IDF 파일을 업로드해 주세요.")
            elif not organize and "organized_ind" not in st.session_state:
                st.info("⚙️ 설정을 확인한 후 **Organize** 버튼을 눌러주세요.")
            else:
                if organize and uploaded_files_ind:
                    st.session_state["organized_ind"]    = True
                    st.session_state["file_data_ind"]    = {f.name: f.read() for f in uploaded_files_ind}
                    st.session_state["time_label_ind"]   = time_label
                    st.session_state["time_divisor_ind"] = time_divisor
                    st.session_state["active_area_ind"]  = active_area
                    st.session_state["downsample_ind"]   = downsample
                    st.session_state["remove5min_ind"]   = remove_5min

                file_data_ind    = st.session_state["file_data_ind"]
                time_label_ind   = st.session_state["time_label_ind"]
                time_divisor_ind = st.session_state["time_divisor_ind"]
                active_area_ind  = st.session_state["active_area_ind"]
                downsample_ind   = st.session_state.get("downsample_ind", 60)
                remove5min_ind   = st.session_state.get("remove5min_ind", False)
                time_col_ind     = f"Time ({time_label_ind})"

                parsed_ind = {}
                for filename, file_bytes in file_data_ind.items():
                    df = parse_idf(file_bytes, downsample_ind)
                    if df is not None:
                        if remove5min_ind:
                            df = df[df["time_s"] >= 300].reset_index(drop=True)
                        parsed_ind[filename] = pd.DataFrame({
                            time_col_ind: df["time_s"] / time_divisor_ind,
                            current_col:  (df["current_a"] / active_area_ind).round(2),
                            voltage_col:  df["voltage_v"].round(2),
                        })

                if not parsed_ind:
                    st.error("파싱된 파일이 없습니다.")
                else:
                    file_tabs = st.tabs(list(parsed_ind.keys()))
                    for ftab, (filename, display_df) in zip(file_tabs, parsed_ind.items()):
                        with ftab:
                            filename_stem = filename.replace(".idf", "")
                            t_min = float(display_df[time_col_ind].min())
                            t_max = float(display_df[time_col_ind].max())
                            avg_current_density = round(float(display_df[current_col].mean()), 2)

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("총 데이터 포인트", f"{len(display_df):,} ({downsample_ind}배 다운샘플)")
                            col2.metric("총 시간", f"{t_max:.2f} {time_label_ind}")
                            col3.metric("전압 범위", f"{display_df[voltage_col].min():.2f} ~ {display_df[voltage_col].max():.2f} V")
                            col4.metric("전류밀도 범위", f"{display_df[current_col].min():.2f} ~ {display_df[current_col].max():.2f} A/cm²")

                            st.divider()
                            st.subheader("📉 전압 기울기 분석")

                            sc1, sc2, sc3, sc4 = st.columns(4)
                            with sc1:
                                x_min = st.number_input(f"X 최솟값 ({time_label_ind})", value=t_min, step=0.1, key=f"xmin_{filename}")
                            with sc2:
                                x_max = st.number_input(f"X 최댓값 ({time_label_ind})", value=t_max, step=0.1, key=f"xmax_{filename}")
                            with sc3:
                                y_min = st.number_input("Y 최솟값 (V)", value=0.0, step=0.01, key=f"ymin_{filename}")
                            with sc4:
                                y_max = st.number_input("Y 최댓값 (V)", value=1.5, step=0.01, key=f"ymax_{filename}")

                            gc1, gc2 = st.columns(2)
                            with gc1:
                                t_start = st.number_input(f"기울기 구간 시작 ({time_label_ind})", min_value=t_min, max_value=t_max, value=t_min, step=0.1, key=f"ts_{filename}")
                            with gc2:
                                t_end = st.number_input(f"기울기 구간 끝 ({time_label_ind})", min_value=t_min, max_value=t_max, value=t_max, step=0.1, key=f"te_{filename}")

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=display_df[time_col_ind], y=display_df[voltage_col],
                                mode="lines", name="Voltage (V)",
                                line=dict(color="#1f77b4", width=1.5),
                            ))

                            if t_start < t_end:
                                mask = (display_df[time_col_ind] >= t_start) & (display_df[time_col_ind] <= t_end)
                                subset = display_df.loc[mask]
                                if not subset.empty:
                                    coeffs = np.polyfit(subset[time_col_ind], subset[voltage_col], 1)
                                    slope, intercept = coeffs[0], coeffs[1]
                                    fig.add_trace(go.Scatter(
                                        x=[t_min, t_max], y=[slope * t_min + intercept, slope * t_max + intercept],
                                        mode="lines", name="Linear Fit",
                                        line=dict(color="#ff7f0e", width=2, dash="dash"),
                                    ))
                                    mc1, mc2, mc3 = st.columns(3)
                                    mc1.metric("기울기", f"{slope:.6f} V/{time_label_ind}")
                                    mc2.metric("절편", f"{intercept:.4f} V")
                                    mc3.metric("선형식", f"V = {slope:.6f}·t + {intercept:.4f}")

                            fig.update_layout(
                                xaxis=dict(title=time_col_ind, range=[x_min, x_max]),
                                yaxis=dict(title="Voltage (V)", range=[y_min, y_max]),
                                height=600, width=720, margin=dict(l=20, r=20, t=20, b=20),
                                legend=dict(orientation="h", y=-0.15),
                            )
                            _, center_col, _ = st.columns([1, 6, 1])
                            with center_col:
                                st.plotly_chart(fig, use_container_width=False)

                            st.divider()
                            st.subheader("📋 데이터 테이블")
                            fmt = {time_col_ind: "{:.2f}", current_col: "{:.2f}", voltage_col: "{:.2f}"}
                            max_rows = st.number_input("표시할 행 수", min_value=10, max_value=min(10000, len(display_df)),
                                                       value=min(500, len(display_df)), step=10, key=f"rows_{filename}")
                            st.dataframe(display_df.head(int(max_rows)).style.format(fmt), use_container_width=True, height=400)

                            output = io.StringIO()
                            output.write(f"{filename_stem},,,\n")
                            output.write(f"Active area (cm2),{active_area_ind},,\n")
                            output.write(f"Current density (A cm-2),{avg_current_density},,\n")
                            output.write(f"{filename_stem} Time ({time_label_ind}),{filename_stem} Current Density (A/cm²),{filename_stem} Voltage (V)\n")
                            for _, row in display_df.iterrows():
                                output.write(f"{row[time_col_ind]:.6f},{row[current_col]:.2f},{row[voltage_col]:.2f}\n")

                            st.download_button(
                                label="⬇️ CSV 다운로드",
                                data=output.getvalue().encode("utf-8-sig"),
                                file_name=f"{today}_Longterm_개별정리_{filename_stem}.csv",
                                mime="text/csv", key=f"dl_{filename}",
                            )


        # ══════════════════════════════════════════════════════════════════════════════
        # 겹치기
        # ══════════════════════════════════════════════════════════════════════════════

    # ── 겹치기 ────────────────────────────────────────────────────────────────
    with tab_겹치기:
            uploaded_files_ov = st.file_uploader(
                "IDF 파일 업로드 (복수 선택 가능)",
                type=["idf"], accept_multiple_files=True, key="uploader_overlay",
            )
            time_divisor, time_label, active_area, organize, downsample, remove_5min = render_settings("ov")

            if not uploaded_files_ov:
                st.info("👆 IDF 파일을 업로드해 주세요.")
            elif not organize and "organized_ov" not in st.session_state:
                st.info("⚙️ 설정을 확인한 후 **Organize** 버튼을 눌러주세요.")
            else:
                if organize and uploaded_files_ov:
                    st.session_state["organized_ov"]    = True
                    st.session_state["file_data_ov"]    = {f.name: f.read() for f in uploaded_files_ov}
                    st.session_state["time_label_ov"]   = time_label
                    st.session_state["time_divisor_ov"] = time_divisor
                    st.session_state["active_area_ov"]  = active_area
                    st.session_state["downsample_ov"]   = downsample
                    st.session_state["remove5min_ov"]   = remove_5min

                file_data_ov    = st.session_state["file_data_ov"]
                time_label_ov   = st.session_state["time_label_ov"]
                time_divisor_ov = st.session_state["time_divisor_ov"]
                active_area_ov  = st.session_state["active_area_ov"]
                downsample_ov   = st.session_state.get("downsample_ov", 60)
                remove5min_ov   = st.session_state.get("remove5min_ov", False)
                time_col_ov     = f"Time ({time_label_ov})"

                parsed_ov = {}
                for filename, file_bytes in file_data_ov.items():
                    df = parse_idf(file_bytes, downsample_ov)
                    if df is not None:
                        if remove5min_ov:
                            df = df[df["time_s"] >= 300].reset_index(drop=True)
                        parsed_ov[filename] = pd.DataFrame({
                            time_col_ov: df["time_s"] / time_divisor_ov,
                            current_col: (df["current_a"] / active_area_ov).round(2),
                            voltage_col: df["voltage_v"].round(2),
                        })

                if not parsed_ov:
                    st.error("파싱된 파일이 없습니다.")
                else:
                    st.subheader("📊 겹치기 — 전압 비교")
                    all_t_max = max(float(df[time_col_ov].max()) for df in parsed_ov.values())
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    with sc1:
                        x_min = st.number_input(f"X 최솟값 ({time_label_ov})", value=0.0, step=0.1, key="ov_xmin")
                    with sc2:
                        x_max = st.number_input(f"X 최댓값 ({time_label_ov})", value=all_t_max, step=0.1, key="ov_xmax")
                    with sc3:
                        y_min = st.number_input("Y 최솟값 (V)", value=0.0, step=0.01, key="ov_ymin")
                    with sc4:
                        y_max = st.number_input("Y 최댓값 (V)", value=1.5, step=0.01, key="ov_ymax")

                    fig_all = go.Figure()
                    for i, (filename, display_df) in enumerate(parsed_ov.items()):
                        fig_all.add_trace(go.Scatter(
                            x=display_df[time_col_ov], y=display_df[voltage_col],
                            mode="lines", name=filename.replace(".idf", ""),
                            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                        ))
                    fig_all.update_layout(
                        xaxis=dict(title=time_col_ov, range=[x_min, x_max]),
                        yaxis=dict(title="Voltage (V)", range=[y_min, y_max]),
                        height=600, width=720, margin=dict(l=20, r=20, t=20, b=20),
                        legend=dict(orientation="h", y=-0.15),
                    )
                    _, center_col_all, _ = st.columns([1, 6, 1])
                    with center_col_all:
                        st.plotly_chart(fig_all, use_container_width=False)

                    st.divider()
                    st.subheader("📉 파일별 전압 기울기 분석")

                    for i, (filename, display_df) in enumerate(parsed_ov.items()):
                        color = COLORS[i % len(COLORS)]
                        stem  = filename.replace(".idf", "")
                        with st.expander(f"📂 {stem}", expanded=False):
                            t_min = float(display_df[time_col_ov].min())
                            t_max = float(display_df[time_col_ov].max())

                            ec1, ec2, ec3, ec4 = st.columns(4)
                            with ec1:
                                ex_min = st.number_input(f"X 최솟값 ({time_label_ov})", value=t_min, step=0.1, key=f"ov_xmin_{filename}")
                            with ec2:
                                ex_max = st.number_input(f"X 최댓값 ({time_label_ov})", value=t_max, step=0.1, key=f"ov_xmax_{filename}")
                            with ec3:
                                ey_min = st.number_input("Y 최솟값 (V)", value=0.0, step=0.01, key=f"ov_ymin_{filename}")
                            with ec4:
                                ey_max = st.number_input("Y 최댓값 (V)", value=1.5, step=0.01, key=f"ov_ymax_{filename}")

                            gc1, gc2 = st.columns(2)
                            with gc1:
                                t_start = st.number_input(f"기울기 구간 시작 ({time_label_ov})", min_value=t_min, max_value=t_max, value=t_min, step=0.1, key=f"ov_ts_{filename}")
                            with gc2:
                                t_end = st.number_input(f"기울기 구간 끝 ({time_label_ov})", min_value=t_min, max_value=t_max, value=t_max, step=0.1, key=f"ov_te_{filename}")

                            fig_ind = go.Figure()
                            fig_ind.add_trace(go.Scatter(
                                x=display_df[time_col_ov], y=display_df[voltage_col],
                                mode="lines", name="Voltage (V)",
                                line=dict(color=color, width=1.5),
                            ))

                            if t_start < t_end:
                                mask = (display_df[time_col_ov] >= t_start) & (display_df[time_col_ov] <= t_end)
                                subset = display_df.loc[mask]
                                if not subset.empty:
                                    coeffs = np.polyfit(subset[time_col_ov], subset[voltage_col], 1)
                                    slope, intercept = coeffs[0], coeffs[1]
                                    fig_ind.add_trace(go.Scatter(
                                        x=[t_min, t_max], y=[slope * t_min + intercept, slope * t_max + intercept],
                                        mode="lines", name="Linear Fit",
                                        line=dict(color="black", width=2, dash="dash"),
                                    ))
                                    mc1, mc2, mc3 = st.columns(3)
                                    mc1.metric("기울기", f"{slope:.6f} V/{time_label_ov}")
                                    mc2.metric("절편", f"{intercept:.4f} V")
                                    mc3.metric("선형식", f"V = {slope:.6f}·t + {intercept:.4f}")

                            fig_ind.update_layout(
                                xaxis=dict(title=time_col_ov, range=[ex_min, ex_max]),
                                yaxis=dict(title="Voltage (V)", range=[ey_min, ey_max]),
                                height=600, width=720, margin=dict(l=20, r=20, t=20, b=20),
                                legend=dict(orientation="h", y=-0.15),
                            )
                            _, center_col_ind, _ = st.columns([1, 6, 1])
                            with center_col_ind:
                                st.plotly_chart(fig_ind, use_container_width=False)

                    st.divider()
                    filenames = list(parsed_ov.keys())
                    output = io.StringIO()
                    row1, row2, row3, row4 = [], [], [], []
                    for filename in filenames:
                        stem   = filename.replace(".idf", "")
                        avg_cd = round(float(parsed_ov[filename][current_col].mean()), 2)
                        row1 += [stem, "", "", ""]
                        row2 += ["Active area (cm2)", str(active_area_ov), "", ""]
                        row3 += ["Current density (A cm-2)", str(avg_cd), "", ""]
                        row4 += [f"{stem} Time ({time_label_ov})", f"{stem} Current Density (A/cm²)", f"{stem} Voltage (V)", ""]

                    output.write(",".join(row1).rstrip(",") + "\n")
                    output.write(",".join(row2).rstrip(",") + "\n")
                    output.write(",".join(row3).rstrip(",") + "\n")
                    output.write(",".join(row4).rstrip(",") + "\n")

                    max_len = max(len(df) for df in parsed_ov.values())
                    for i in range(max_len):
                        row = []
                        for df in parsed_ov.values():
                            if i < len(df):
                                r = df.iloc[i]
                                row += [f"{r[time_col_ov]:.6f}", f"{r[current_col]:.2f}", f"{r[voltage_col]:.2f}", ""]
                            else:
                                row += ["", "", "", ""]
                        output.write(",".join(row).rstrip(",") + "\n")

                    st.download_button(
                        label="⬇️ 겹치기 CSV 다운로드",
                        data=output.getvalue().encode("utf-8-sig"),
                        file_name=f"{today}_Longterm_겹치기.csv",
                        mime="text/csv",
                    )


        # ══════════════════════════════════════════════════════════════════════════════
        # 이어붙이기
        # ══════════════════════════════════════════════════════════════════════════════

    # ── 이어붙이기 ────────────────────────────────────────────────────────────
    with tab_이어붙이기:
            st.caption("순서대로 파일을 올려주세요. 비어있는 슬롯은 건너뜁니다. (최대 10개)")

            slot_files = {}
            for row_start in range(0, 10, 2):
                cols = st.columns(2)
                for col_idx, col in enumerate(cols):
                    slot_num = row_start + col_idx + 1
                    with col:
                        f = st.file_uploader(f"📂 {slot_num}번째 파일", type=["idf"], key=f"slot_{slot_num}")
                        if f is not None:
                            slot_files[slot_num] = f

            time_divisor, time_label, active_area, organize, downsample, remove_5min = render_settings("ct")

            if not slot_files:
                st.info("👆 파일을 순서대로 올린 후 **Organize** 버튼을 눌러주세요.")
            elif not organize and "organized_concat" not in st.session_state:
                st.info("⚙️ **Organize** 버튼을 눌러주세요.")
            else:
                if organize and slot_files:
                    st.session_state["organized_concat"] = True
                    st.session_state["concat_slots"]     = {k: v.read() for k, v in sorted(slot_files.items())}
                    st.session_state["concat_names"]     = {k: v.name  for k, v in sorted(slot_files.items())}
                    st.session_state["time_label_ct"]    = time_label
                    st.session_state["time_divisor_ct"]  = time_divisor
                    st.session_state["active_area_ct"]   = active_area

                concat_slots    = st.session_state["concat_slots"]
                concat_names    = st.session_state["concat_names"]
                time_label_ct   = st.session_state["time_label_ct"]
                time_divisor_ct = st.session_state["time_divisor_ct"]
                active_area_ct  = st.session_state["active_area_ct"]
                time_col_ct     = f"Time ({time_label_ct})"

                combined_parts = []
                time_offset = 0.0
                for slot_num in sorted(concat_slots.keys()):
                    raw  = concat_slots[slot_num]
                    name = concat_names[slot_num]
                    df   = parse_idf(raw, downsample_ct)
                    if df is None:
                        st.warning(f"⚠️ {slot_num}번째 파일({name}) 파싱 실패, 건너뜁니다.")
                        continue
                    if remove5min_ct:
                        df = df[df["time_s"] >= 300].reset_index(drop=True)
                    part = pd.DataFrame({
                        time_col_ct: df["time_s"] / time_divisor_ct + time_offset,
                        current_col: (df["current_a"] / active_area_ct).round(2),
                        voltage_col: df["voltage_v"].round(2),
                    })
                    time_offset = float(part[time_col_ct].max())
                    part["_source"] = name.replace(".idf", "")
                    combined_parts.append(part)

                if not combined_parts:
                    st.error("파싱된 파일이 없습니다.")
                else:
                    combined_df = pd.concat(combined_parts, ignore_index=True)
                    total_time  = float(combined_df[time_col_ct].max())
                    avg_current_density = round(float(combined_df[current_col].mean()), 2)

                    st.divider()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("총 데이터 포인트", f"{len(combined_df):,}")
                    col2.metric("총 시간", f"{total_time:.2f} {time_label_ct}")
                    col3.metric("전압 범위", f"{combined_df[voltage_col].min():.2f} ~ {combined_df[voltage_col].max():.2f} V")
                    col4.metric("전류밀도 범위", f"{combined_df[current_col].min():.2f} ~ {combined_df[current_col].max():.2f} A/cm²")

                    st.divider()
                    st.subheader("📉 전압 기울기 분석")

                    sc1, sc2, sc3, sc4 = st.columns(4)
                    with sc1:
                        x_min = st.number_input(f"X 최솟값 ({time_label_ct})", value=0.0, step=0.1, key="ct_xmin")
                    with sc2:
                        x_max = st.number_input(f"X 최댓값 ({time_label_ct})", value=total_time, step=0.1, key="ct_xmax")
                    with sc3:
                        y_min = st.number_input("Y 최솟값 (V)", value=0.0, step=0.01, key="ct_ymin")
                    with sc4:
                        y_max = st.number_input("Y 최댓값 (V)", value=1.5, step=0.01, key="ct_ymax")

                    gc1, gc2 = st.columns(2)
                    with gc1:
                        t_start = st.number_input(f"기울기 구간 시작 ({time_label_ct})", min_value=0.0, max_value=total_time, value=0.0, step=0.1, key="ct_ts")
                    with gc2:
                        t_end = st.number_input(f"기울기 구간 끝 ({time_label_ct})", min_value=0.0, max_value=total_time, value=total_time, step=0.1, key="ct_te")

                    fig = go.Figure()
                    for i, part in enumerate(combined_parts):
                        fig.add_trace(go.Scatter(
                            x=part[time_col_ct], y=part[voltage_col],
                            mode="lines", name=part["_source"].iloc[0],
                            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
                        ))
                    for i, part in enumerate(combined_parts[:-1]):
                        fig.add_vline(x=float(part[time_col_ct].max()), line_dash="dot", line_color="gray", line_width=1,
                                      annotation_text=combined_parts[i+1]["_source"].iloc[0],
                                      annotation_position="top right", annotation_font_size=10)

                    if t_start < t_end:
                        mask = (combined_df[time_col_ct] >= t_start) & (combined_df[time_col_ct] <= t_end)
                        subset = combined_df.loc[mask]
                        if not subset.empty:
                            coeffs = np.polyfit(subset[time_col_ct], subset[voltage_col], 1)
                            slope, intercept = coeffs[0], coeffs[1]
                            fig.add_trace(go.Scatter(
                                x=[0.0, total_time], y=[intercept, slope * total_time + intercept],
                                mode="lines", name="Linear Fit",
                                line=dict(color="black", width=2, dash="dash"),
                            ))
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("기울기", f"{slope:.6f} V/{time_label_ct}")
                            mc2.metric("절편", f"{intercept:.4f} V")
                            mc3.metric("선형식", f"V = {slope:.6f}·t + {intercept:.4f}")

                    fig.update_layout(
                        xaxis=dict(title=time_col_ct, range=[x_min, x_max]),
                        yaxis=dict(title="Voltage (V)", range=[y_min, y_max]),
                        height=600, width=720, margin=dict(l=20, r=20, t=20, b=20),
                        legend=dict(orientation="h", y=-0.15),
                    )
                    _, center_col_ct, _ = st.columns([1, 6, 1])
                    with center_col_ct:
                        st.plotly_chart(fig, use_container_width=False)

                    st.divider()
                    st.subheader("📋 데이터 테이블")
                    fmt = {time_col_ct: "{:.2f}", current_col: "{:.2f}", voltage_col: "{:.2f}"}
                    max_rows = st.number_input("표시할 행 수", min_value=10, max_value=min(10000, len(combined_df)),
                                               value=min(500, len(combined_df)), step=10, key="ct_rows")
                    st.dataframe(combined_df[[time_col_ct, current_col, voltage_col]].head(int(max_rows)).style.format(fmt),
                                 use_container_width=True, height=400)

                    output = io.StringIO()
                    output.write("combined,,,\n")
                    output.write(f"Active area (cm2),{active_area_ct},,\n")
                    output.write(f"Current density (A cm-2),{avg_current_density},,\n")
                    output.write(f"Time ({time_label_ct}),Current Density (A/cm²),Voltage (V)\n")
                    for _, row in combined_df.iterrows():
                        output.write(f"{row[time_col_ct]:.6f},{row[current_col]:.2f},{row[voltage_col]:.2f}\n")

                    st.download_button(
                        label="⬇️ 이어붙이기 CSV 다운로드",
                        data=output.getvalue().encode("utf-8-sig"),
                        file_name=f"{today}_Longterm_이어붙이기.csv",
                        mime="text/csv",
                    )


        # ══════════════════════════════════════════════════════════════════════════════
        # IVP 변환
        # ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# IVP 정리
# ══════════════════════════════════════════════════════════════════════════════
with tab_ivp:
    st.markdown("IviumStat `.idf` 파일을 업로드하면 전류밀도·파워밀도 CSV로 변환합니다.")

    uploaded_files_ivp = st.file_uploader(
        "📂 .idf 파일 업로드 (최대 500개)",
        type=["idf"], accept_multiple_files=True, key="uploader_ivp",
    )

    col_date, col_area, col_unit = st.columns([2, 2, 1])
    with col_date:
        selected_date = st.date_input("날짜", value=_date.today(), key="ivp_date")
    with col_area:
        ivp_area = st.number_input("활성 면적 (Active Area)", min_value=0.0001,
                                   value=1.0, step=0.01, format="%.4f", key="ivp_area")
    with col_unit:
        ivp_area_unit = st.selectbox("단위", ["cm²", "mm²", "m²"], key="ivp_unit")

    unit_to_cm2  = {"cm²": 1.0, "mm²": 0.01, "m²": 10000.0}
    ivp_area_cm2 = ivp_area * unit_to_cm2[ivp_area_unit]
    ivp_date_str = selected_date.strftime("%Y%m%d")
    ivp_out_filename = f"{ivp_date_str}_IVP 정리.csv"
    st.caption(f"입력 면적: **{ivp_area} {ivp_area_unit}** = {ivp_area_cm2:.4f} cm²　|　저장 파일명: **{ivp_out_filename}**")

    st.divider()
    ivp_organize = st.button("⚙️ Organize", type="primary",
                             disabled=not uploaded_files_ivp, key="ivp_organize")

    if "ivp_processed" not in st.session_state:
        st.session_state.ivp_processed    = False
        st.session_state.ivp_all_results  = []
        st.session_state.ivp_plot_data    = []
        st.session_state.ivp_csv_bytes    = b""
        st.session_state.ivp_out_filename = ""

    if ivp_organize and uploaded_files_ivp:
        if len(uploaded_files_ivp) > 500:
            st.error("❌ 파일은 최대 500개까지 업로드 가능합니다.")
        else:
            st.info(f"📁 {len(uploaded_files_ivp)}개 파일 처리 중...")
            all_blocks, all_results, plot_data, errors = [], [], [], []

            progress = st.progress(0, text="변환 중...")
            for idx, uf in enumerate(uploaded_files_ivp):
                try:
                    file_stem = os.path.splitext(uf.name)[0]
                    raw = uf.read()
                    block, ppd, ocv, interp, df, mode = process_file_ivp(
                        raw, ivp_area_cm2, ivp_area, ivp_area_unit, file_stem)
                    all_blocks.append(block)
                    plot_data.append({"name": file_stem, "df": df, "mode": mode})
                    all_results.append({"file_stem": file_stem, "ppd": ppd, "ocv": ocv,
                                        "interp": interp, "n_points": len(df), "mode": mode})
                except Exception as e:
                    errors.append(f"{uf.name}: {e}")
                progress.progress((idx + 1) / len(uploaded_files_ivp),
                                  text=f"변환 중... ({idx+1}/{len(uploaded_files_ivp)})")

            progress.empty()
            for err in errors:
                st.error(f"❌ {err}")

            st.session_state.ivp_processed    = True
            st.session_state.ivp_all_results  = all_results
            st.session_state.ivp_plot_data    = plot_data
            st.session_state.ivp_csv_bytes    = blocks_to_csv(all_blocks)
            st.session_state.ivp_out_filename = ivp_out_filename

    if st.session_state.ivp_processed:
        all_results = st.session_state.ivp_all_results
        plot_data   = st.session_state.ivp_plot_data

        st.subheader(f"✅ 변환 완료 — {len(all_results)}개 파일")

        targets = [0.5, 1.0, 2.0, 3.0]
        summary_rows = []
        for r in all_results:
            row = {"파일명": r["file_stem"], "모드": r["mode"], "포인트 수": r["n_points"],
                   "OCV (V)": f"{r['ocv']:.6g}", "PPD (W/cm²)": f"{r['ppd']:.6g}"}
            for t in targets:
                v, _ = r["interp"][t]
                row[f"V @ {t}A (V)"] = f"{v:.6g}" if v is not None else "N/A"
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.subheader("📈 IVP 커브")
        default_x_min, default_x_max = auto_x_range(plot_data)

        with st.expander("⚙️ 축 범위 설정", expanded=False):
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            with sc1:
                x_min  = st.number_input("X min",          value=default_x_min, step=0.1, format="%.2f", key="ivp_xmin")
            with sc2:
                x_max  = st.number_input("X max",          value=default_x_max, step=0.1, format="%.2f", key="ivp_xmax")
            with sc3:
                y1_min = st.number_input("Y1 min (V)",     value=0.0, step=0.1, format="%.2f", key="ivp_y1min")
            with sc4:
                y1_max = st.number_input("Y1 max (V)",     value=1.2, step=0.1, format="%.2f", key="ivp_y1max")
            with sc5:
                y2_min = st.number_input("Y2 min (W/cm²)", value=0.0, step=0.1, format="%.2f", key="ivp_y2min")
            with sc6:
                y2_max = st.number_input("Y2 max (W/cm²)", value=3.0, step=0.1, format="%.2f", key="ivp_y2max")

        fig = make_ivp_chart(plot_data, x_min, x_max, y1_min, y1_max, y2_min, y2_max)
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            label=f"⬇️ {st.session_state.ivp_out_filename} 다운로드",
            data=st.session_state.ivp_csv_bytes,
            file_name=st.session_state.ivp_out_filename,
            mime="text/csv", key="ivp_download",
        )
    elif not uploaded_files_ivp:
        st.info(".idf 파일을 업로드하고 Organize 버튼을 누르면 변환됩니다.")


# ══════════════════════════════════════════════════════════════════════════════
# EIS 정리
# ══════════════════════════════════════════════════════════════════════════════

def make_nyquist(plot_data_eis, xmin, xmax, ymin, ymax):
    """나이키스트 플랏: Z'(a) vs -Z''(b)"""
    fig = go.Figure()
    for i, item in enumerate(plot_data_eis):
        color = COLORS[i % len(COLORS)]
        df, name = item["df"], item["name"]
        fig.add_trace(go.Scatter(
            x=df["Z'(a)"], y=-df["Z''(b)"],
            mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))
    fig.update_layout(
        xaxis=dict(title="Z' (Ω)", range=[xmin, xmax] if xmin != xmax else None,
                   showgrid=True, gridcolor="#e0e0e0",
                   zeroline=True, zerolinecolor="black", zerolinewidth=2.5),
        yaxis=dict(title="-Z'' (Ω)", range=[ymin, ymax] if ymin != ymax else None,
                   showgrid=True, gridcolor="#e0e0e0",
                   zeroline=True, zerolinecolor="black", zerolinewidth=2.5),
        plot_bgcolor="white", paper_bgcolor="white",
        height=450, margin=dict(l=60, r=40, t=40, b=60),
        legend=dict(orientation="v", x=1.01, y=1),
    )
    return fig


def make_bode(plot_data_eis, fmin, fmax, zmin, zmax):
    """보데 플랏: Freq vs -Z''(b), 로그 X축"""
    fig = go.Figure()
    for i, item in enumerate(plot_data_eis):
        color = COLORS[i % len(COLORS)]
        df, name = item["df"], item["name"]
        freq = df["Freq(Hz)"]
        z_imag = -df["Z''(b)"]
        fig.add_trace(go.Scatter(
            x=freq, y=z_imag, mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=4),
        ))
    fig.update_layout(
        xaxis=dict(title="Frequency (Hz)", type="log",
                   range=[np.log10(fmin) if fmin > 0 else None,
                          np.log10(fmax) if fmax > 0 else None],
                   showgrid=True, gridcolor="#e0e0e0",
                   zeroline=True, zerolinecolor="black", zerolinewidth=2.5),
        yaxis=dict(title="-Z'' (Ω)", range=[zmin, zmax] if zmin != zmax else None,
                   showgrid=True, gridcolor="#e0e0e0",
                   zeroline=True, zerolinecolor="black", zerolinewidth=2.5),
        plot_bgcolor="white", paper_bgcolor="white",
        height=450, margin=dict(l=60, r=40, t=40, b=60),
        legend=dict(orientation="v", x=1.01, y=1),
    )
    return fig


# ── EIS 공통 함수 ──────────────────────────────────────────────────────────────
def natural_sort_key(s):
    return re.sub(r'\d+', lambda x: x.group(0).zfill(10), s.lower())


def parse_z_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8", errors='ignore').splitlines()
        header_line = ""
        data_start_idx = 0

        header_text = "\n".join(content[:100])
        date_match = re.search(r"Date\s*[:,-]\s*([\d\-/.]+)", header_text, re.IGNORECASE)
        time_match = re.search(r"Time\s*[:,-]\s*([\d:]+)", header_text, re.IGNORECASE)

        measure_date = date_match.group(1) if date_match else "Unknown"
        measure_time = "Unknown"
        if time_match:
            time_str = time_match.group(1)
            try:
                t_obj = datetime.strptime(time_str.strip(), "%H:%M:%S")
                am_pm = "오전" if t_obj.hour < 12 else "오후"
                hour_12 = t_obj.hour if t_obj.hour <= 12 else t_obj.hour - 12
                if hour_12 == 0: hour_12 = 12
                measure_time = f"{am_pm} {hour_12:02d}:{t_obj.minute:02d}:{t_obj.second:02d}"
            except ValueError:
                measure_time = time_str

        for idx, line in enumerate(content):
            if "Freq(Hz)" in line:
                header_line = line.strip()
            if "End Comments" in line:
                data_start_idx = idx + 1
                break

        if not header_line and data_start_idx > 0:
            header_line = content[data_start_idx - 2].strip()

        headers = re.split(r'\t', header_line)
        if len(headers) < 3: headers = re.split(r'\s{2,}', header_line)
        if len(headers) < 3: headers = re.split(r'\s+', header_line)

        target_cols = ["Freq(Hz)", "Z'(a)", "Z''(b)"]
        col_indices = {}
        for target in target_cols:
            for c_idx, h in enumerate(headers):
                if target in h:
                    col_indices[target] = c_idx
                    break

        if len(col_indices) < 3:
            return None, measure_date, measure_time

        extracted_data = {col: [] for col in target_cols}
        for line in content[data_start_idx:]:
            line = line.strip()
            if not line: continue
            parts = re.split(r'\s+', line)
            try:
                for col in target_cols:
                    val = float(parts[col_indices[col]])
                    extracted_data[col].append(val)
            except (IndexError, ValueError):
                continue

        return pd.DataFrame(extracted_data), measure_date, measure_time
    except Exception:
        return None, "Unknown", "Unknown"


def analyze_eis_logic(df, filename, measure_date, measure_time):
    df_sorted = df.sort_values(by="Freq(Hz)", ascending=False).reset_index(drop=True)
    transition = (df_sorted["Z''(b)"].shift(1) > 0) & (df_sorted["Z''(b)"] < 0)

    if transition.any():
        r1_idx = transition.idxmax()
        r1_freq = df_sorted.loc[r1_idx, "Freq(Hz)"]
        r1_val  = df_sorted.loc[r1_idx, "Z'(a)"]
    else:
        neg_indices = df_sorted.index[df_sorted["Z''(b)"] < 0].tolist()
        if neg_indices:
            r1_idx = neg_indices[0]
            r1_freq = df_sorted.loc[r1_idx, "Freq(Hz)"]
            r1_val  = df_sorted.loc[r1_idx, "Z'(a)"]
        else:
            r1_freq, r1_val = np.nan, np.nan

    r2_val = df_sorted.iloc[-1]["Z'(a)"]
    rp_val = r2_val - r1_val if not pd.isna(r1_val) else np.nan

    return {"파일명": filename, "Freq(Hz)": r1_freq, "R1": r1_val, "R2": r2_val,
            "Rp": rp_val, "측정일자": measure_date, "측정시간": measure_time}


def process_correction_logic(uploaded_file, active_area, area_num):
    df, _, _ = parse_z_file(uploaded_file)
    if df is None:
        return None, "파싱 실패"

    df_sorted = df.sort_values(by="Freq(Hz)", ascending=False).reset_index(drop=True)
    transition = (df_sorted["Z''(b)"].shift(1) > 0) & (df_sorted["Z''(b)"] < 0)

    if transition.any():
        z_a_val = df_sorted.loc[transition.idxmax(), "Z'(a)"]
    else:
        neg_indices = df_sorted.index[df_sorted["Z''(b)"] < 0].tolist()
        z_a_val = df_sorted.loc[neg_indices[0], "Z'(a)"] if neg_indices else np.nan

    if pd.isna(z_a_val):
        return None, "Z'(a) 특정 불가"

    df_sorted["Z'(a)_area"]             = (df_sorted["Z'(a)"] * active_area) / area_num
    df_sorted["Z'(a)(Ohmic x)_area"]    = ((df_sorted["Z'(a)"] - z_a_val) * active_area) / area_num
    df_sorted["Z''(b)_area"]            = (df_sorted["Z''(b)"] * active_area) / area_num

    return df_sorted[["Freq(Hz)", "Z'(a)_area", "Z'(a)(Ohmic x)_area", "Z''(b)_area"]], "성공"


# ── EIS 탭 본문 ────────────────────────────────────────────────────────────────
with tab_eis:
    eis_sub_calc, eis_sub_rem, eis_sub_fit = st.tabs(["🧪 Ohmic·Rp 계산기", "⚡ 옴믹 저항 제거기", "📈 임피던스 피팅"])

    # ── 계산기 ─────────────────────────────────────────────────────────────────
    with eis_sub_calc:
        st.subheader("🧪 EIS Ohmic, Rp 계산기")
        st.write("원본 .z 파일을 업로드하시고 **Active Area**와 **Area #**를 기입하시면 즉시 연산이 완료됩니다.")

        col_upload, col_active_area, col_area_num = st.columns([2, 1, 1])
        with col_upload:
            uploaded_files_calc = st.file_uploader(
                "분석할 .z 또는 .txt 파일을 선택해 주세요",
                type=["z", "txt"], accept_multiple_files=True, key="calc_upload"
            )
        with col_active_area:
            active_area_input = st.number_input(
                "Active Area [cm²]", min_value=0.0001, value=1.0,
                step=0.1, format="%.4f", key="calc_area"
            )
        with col_area_num:
            area_num_input = st.selectbox("Area #", options=[1, 2], index=0, key="calc_num")

        if uploaded_files_calc:
            all_results = []
            zip_buffer_calc = io.BytesIO()
            progress_bar = st.progress(0)

            with zipfile.ZipFile(zip_buffer_calc, "a", zipfile.ZIP_DEFLATED) as new_zip:
                for i, f in enumerate(uploaded_files_calc):
                    df_extracted, m_date, m_time = parse_z_file(f)
                    if df_extracted is not None:
                        csv_content = df_extracted.to_csv(index=False, encoding='utf-8-sig')
                        new_zip.writestr(os.path.splitext(f.name)[0] + "_extracted.csv", csv_content)
                        all_results.append(analyze_eis_logic(df_extracted, f.name, m_date, m_time))
                    progress_bar.progress((i + 1) / len(uploaded_files_calc))

            if all_results:
                st.success("✅ 분석 완료")
                st.download_button(
                    "🎁 1단계 다운로드 (ZIP)", data=zip_buffer_calc.getvalue(),
                    file_name="extracted_csv_files.zip", mime="application/zip",
                    key="calc_zip_dl"
                )

                df_final_calc = pd.DataFrame(all_results)
                df_final_calc["R1(area)"] = (df_final_calc["R1"] * active_area_input) / area_num_input
                df_final_calc["R2(area)"] = (df_final_calc["R2"] * active_area_input) / area_num_input
                df_final_calc["Rp(area)"] = (df_final_calc["Rp"] * active_area_input) / area_num_input

                c1, c2, c3 = st.columns(3)
                search_q   = c1.text_input("🔍 파일명 검색", "", key="calc_search")
                sort_by    = c2.selectbox("🎯 정렬 기준", ["파일명", "측정일자", "측정시간", "Rp", "Freq(Hz)"], key="calc_sort_by")
                sort_order = c3.radio("↕️ 정렬 순서", ["오름차순", "내림차순"], horizontal=True, key="calc_sort_ord")

                if search_q:
                    df_final_calc = df_final_calc[df_final_calc["파일명"].str.contains(search_q, case=False)]

                if sort_by == "파일명":
                    df_final_calc["_sort_key"] = df_final_calc["파일명"].apply(natural_sort_key)
                    df_final_calc = df_final_calc.sort_values(
                        by="_sort_key", ascending=(sort_order == "오름차순")
                    ).drop(columns=["_sort_key"])
                else:
                    df_final_calc = df_final_calc.sort_values(
                        by=sort_by, ascending=(sort_order == "오름차순")
                    )

                display_cols = ["파일명", "측정일자", "측정시간", "Freq(Hz)", "R1", "R2", "Rp", "R1(area)", "R2(area)", "Rp(area)"]
                st.dataframe(df_final_calc[display_cols], use_container_width=True, hide_index=True)

                csv_report = df_final_calc[display_cols].to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    "💾 종합 보고서 다운로드 (CSV)", data=csv_report,
                    file_name=f"EIS_Analysis_Report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", key="calc_csv_dl"
                )

                # ── 나이키스트 / 보데 플랏 ────────────────────────────────────
                st.divider()
                st.subheader("📈 EIS 플랏")

                # 파일별 df 수집 (parse_z_file 재호출)
                eis_plot_data_calc = []
                for uf in uploaded_files_calc:
                    uf.seek(0)
                    df_p, _, _ = parse_z_file(uf)
                    if df_p is not None:
                        eis_plot_data_calc.append({"df": df_p, "name": os.path.splitext(uf.name)[0]})

                if eis_plot_data_calc:
                    col_ny, col_bo = st.columns(2)

                    with col_ny:
                        st.markdown("**🔵 나이키스트 플랏**")
                        with st.expander("⚙️ 축 범위 설정", expanded=False):
                            ny1, ny2 = st.columns(2)
                            ny3, ny4 = st.columns(2)
                            ny_xmin = ny1.number_input("X min (Z')", value=0.0, step=0.01, key="calc_ny_xmin")
                            ny_xmax = ny2.number_input("X max (Z')", value=0.0, step=0.01, key="calc_ny_xmax")
                            ny_ymin = ny3.number_input("Y min (-Z'')", value=0.0, step=0.01, key="calc_ny_ymin")
                            ny_ymax = ny4.number_input("Y max (-Z'')", value=0.0, step=0.01, key="calc_ny_ymax")
                        st.plotly_chart(make_nyquist(eis_plot_data_calc, ny_xmin, ny_xmax, ny_ymin, ny_ymax), use_container_width=True)

                    with col_bo:
                        st.markdown("**📊 보데 플랏**")
                        with st.expander("⚙️ 축 범위 설정", expanded=False):
                            bo1, bo2 = st.columns(2)
                            bo3, bo4 = st.columns(2)
                            bo_fmin = bo1.number_input("Freq min", value=0.1, step=0.1, key="calc_bo_fmin")
                            bo_fmax = bo2.number_input("Freq max", value=100000.0, step=1000.0, key="calc_bo_fmax")
                            bo_zmin = bo3.number_input("-Z'' min", value=0.0, step=0.01, key="calc_bo_zmin")
                            bo_zmax = bo4.number_input("-Z'' max", value=0.0, step=0.01, key="calc_bo_zmax")
                        st.plotly_chart(make_bode(eis_plot_data_calc, bo_fmin, bo_fmax, bo_zmin, bo_zmax), use_container_width=True)


    # ── 옴믹 저항 제거기 ────────────────────────────────────────────────────────
    with eis_sub_rem:
        st.subheader("⚡ EIS 옴믹 저항 제거기")
        st.write("모든 결과 열에 **Active Area** 및 **Area #** 보정이 적용되어 있습니다.")

        col_file, col_active, col_num = st.columns([2, 1, 1])
        with col_file:
            uploaded_files_rem = st.file_uploader(
                "분석할 데이터를 모두 선택하세요",
                type=["z", "txt"], accept_multiple_files=True, key="rem_upload"
            )
        with col_active:
            active_area_rem = st.number_input(
                "Active Area [cm²]", min_value=0.0001, value=0.5,
                step=0.1, format="%.4f", key="rem_area"
            )
        with col_num:
            area_num_rem = st.selectbox("Area #", options=[1, 2], index=1, key="rem_num")

        if uploaded_files_rem:
            zip_buffer_rem = io.BytesIO()
            combined_dfs   = []
            progress_bar2  = st.progress(0)

            with zipfile.ZipFile(zip_buffer_rem, "a", zipfile.ZIP_DEFLATED) as new_zip:
                for i, f in enumerate(uploaded_files_rem):
                    df_res, msg = process_correction_logic(f, active_area_rem, area_num_rem)
                    if df_res is not None:
                        csv_data = df_res.to_csv(index=False, encoding='utf-8-sig')
                        new_zip.writestr(os.path.splitext(f.name)[0] + "_processed.csv", csv_data)

                        file_name_clean = os.path.splitext(f.name)[0]
                        df_target = df_res.copy()
                        multi_cols = pd.MultiIndex.from_tuples([
                            (file_name_clean,                          "Freq(Hz)"),
                            (f"{file_name_clean}_Z'(a)_area",          "Z'(a)_area"),
                            (f"{file_name_clean}_Z'(a)(Ohmic x)_area", "Z'(a)(Ohmic x)_area"),
                            (f"{file_name_clean}_Z''(b)_area",         "Z''(b)_area"),
                        ])
                        df_target.columns = multi_cols
                        combined_dfs.extend([
                            df_target,
                            pd.DataFrame(np.nan, index=df_target.index,
                                         columns=pd.MultiIndex.from_tuples([("", "")]))
                        ])
                    else:
                        st.warning(f"'{f.name}' 처리 실패: {msg}")
                    progress_bar2.progress((i + 1) / len(uploaded_files_rem))

            if combined_dfs:
                st.success("✅ 보정 및 병합 완료")
                st.download_button(
                    "📥 개별 보정 결과 다운로드 (ZIP)", data=zip_buffer_rem.getvalue(),
                    file_name="EIS_Correction_Individual.zip", mime="application/zip",
                    key="rem_zip_dl"
                )
                df_combined = pd.concat(combined_dfs, axis=1).iloc[:, :-1]
                combined_csv = df_combined.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                st.download_button(
                    "💾 단일 CSV 다운로드", data=combined_csv,
                    file_name=f"EIS_Combined_Data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv", key="rem_csv_dl"
                )

            # ── 나이키스트 / 보데 플랏 ────────────────────────────────────────
            st.divider()
            st.subheader("📈 EIS 플랏")

            # 원본 + Ohmic 제거 데이터 각각 수집
            eis_plot_data_raw  = []   # 원본 Z'(a)_area
            eis_plot_data_corr = []   # Ohmic 제거 Z'(a)(Ohmic x)_area
            eis_plot_data_bode = []   # 보데용 (원본 주파수 기준)

            for uf in uploaded_files_rem:
                uf.seek(0)
                df_corr, msg_c = process_correction_logic(uf, active_area_rem, area_num_rem)
                if df_corr is not None:
                    stem = os.path.splitext(uf.name)[0]
                    # 원본 (Ohmic 포함)
                    df_raw_plot = pd.DataFrame({
                        "Freq(Hz)": df_corr["Freq(Hz)"],
                        "Z'(a)":   df_corr["Z'(a)_area"],
                        "Z''(b)": df_corr["Z''(b)_area"],
                    })
                    eis_plot_data_raw.append({"df": df_raw_plot.rename(columns={"Freq(Hz)":"Freq","Z'(a)":"Zr","Z''(b)":"Zi"}), "name": f"{stem} (원본)"})
                    # Ohmic 제거
                    df_corr_plot = pd.DataFrame({
                        "Freq(Hz)": df_corr["Freq(Hz)"],
                        "Z'(a)":   df_corr["Z'(a)(Ohmic x)_area"],
                        "Z''(b)": df_corr["Z''(b)_area"],
                    })
                    eis_plot_data_corr.append({"df": df_corr_plot.rename(columns={"Freq(Hz)":"Freq","Z'(a)":"Zr","Z''(b)":"Zi"}), "name": f"{stem} (Ohmic 제거)"})

            if eis_plot_data_raw:
                with st.expander("⚙️ 나이키스트 축 범위", expanded=False):
                    ny1, ny2, ny3, ny4 = st.columns(4)
                    ny_xmin2 = ny1.number_input("X min (Z')", value=0.0, step=0.01, key="rem_ny_xmin")
                    ny_xmax2 = ny2.number_input("X max (Z')", value=0.0, step=0.01, key="rem_ny_xmax")
                    ny_ymin2 = ny3.number_input("Y min (-Z'')", value=0.0, step=0.01, key="rem_ny_ymin")
                    ny_ymax2 = ny4.number_input("Y max (-Z'')", value=0.0, step=0.01, key="rem_ny_ymax")

                col_ny_raw, col_ny_corr = st.columns(2)

                def _nyquist_from_zr_zi(data_list, xmin, xmax, ymin, ymax):
                    """Zr/Zi 컬럼 기반 나이키스트 플랏"""
                    import plotly.graph_objects as _go
                    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                              "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
                    fig = _go.Figure()
                    for i, item in enumerate(data_list):
                        df_i, name = item["df"], item["name"]
                        fig.add_trace(_go.Scatter(
                            x=df_i["Zr"], y=-df_i["Zi"],
                            mode="lines+markers", name=name,
                            line=dict(color=colors[i % len(colors)], width=2),
                            marker=dict(size=4),
                        ))
                    fig.update_layout(
                        xaxis=dict(title="Z' (Ω·cm²)",
                                   range=[xmin, xmax] if xmin != xmax else None,
                                   showgrid=True, gridcolor="#ebebeb",
                                   zeroline=True, zerolinecolor="#333", zerolinewidth=2.5),
                        yaxis=dict(title="-Z'' (Ω·cm²)",
                                   range=[ymin, ymax] if ymin != ymax else None,
                                   showgrid=True, gridcolor="#ebebeb",
                                   zeroline=True, zerolinecolor="#333", zerolinewidth=2.5),
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=380, margin=dict(l=55, r=10, t=30, b=50),
                        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)", font=dict(size=10)),
                    )
                    return fig

                with col_ny_raw:
                    st.markdown("**🔵 나이키스트 — 원본 (Ohmic 포함)**")
                    st.plotly_chart(
                        _nyquist_from_zr_zi(eis_plot_data_raw, ny_xmin2, ny_xmax2, ny_ymin2, ny_ymax2),
                        use_container_width=True
                    )

                with col_ny_corr:
                    st.markdown("**🔴 나이키스트 — Ohmic 저항 제거**")
                    st.plotly_chart(
                        _nyquist_from_zr_zi(eis_plot_data_corr, ny_xmin2, ny_xmax2, ny_ymin2, ny_ymax2),
                        use_container_width=True
                    )

                st.markdown("**📊 보데 플랏**")
                with st.expander("⚙️ 보데 축 범위", expanded=False):
                    bo1, bo2, bo3, bo4 = st.columns(4)
                    bo_fmin2 = bo1.number_input("Freq min", value=0.1, step=0.1, key="rem_bo_fmin")
                    bo_fmax2 = bo2.number_input("Freq max", value=100000.0, step=1000.0, key="rem_bo_fmax")
                    bo_zmin2 = bo3.number_input("-Z'' min", value=0.0, step=0.01, key="rem_bo_zmin")
                    bo_zmax2 = bo4.number_input("-Z'' max", value=0.0, step=0.01, key="rem_bo_zmax")

                import plotly.graph_objects as _go2
                colors_b = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                            "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
                fig_b = _go2.Figure()
                for i, item in enumerate(eis_plot_data_raw + eis_plot_data_corr):
                    df_i, name = item["df"], item["name"]
                    dash = "solid" if "원본" in name else "dash"
                    fig_b.add_trace(_go2.Scatter(
                        x=df_i["Freq"], y=-df_i["Zi"],
                        mode="lines+markers", name=name,
                        line=dict(color=colors_b[i % len(colors_b)], width=2, dash=dash),
                        marker=dict(size=4),
                    ))
                log_range = [np.log10(max(bo_fmin2, 1e-9)), np.log10(max(bo_fmax2, 1e-9))]
                fig_b.update_layout(
                    xaxis=dict(title="Frequency (Hz)", type="log", range=log_range,
                               showgrid=True, gridcolor="#ebebeb",
                               zeroline=True, zerolinecolor="#333", zerolinewidth=2.5),
                    yaxis=dict(title="-Z'' (Ω·cm²)",
                               range=[bo_zmin2, bo_zmax2] if bo_zmin2 != bo_zmax2 else None,
                               showgrid=True, gridcolor="#ebebeb",
                               zeroline=True, zerolinecolor="#333", zerolinewidth=2.5),
                    plot_bgcolor="white", paper_bgcolor="white",
                    height=380, margin=dict(l=55, r=10, t=30, b=50),
                    legend=dict(x=0.99, y=0.99, xanchor="right",
                                bgcolor="rgba(255,255,255,0.85)", font=dict(size=10)),
                )
                st.plotly_chart(fig_b, use_container_width=True)

with tab_eis:
    with eis_sub_fit:
        st.subheader("📈 EIS 임피던스 피팅")
        st.caption("회로 모델: L — Rs — (R1‖CPE1) — ... — (Rn‖CPEn)")
        eis_fitting_tab()
