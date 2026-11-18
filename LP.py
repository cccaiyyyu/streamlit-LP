import json
import os
import time
import urllib.request
import urllib.error
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


def fmt_num(val):
    if abs(val) < 1e-6:
        return "0"
    return f"{int(round(val))}" if abs(val - round(val)) < 1e-5 else f"{val:.2f}"


st.set_page_config(
    page_title="線性規劃與作業研究決策支援系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.8rem; padding-bottom: 6rem; max-width: 1150px; }
    h1, h2, h3, h4 { font-weight: 600 !important; color: #0f172a !important; letter-spacing: -0.02em; }
    .stMetric { background: #f8fafc; border: 1px solid #e2e8f0; padding: 14px 18px; border-radius: 8px; }
    [data-testid="stBottom"] {
        background: linear-gradient(to top, #ffffff 80%, rgba(255,255,255,0)) !important;
        padding-bottom: 12px !important;
    }
    [data-testid="stChatInput"] {
        max-width: 1100px !important;
        margin: 0 auto !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08) !important;
    }
    [data-testid="stChatInput"] > div {
        background-color: #ffffff !important;
    }
    [data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInputSubmitButton"]:disabled {
        background-color: #ff4b4b !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInputSubmitButton"]:hover {
        background-color: #ff3333 !important;
    }
    [data-testid="stChatInputSubmitButton"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


def get_gemini_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return str(key).strip()
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "").strip()


def call_gemini_text(api_key, system_prompt, contents, temperature=0.25):
    candidate_models = ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash-lite"]
    last_err = None
    for m_name in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m_name}:generateContent?key={api_key}"
        gen_config = {
            "temperature": temperature,
            "maxOutputTokens": 4096
        }
        if "3.8-flash" in m_name and "lite" not in m_name:
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}

        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": gen_config
        }
        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=35) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
    raise last_err


def generate_ai_management_insight(api_key, context_summary):
    sys_prompt = (
        "你是一位精通作業研究（Operations Research）與管理科學（Management Science）的資深決策顧問。\n"
        "請一律使用『繁體中文（台灣慣用語）』回覆，語氣專業、清晰、具體且具商業洞察力。\n"
        "請根據系統提供給你的「線性規劃／整數規劃數學模型」與「SciPy HiGHS 精確求解結果（包含最佳決策變數、最佳目標值 Z*、各限制式左右側數值、鬆弛／剩餘量、緊束瓶頸狀態與影子價格）」，撰寫一份結構完整的『管理意涵與敏感度決策解讀』：\n"
        "1. **最佳決策方案與資源配置建議**：具體說明每個決策變數應該執行多少單位（若為 0-1 規劃則說明選取與放棄哪些項目），以及預期能達到的最高利潤／最低成本總額（Z*）。\n"
        "2. **瓶頸限制式與影子價格（Shadow Price）評估**：指出哪些限制式是緊束瓶頸（Binding，鬆弛量為 0）。若有影子價格（連續型 LP），請白話解釋『該資源每增加 1 單位，目標值 Z 會增加或減少多少』，並建議管理者是否值得花額外成本擴充該項瓶頸資源（同時提醒影子價格僅在允許變動範圍 Allowable Range 內有效）；若為整數／0-1 規劃無影子價格，則分析哪項資源限制最緊迫、成為阻礙目標進一步提升的主因。\n"
        "3. **非緊束限制式（Nonbinding）與餘裕資源處置**：指出哪些資源還有剩餘（鬆弛量 > 0），並建議管理者如何避免資源閒置或進行內部調度。\n"
        "請完整輸出上述三點分析，不要重複貼出原始 JSON，也不要使用任何簡體字。"
    )
    contents = [{"role": "user", "parts": [{"text": context_summary}]}]
    return call_gemini_text(api_key, sys_prompt, contents, temperature=0.3)


def get_2d_feasible_vertices(obj_c, constraints_input, non_neg=True, is_binary=False):
    all_ineqs = [{'a': [float(c['a'][0]), float(c['a'][1])], 'op': c['op'], 'rhs': float(c['rhs'])} for c in constraints_input]
    if is_binary:
        all_ineqs.extend([
            {'a': [1.0, 0.0], 'op': '>=', 'rhs': 0.0}, {'a': [1.0, 0.0], 'op': '<=', 'rhs': 1.0},
            {'a': [0.0, 1.0], 'op': '>=', 'rhs': 0.0}, {'a': [0.0, 1.0], 'op': '<=', 'rhs': 1.0}
        ])
    elif non_neg:
        all_ineqs.extend([{'a': [1.0, 0.0], 'op': '>=', 'rhs': 0.0}, {'a': [0.0, 1.0], 'op': '>=', 'rhs': 0.0}])

    cand_pts = []
    for i in range(len(all_ineqs)):
        for j in range(i + 1, len(all_ineqs)):
            try:
                pt = np.linalg.solve([all_ineqs[i]['a'], all_ineqs[j]['a']], [all_ineqs[i]['rhs'], all_ineqs[j]['rhs']])
                if all(
                    (ineq['op'] == '<=' and ineq['a'][0]*pt[0] + ineq['a'][1]*pt[1] <= ineq['rhs'] + 1e-4) or
                    (ineq['op'] == '>=' and ineq['a'][0]*pt[0] + ineq['a'][1]*pt[1] >= ineq['rhs'] - 1e-4) or
                    (ineq['op'] == '=' and abs(ineq['a'][0]*pt[0] + ineq['a'][1]*pt[1] - ineq['rhs']) <= 1e-4)
                    for ineq in all_ineqs
                ):
                    pt = np.where(np.abs(pt) < 1e-6, 0.0, pt)
                    cand_pts.append(pt)
            except np.linalg.LinAlgError:
                continue

    if not cand_pts:
        return []
    unique_pts = np.unique(np.array(cand_pts).round(decimals=4), axis=0)
    vertices = []
    for pt in unique_pts:
        z_val = float(obj_c[0]) * pt[0] + float(obj_c[1]) * pt[1]
        z_val = 0.0 if abs(z_val) < 1e-6 else z_val
        vertices.append((float(pt[0]), float(pt[1]), z_val))
    return vertices


def render_2d_vertices_table(obj_c, constraints_input, sol, opt_type, non_neg=True, is_int=False, is_binary=False, sec_prefix="7."):
    st.subheader(f"{sec_prefix} 可行解頂點目標值分析 (Corner Point Analysis)")
    if is_binary:
        st.caption("列出所有滿足限制式之 0-1 可行候選組合與目標函數 Z 值")
        pts = []
        for x1 in [0.0, 1.0]:
            for x2 in [0.0, 1.0]:
                if all(
                    (c['op'] == '<=' and c['a'][0]*x1 + c['a'][1]*x2 <= c['rhs'] + 1e-4) or
                    (c['op'] == '>=' and c['a'][0]*x1 + c['a'][1]*x2 >= c['rhs'] - 1e-4) or
                    (c['op'] == '=' and abs(c['a'][0]*x1 + c['a'][1]*x2 - c['rhs']) <= 1e-4)
                    for c in constraints_input
                ):
                    z_val = float(obj_c[0])*x1 + float(obj_c[1])*x2
                    pts.append((x1, x2, 0.0 if abs(z_val) < 1e-6 else z_val))
    else:
        st.caption("列出線性規劃可行多邊形各角點（頂點）及其對應目標函數 Z 值" + ("（整數規劃之最佳解可能位於可行域內部整數格點）" if is_int else ""))
        pts = get_2d_feasible_vertices(obj_c, constraints_input, non_neg, is_binary)

    if not pts:
        st.info("目前條件下無封閉頂點可列舉。")
        return

    pts.sort(key=lambda item: item[2], reverse=(opt_type == 'Max'))

    rows = []
    opt_found_in_vertices = False
    for idx, (vx, vy, vz) in enumerate(pts, start=1):
        is_opt = (abs(vx - sol[0]) < 1e-3 and abs(vy - sol[1]) < 1e-3)
        if is_opt:
            opt_found_in_vertices = True
        rows.append({
            "#": idx,
            "決策變數 x₁": fmt_num(vx),
            "決策變數 x₂": fmt_num(vy),
            "目標函數值 (Z)": fmt_num(vz),
            "備註": "★ 最佳解" if is_opt else ""
        })

    if is_int and not is_binary and not opt_found_in_vertices:
        rows.insert(0, {
            "#": "IP*",
            "決策變數 x₁": fmt_num(sol[0]),
            "決策變數 x₂": fmt_num(sol[1]),
            "目標函數值 (Z)": fmt_num(float(obj_c[0])*sol[0] + float(obj_c[1])*sol[1]),
            "備註": "★ 整數最佳解 (內部格點)"
        })

    df_v = pd.DataFrame(rows)

    def highlight_best(row):
        if "★" in str(row["備註"]):
            return ["background-color: #fef3c7; font-weight: 600; color: #92400e;"] * len(row)
        return [""] * len(row)

    st.dataframe(df_v.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)


def solve_lp_highs(opt_type, c, constraints, non_neg=True, is_int=False, is_binary=False):
    n = len(c)
    c_factor = -1.0 if opt_type == 'Max' else 1.0
    c_arr = np.array(c, dtype=float) * c_factor

    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for con in constraints:
        a = list(map(float, con['a']))
        rhs = float(con['rhs'])
        if con['op'] == '<=':
            A_ub.append(a); b_ub.append(rhs)
        elif con['op'] == '>=':
            A_ub.append([-v for v in a]); b_ub.append(-rhs)
        elif con['op'] == '=':
            A_eq.append(a); b_eq.append(rhs)

    if is_binary:
        bounds = [(0, 1) for _ in range(n)]
        integrality = [1] * n
    else:
        bounds = [(0, None) if non_neg else (None, None) for _ in range(n)]
        integrality = [1] * n if is_int else None

    has_discrete = is_int or is_binary
    res = linprog(
        c=c_arr,
        A_ub=A_ub or None, b_ub=b_ub or None,
        A_eq=A_eq or None, b_eq=b_eq or None,
        bounds=bounds, integrality=integrality, method='highs'
    )

    if not res.success:
        return False, None, res.message

    shadow_prices = []
    if not has_discrete:
        ub_idx = eq_idx = 0
        dual_ub = res.ineqlin.marginals if getattr(res, 'ineqlin', None) is not None else []
        dual_eq = res.eqlin.marginals if getattr(res, 'eqlin', None) is not None else []

        for con in constraints:
            if con['op'] == '<=':
                val = dual_ub[ub_idx] if len(dual_ub) > ub_idx else 0.0
                shadow_prices.append(abs(val) if opt_type == 'Max' else -abs(val))
                ub_idx += 1
            elif con['op'] == '>=':
                val = dual_ub[ub_idx] if len(dual_ub) > ub_idx else 0.0
                shadow_prices.append(-abs(val) if opt_type == 'Max' else abs(val))
                ub_idx += 1
            elif con['op'] == '=':
                val = dual_eq[eq_idx] if len(dual_eq) > eq_idx else 0.0
                shadow_prices.append(-val if opt_type == 'Max' else val)
                eq_idx += 1
    else:
        shadow_prices = [None] * len(constraints)

    return True, {'x': res.x, 'z': res.fun * c_factor, 'shadow_prices': shadow_prices}, ""


def render_interactive_2d_plot(obj_c, constraints_input, sol, best_z, non_neg, is_int=False, is_binary=False):
    c_x1, c_x2 = float(obj_c[0]), float(obj_c[1])

    true_vertices = get_2d_feasible_vertices(obj_c, constraints_input, non_neg, is_binary)
    true_vertex_coords = [[v[0], v[1]] for v in true_vertices]

    all_lines = [{'a': [float(c['a'][0]), float(c['a'][1])], 'rhs': float(c['rhs'])} for c in constraints_input]
    all_lines.extend([{'a': [1.0, 0.0], 'rhs': 0.0}, {'a': [0.0, 1.0], 'rhs': 0.0}])
    ref_xs = [0.0, float(sol[0])] + [v[0] for v in true_vertices]
    ref_ys = [0.0, float(sol[1])] + [v[1] for v in true_vertices]

    for i in range(len(all_lines)):
        for j in range(i + 1, len(all_lines)):
            try:
                pt = np.linalg.solve([all_lines[i]['a'], all_lines[j]['a']], [all_lines[i]['rhs'], all_lines[j]['rhs']])
                if np.isfinite(pt).all() and abs(pt[0]) < 1e5 and abs(pt[1]) < 1e5:
                    ref_xs.append(float(pt[0]))
                    ref_ys.append(float(pt[1]))
            except np.linalg.LinAlgError:
                continue

    if is_binary:
        min_x, max_x = -0.5, 1.8
        min_y, max_y = -0.5, 1.8
    else:
        raw_min_x, raw_max_x = min(ref_xs), max(ref_xs)
        raw_min_y, raw_max_y = min(ref_ys), max(ref_ys)
        span_x = max(6.0, raw_max_x - raw_min_x)
        span_y = max(6.0, raw_max_y - raw_min_y)
        pad_x = max(1.5, span_x * 0.22)
        pad_y = max(1.5, span_y * 0.22)

        min_x = float(np.floor((raw_min_x - pad_x) if (not non_neg or raw_min_x < -1e-4) else -min(2.5, span_x * 0.18)))
        max_x = float(np.ceil(raw_max_x + pad_x))
        min_y = float(np.floor((raw_min_y - pad_y) if (not non_neg or raw_min_y < -1e-4) else -min(2.5, span_y * 0.18)))
        max_y = float(np.ceil(raw_max_y + pad_y))

    clip_ineqs = [{'a': [float(c['a'][0]), float(c['a'][1])], 'op': c['op'], 'rhs': float(c['rhs'])} for c in constraints_input]
    if is_binary:
        clip_ineqs.extend([
            {'a': [1.0, 0.0], 'op': '>=', 'rhs': 0.0}, {'a': [1.0, 0.0], 'op': '<=', 'rhs': 1.0},
            {'a': [0.0, 1.0], 'op': '>=', 'rhs': 0.0}, {'a': [0.0, 1.0], 'op': '<=', 'rhs': 1.0}
        ])
    elif non_neg:
        clip_ineqs.extend([{'a': [1.0, 0.0], 'op': '>=', 'rhs': 0.0}, {'a': [0.0, 1.0], 'op': '>=', 'rhs': 0.0}])
    clip_ineqs.extend([
        {'a': [1.0, 0.0], 'op': '>=', 'rhs': min_x}, {'a': [1.0, 0.0], 'op': '<=', 'rhs': max_x},
        {'a': [0.0, 1.0], 'op': '>=', 'rhs': min_y}, {'a': [0.0, 1.0], 'op': '<=', 'rhs': max_y}
    ])

    poly_pts = []
    for i in range(len(clip_ineqs)):
        for j in range(i + 1, len(clip_ineqs)):
            try:
                pt = np.linalg.solve([clip_ineqs[i]['a'], clip_ineqs[j]['a']], [clip_ineqs[i]['rhs'], clip_ineqs[j]['rhs']])
                if all(
                    (ineq['op'] == '<=' and ineq['a'][0]*pt[0] + ineq['a'][1]*pt[1] <= ineq['rhs'] + 1e-4) or
                    (ineq['op'] == '>=' and ineq['a'][0]*pt[0] + ineq['a'][1]*pt[1] >= ineq['rhs'] - 1e-4) or
                    (ineq['op'] == '=' and abs(ineq['a'][0]*pt[0] + ineq['a'][1]*pt[1] - ineq['rhs']) <= 1e-4)
                    for ineq in clip_ineqs
                ):
                    pt = np.where(np.abs(pt) < 1e-6, 0.0, pt)
                    poly_pts.append(pt)
            except np.linalg.LinAlgError:
                continue

    hull_coords = []
    if poly_pts:
        poly_pts = np.unique(np.array(poly_pts).round(decimals=4), axis=0)
        if len(poly_pts) >= 3:
            try:
                hull_coords = poly_pts[ConvexHull(poly_pts).vertices].tolist()
            except Exception:
                hull_coords = poly_pts.tolist()
        else:
            hull_coords = poly_pts.tolist()

    colors = ["#8b5cf6", "#ef4444", "#f59e0b", "#0ea5e9", "#10b981", "#ec4899", "#6366f1", "#14b8a6", "#f97316", "#84cc16"]
    cons_payload = [{
        "a1": float(c['a'][0]), "a2": float(c['a'][1]), "rhs": float(c['rhs']), "op": c['op'],
        "color": colors[k % len(colors)],
        "label": f"({k+1}) {fmt_num(c['a'][0])}x₁ + {fmt_num(c['a'][1])}x₂ {c['op']} {fmt_num(c['rhs'])}"
    } for k, c in enumerate(constraints_input)]

    has_discrete = bool(is_int or is_binary)
    payload = {
        "minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y,
        "c1": c_x1, "c2": c_x2,
        "optX": float(0.0 if abs(sol[0]) < 1e-6 else sol[0]),
        "optY": float(0.0 if abs(sol[1]) < 1e-6 else sol[1]),
        "optZ": float(0.0 if abs(best_z) < 1e-6 else best_z),
        "hull": hull_coords, "vertices": true_vertex_coords,
        "constraints": cons_payload, "isInt": has_discrete, "isBinary": bool(is_binary)
    }

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8">
    <style>
        body {{ font-family: system-ui, -apple-system, "Noto Sans TC", sans-serif; margin: 0; padding: 6px; color: #0f172a; }}
        .bar {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 10px 16px; margin-bottom: 10px; gap: 10px; }}
        .eq {{ font-size: 16px; font-weight: 600; }}
        .hl-z {{ color: #7e22ce; font-weight: 700; background: #f3e8ff; padding: 2px 8px; border-radius: 6px; }}
        .hl-p {{ color: #dc2626; font-weight: 700; background: #fee2e2; padding: 2px 8px; border-radius: 6px; }}
        .ctrl {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}
        button {{ background: #2563eb; color: #fff; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
        button:hover {{ background: #1d4ed8; }}
        .wrap {{ display: flex; gap: 14px; align-items: flex-start; }}
        canvas {{ border: 1px solid #e2e8f0; border-radius: 8px; background: #fafafa; }}
        .legend {{ flex: 1; min-width: 225px; max-height: 480px; overflow-y: auto; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; font-size: 13px; line-height: 1.85; }}
        .item {{ display: flex; align-items: center; gap: 8px; }}
        .sw {{ width: 16px; height: 3px; display: inline-block; flex-shrink: 0; }}
    </style>
    </head>
    <body>
        <div class="bar">
            <div class="eq" id="eqBox"></div>
            <div class="ctrl">
                <label><input type="checkbox" id="showV" checked> 顯示頂點座標</label>
                <label><input type="checkbox" id="snap" {"checked" if has_discrete else ""}> {"吸附 0-1 格點" if is_binary else "吸附整數格點"}</label>
                <button id="reset">回到最佳解 (Z*)</button>
            </div>
        </div>
        <div class="wrap">
            <canvas id="cv" width="760" height="480"></canvas>
            <div class="legend" id="lg"></div>
        </div>
    <script>
        const D = {json.dumps(payload)};
        const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
        const eqBox = document.getElementById("eqBox"), lg = document.getElementById("lg");
        const snap = document.getElementById("snap"), showV = document.getElementById("showV"), reset = document.getElementById("reset");
        const pad = {{ l: 42, r: 35, t: 32, b: 36 }}, W = cv.width - pad.l - pad.r, H = cv.height - pad.t - pad.b;

        let sx = D.optX, sy = D.optY, cz = D.optZ, drag = null;
        const fmt = v => Math.abs(v) < 1e-6 ? "0" : (Math.abs(v - Math.round(v)) < 1e-4 ? String(Math.round(v)) : v.toFixed(2));
        const fmt2 = v => Math.abs(v) < 1e-6 ? "0.00" : v.toFixed(2);
        const pxX = x => pad.l + ((x - D.minX) / (D.maxX - D.minX)) * W;
        const pxY = y => pad.t + H - ((y - D.minY) / (D.maxY - D.minY)) * H;
        const dtX = px => D.minX + ((px - pad.l) / W) * (D.maxX - D.minX);
        const dtY = py => D.minY + ((pad.t + H - py) / H) * (D.maxY - D.minY);

        function clampCoord(x, y) {{
            if (D.isBinary && snap.checked) {{
                return [Math.max(0, Math.min(1, Math.round(x))), Math.max(0, Math.min(1, Math.round(y)))];
            }}
            let nx = Math.max(D.minX, Math.min(D.maxX, x));
            let ny = Math.max(D.minY, Math.min(D.maxY, y));
            if (snap.checked) {{ nx = Math.round(nx); ny = Math.round(ny); }}
            if (Math.abs(nx) < 1e-6) nx = 0;
            if (Math.abs(ny) < 1e-6) ny = 0;
            return [nx, ny];
        }}

        function clipLine(a1, a2, rhs) {{
            const pts = [], eps = 1e-7;
            if (Math.abs(a2) > eps) {{
                let y0 = (rhs - a1 * D.minX) / a2;
                let y1 = (rhs - a1 * D.maxX) / a2;
                if (y0 >= D.minY - eps && y0 <= D.maxY + eps) pts.push([D.minX, Math.max(D.minY, Math.min(D.maxY, y0))]);
                if (y1 >= D.minY - eps && y1 <= D.maxY + eps) pts.push([D.maxX, Math.max(D.minY, Math.min(D.maxY, y1))]);
            }}
            if (Math.abs(a1) > eps) {{
                let x0 = (rhs - a2 * D.minY) / a1;
                let x1 = (rhs - a2 * D.maxY) / a1;
                if (x0 >= D.minX - eps && x0 <= D.maxX + eps) pts.push([Math.max(D.minX, Math.min(D.maxX, x0)), D.minY]);
                if (x1 >= D.minX - eps && x1 <= D.maxX + eps) pts.push([Math.max(D.minX, Math.min(D.maxX, x1)), D.maxY]);
            }}
            const u = [];
            pts.forEach(p => {{ if (!u.some(q => Math.hypot(q[0]-p[0], q[1]-p[1]) < 1e-4)) u.push(p); }});
            return u.slice(0, 2);
        }}

        function drawArrow(x1, y1, x2, y2, color) {{
            const head = 8, ang = Math.atan2(y2 - y1, x2 - x1);
            ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
            ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x2, y2);
            ctx.lineTo(x2 - head * Math.cos(ang - Math.PI / 6), y2 - head * Math.sin(ang - Math.PI / 6));
            ctx.lineTo(x2 - head * Math.cos(ang + Math.PI / 6), y2 - head * Math.sin(ang + Math.PI / 6));
            ctx.closePath(); ctx.fillStyle = color; ctx.fill();
        }}

        function drawStar(cx, cy) {{
            let rot = Math.PI * 1.5, step = Math.PI / 5;
            ctx.beginPath(); ctx.moveTo(cx, cy - 11);
            for (let i = 0; i < 5; i++) {{
                ctx.lineTo(cx + Math.cos(rot) * 11, cy + Math.sin(rot) * 11); rot += step;
                ctx.lineTo(cx + Math.cos(rot) * 5, cy + Math.sin(rot) * 5); rot += step;
            }}
            ctx.closePath(); ctx.fillStyle = "#ef4444"; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = "#991b1b"; ctx.stroke();
        }}

        function draw() {{
            ctx.clearRect(0, 0, cv.width, cv.height);
            const xTicks = 8, yTicks = 6;
            const originX = pxX(Math.max(D.minX, Math.min(D.maxX, 0)));
            const originY = pxY(Math.max(D.minY, Math.min(D.maxY, 0)));

            ctx.strokeStyle = "#e2e8f0"; ctx.lineWidth = 1;
            ctx.font = "11px sans-serif"; ctx.fillStyle = "#64748b";
            for (let i = 0; i <= xTicks; i++) {{
                const vx = D.minX + ((D.maxX - D.minX) / xTicks) * i;
                const gx = pxX(vx);
                ctx.beginPath(); ctx.moveTo(gx, pad.t); ctx.lineTo(gx, pad.t + H); ctx.stroke();
                ctx.textAlign = "center";
                ctx.fillText(vx.toFixed(1), gx, Math.min(cv.height - 16, originY + 16));
            }}
            for (let i = 0; i <= yTicks; i++) {{
                const vy = D.minY + ((D.maxY - D.minY) / yTicks) * i;
                const gy = pxY(vy);
                ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(pad.l + W, gy); ctx.stroke();
                ctx.textAlign = "right";
                if (Math.abs(vy) > 1e-3) ctx.fillText(vy.toFixed(1), Math.max(36, originX - 8), gy + 4);
            }}

            if (D.minY <= 0 && D.maxY >= 0) {{
                drawArrow(pad.l, originY, pad.l + W + 10, originY, "#1e293b");
                ctx.fillStyle = "#1e293b"; ctx.font = "bold 14px sans-serif"; ctx.textAlign = "left";
                ctx.fillText("x₁", pad.l + W + 15, originY + 5);
            }}
            if (D.minX <= 0 && D.maxX >= 0) {{
                drawArrow(originX, pad.t + H, originX, pad.t - 10, "#0f766e");
                ctx.fillStyle = "#0f766e"; ctx.font = "bold 14px sans-serif"; ctx.textAlign = "center";
                ctx.fillText("x₂", originX, pad.t - 16);
            }}

            if (D.hull.length >= 3) {{
                ctx.beginPath();
                D.hull.forEach((p, i) => i ? ctx.lineTo(pxX(p[0]), pxY(p[1])) : ctx.moveTo(pxX(p[0]), pxY(p[1])));
                ctx.closePath();
                const grad = ctx.createLinearGradient(pad.l, pad.t, pad.l + W, pad.t + H);
                grad.addColorStop(0, "rgba(20, 184, 166, 0.28)");
                grad.addColorStop(1, "rgba(59, 130, 246, 0.18)");
                ctx.fillStyle = grad; ctx.fill();
            }} else if (D.hull.length === 2) {{
                ctx.beginPath();
                ctx.moveTo(pxX(D.hull[0][0]), pxY(D.hull[0][1]));
                ctx.lineTo(pxX(D.hull[1][0]), pxY(D.hull[1][1]));
                ctx.strokeStyle = "rgba(20, 184, 166, 0.65)"; ctx.lineWidth = 6; ctx.stroke();
            }}

            if (D.isInt && (D.maxX - D.minX) <= 35 && (D.maxY - D.minY) <= 35) {{
                const gx0 = D.isBinary ? 0 : Math.ceil(D.minX), gx1 = D.isBinary ? 1 : Math.floor(D.maxX);
                const gy0 = D.isBinary ? 0 : Math.ceil(D.minY), gy1 = D.isBinary ? 1 : Math.floor(D.maxY);
                ctx.fillStyle = D.isBinary ? "#0f172a" : "rgba(100, 116, 139, 0.35)";
                for (let ix = gx0; ix <= gx1; ix++) {{
                    for (let iy = gy0; iy <= gy1; iy++) {{
                        ctx.beginPath(); ctx.arc(pxX(ix), pxY(iy), D.isBinary ? 4 : 2, 0, Math.PI * 2); ctx.fill();
                    }}
                }}
            }}

            D.constraints.forEach(c => {{
                const s = clipLine(c.a1, c.a2, c.rhs);
                if (s.length === 2) {{
                    ctx.beginPath();
                    ctx.moveTo(pxX(s[0][0]), pxY(s[0][1]));
                    ctx.lineTo(pxX(s[1][0]), pxY(s[1][1]));
                    ctx.strokeStyle = c.color;
                    ctx.lineWidth = 2.4;
                    ctx.stroke();
                }}
            }});

            if (showV.checked && D.vertices) {{
                D.vertices.forEach(v => {{
                    const vx = pxX(v[0]), vy = pxY(v[1]);
                    ctx.beginPath(); ctx.arc(vx, vy, 5, 0, Math.PI * 2);
                    ctx.fillStyle = "#dc2626"; ctx.fill();
                    ctx.lineWidth = 1.5; ctx.strokeStyle = "#ffffff"; ctx.stroke();

                    ctx.fillStyle = "#dc2626";
                    ctx.font = "bold 12px sans-serif";
                    ctx.textAlign = "left";
                    ctx.fillText(`(${{fmt2(v[0])}}, ${{fmt2(v[1])}})`, vx + 8, vy - 6);
                }});
            }}

            const iso = clipLine(D.c1, D.c2, cz);
            if (iso.length === 2) {{
                ctx.save(); ctx.setLineDash([7, 5]); ctx.beginPath();
                ctx.moveTo(pxX(iso[0][0]), pxY(iso[0][1])); ctx.lineTo(pxX(iso[1][0]), pxY(iso[1][1]));
                ctx.strokeStyle = "#7e22ce"; ctx.lineWidth = 2.5; ctx.stroke(); ctx.restore();
                ctx.fillStyle = "#6b21a8"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "left";
                ctx.fillText(`Z = ${{fmt(cz)}}`, (pxX(iso[0][0]) + pxX(iso[1][0])) / 2 + 6, (pxY(iso[0][1]) + pxY(iso[1][1])) / 2 - 6);
            }}

            const spx = pxX(sx), spy = pxY(sy);
            drawStar(spx, spy);
            if (!showV.checked) {{
                ctx.fillStyle = "#b91c1c"; ctx.font = "bold 12px sans-serif"; ctx.textAlign = "left";
                ctx.fillText(`(${{fmt(sx)}}, ${{fmt(sy)}})`, spx + 13, spy - 8);
            }}

            const c2s = D.c2 >= 0 ? `+ ${{fmt(D.c2)}}` : `- ${{fmt(Math.abs(D.c2))}}`;
            const isOpt = Math.hypot(sx - D.optX, sy - D.optY) < 1e-2 && Math.abs(cz - D.optZ) < 1e-2;
            const optLabel = D.isBinary ? "(0-1 最佳解 Z*)" : (D.isInt ? "(整數最佳解 Z*)" : "(最佳解 Z*)");
            eqBox.innerHTML = `等利潤線：<b>Z = ${{fmt(D.c1)}}x₁ ${{c2s}}x₂ = <span class="hl-z">${{fmt(cz)}}</span></b>${{isOpt ? ` <span style='color:#16a34a;font-size:13px;'>${{optLabel}}</span>` : ""}} &nbsp;|&nbsp; 座標：<span class="hl-p">(${{fmt(sx)}}, ${{fmt(sy)}})</span>`;
        }}

        function lineDist(mx, my) {{
            const iso = clipLine(D.c1, D.c2, cz);
            if (iso.length < 2) return 9999;
            const x1 = pxX(iso[0][0]), y1 = pxY(iso[0][1]), x2 = pxX(iso[1][0]), y2 = pxY(iso[1][1]);
            const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2;
            let t = l2 ? Math.max(0, Math.min(1, ((mx - x1) * (x2 - x1) + (my - y1) * (y2 - y1)) / l2)) : 0;
            return Math.hypot(mx - (x1 + t * (x2 - x1)), my - (y1 + t * (y2 - y1)));
        }}

        cv.addEventListener("mousedown", e => {{
            const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
            if (Math.hypot(mx - pxX(sx), my - pxY(sy)) <= 16) drag = "star";
            else if (lineDist(mx, my) <= 10) drag = "line";
            if (drag) cv.style.cursor = "grabbing";
        }});

        window.addEventListener("mousemove", e => {{
            const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
            if (!drag) {{
                cv.style.cursor = (Math.hypot(mx - pxX(sx), my - pxY(sy)) <= 16 || lineDist(mx, my) <= 10) ? "grab" : "default";
                return;
            }}
            const [dx, dy] = clampCoord(dtX(mx), dtY(my));
            if (drag === "star") {{
                sx = dx; sy = dy; cz = D.c1 * sx + D.c2 * sy;
            }} else {{
                cz = D.c1 * dx + D.c2 * dy;
                const n2 = D.c1 * D.c1 + D.c2 * D.c2;
                if (n2 > 1e-9) {{
                    const dz = cz - (D.c1 * sx + D.c2 * sy);
                    [sx, sy] = clampCoord(sx + (dz * D.c1) / n2, sy + (dz * D.c2) / n2);
                    cz = D.c1 * sx + D.c2 * sy;
                }}
            }}
            draw();
        }});

        window.addEventListener("mouseup", () => {{ drag = null; cv.style.cursor = "default"; }});
        showV.addEventListener("change", draw);
        reset.addEventListener("click", () => {{ sx = D.optX; sy = D.optY; cz = D.optZ; draw(); }});

        lg.innerHTML = `<div style="font-weight:700;margin-bottom:6px;">圖例說明</div>
            <div class="item"><span class="sw" style="height:12px;background:rgba(20,184,166,0.28);border:1px solid #0d9488;"></span>可行解陰影區域</div>` +
            D.constraints.map(c => `<div class="item"><span class="sw" style="background:${{c.color}};"></span>${{c.label}}</div>`).join("") +
            `<div class="item"><span style="color:#dc2626;font-size:14px;">●</span>可行域頂點 (角點)</div>
             <div class="item"><span class="sw" style="border-top:2.5px dashed #7e22ce;height:0;"></span>等利潤線 (可拖曳)</div>
             <div class="item"><span style="color:#ef4444;font-size:15px;">★</span>解點座標 (可拖曳)</div>`;
        draw();
    </script>
    </body>
    </html>
    """
    components.html(html, height=580)


def call_gemini_lp_parser(api_key, chat_history):
    fence = chr(96) * 3
    system_prompt = (
        "你是一位精通作業研究（Operations Research）與線性規劃（Linear Programming / Integer Programming）的 AI 數學建模導師。\n"
        "請一律使用繁體中文（台灣慣用語）回覆，且務必完整輸出所有段落，不可中途省略或只寫一句開場白。\n"
        "你的任務是將使用者輸入的自然語言商業情境或應用問題，完整轉換為數學規劃模型並進行白話說明：\n"
        "1. 若題目資訊完整（包含決策變數、明確目標函數係數與所有限制式），請依序完整輸出以下四個部分：\n"
        "   - **一、決策變數定義**：清楚條列 $x_1, x_2, \\dots$ 代表的實體意義與單位。\n"
        "   - **二、目標函數（Objective Function）**：寫出 $\\max Z$ 或 $\\min Z$ 的數學式並說明其商業意義。\n"
        "   - **三、限制式條件（Constraints）**：逐條列出所有資源／需求限制式與非負（或整數、0-1）條件。\n"
        "   - **四、結構化模型 JSON**：在回覆的最末端附上一段 " + fence + "json ... " + fence + " 區塊，格式必須嚴格符合以下結構（系統會自動提取此 JSON 呼叫 SciPy HiGHS 精確求解器、繪圖並生成管理意涵解讀）：\n"
        + fence + "json\n"
        '{\n'
        '  "can_solve": true,\n'
        '  "opt_type": "Max",\n'
        '  "var_descriptions": ["x1 的單位與意義", "x2 的單位與意義"],\n'
        '  "c": [20.0, 15.0],\n'
        '  "constraints": [\n'
        '    {"name": "工時限制", "a": [2.0, 1.0], "op": "<=", "rhs": 20.0},\n'
        '    {"name": "原料限制", "a": [1.0, 3.0], "op": "<=", "rhs": 35.0}\n'
        '  ],\n'
        '  "non_neg": true,\n'
        '  "is_int": false,\n'
        '  "is_binary": false\n'
        '}\n'
        + fence + "\n"
        "2. 若題目只有資源限制、尚未提供完整的求解目標（例如未給各產品利潤或成本）或缺少關鍵數據，請完整說明「目前題目資料尚不完整，無法算出最終最佳解，但可先建立部分數學模型」，並完整列出 1. 決策變數定義、2. 已建立之資源限制式、3. 尚需補充的關鍵數據說明，最後在末端附上 \"can_solve\": false 的 JSON（其餘已知部分可填入，未知的 c 填 []）。"
    )

    contents = []
    for m in chat_history:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    return call_gemini_text(api_key, system_prompt, contents, temperature=0.2)


def extract_lp_json_and_clean_text(raw_text):
    fence = chr(96) * 3
    json_tag = fence + "json"
    clean_text = raw_text
    parsed_json = None
    if json_tag in raw_text:
        parts = raw_text.split(json_tag)
        clean_text = parts[0].strip()
        json_part = parts[1].split(fence)[0].strip()
        try:
            parsed_json = json.loads(json_part)
        except Exception:
            parsed_json = None
    return clean_text, parsed_json


# =================================================================================================
# 側邊欄導航
# =================================================================================================
app_mode = st.sidebar.radio(
    "系統功能導覽",
    [
        "AI 自然語言建模求解器",
        "線性規劃計算器",
        "0-1 整數規劃計算器",
        "運輸問題排程系統",
        "薪資最佳化決策支援系統",
        "可自訂限制式參數之薪資決策系統"
    ]
)

# -------------------------------------------------------------------------------------------------
# 頁面零：AI 自然語言建模求解器
# -------------------------------------------------------------------------------------------------
if app_mode == "AI 自然語言建模求解器":
    st.title("AI 自然語言數學建模與求解器")
    st.caption("輸入應用題或商業情境描述，自動轉換為線性／整數規劃數學模型，串接求解器計算最佳解並由 AI 深度解讀管理意涵")

    if "ai_lp_messages" not in st.session_state:
        st.session_state["ai_lp_messages"] = []

    has_started = len(st.session_state["ai_lp_messages"]) > 0

    st.markdown("**快速帶入經典作業研究題目測試：**")
    if has_started:
        ex1, ex2, ex3, ex_clr = st.columns([3, 3, 3, 1.4])
    else:
        ex1, ex2, ex3 = st.columns(3)

    sample_prompt = None
    if ex1.button("範例 1：家具工廠利潤最大化", use_container_width=True):
        sample_prompt = "某家具工廠生產木桌 (x1) 與木椅 (x2)。每張木桌可獲利 20 元，每張木椅可獲利 15 元。木工部門每週有 20 小時工時，製作一張木桌需 2 小時、木椅需 1 小時；塗裝部門每週有 35 小時工時，製作一張木桌需 1 小時、木椅需 3 小時。請問每週應各生產多少張木桌與木椅才能使總利潤最大？"
    if ex2.button("範例 2：Quik-Screen 運動衫（部分資料題）", use_container_width=True):
        sample_prompt = "Quik-Screen是一家成衣製造公司，已簽訂契約要生產一組標準的運動衫，包括兩件長袖衫（一件胸前有絹印，另一件前後都有絹印）以及兩件T恤（絹印形式與長袖衫一樣）。該公司必須在賽後的72小時內完成所有產品，採24小時全天候生產。卡車有足夠空間可裝載1,200個標準尺寸箱子，每個標準箱子可裝12件T恤，裝一打長袖衫的箱子則是標準箱子的三倍大。該公司備有$25,000的預算用來生產，同時庫房各有500打未絹印的長袖衫和T恤可供使用。請幫我建立數學模型。"
    if ex3.button("範例 3：專案投資 0-1 決策選取", use_container_width=True):
        sample_prompt = "公司正在評估 4 個研發專案是否執行（只能選 1 或不選 0），預期淨現值分別為 16 萬、22 萬、12 萬與 8 萬元，所需研發經費分別為 5 萬、7 萬、4 萬與 3 萬元。目前總預算上限為 14 萬元，且受限於人力，最多只能同時執行 3 個專案。請問應選擇哪些專案？"

    if has_started and ex_clr.button("🗑️ 清除對話", use_container_width=True):
        st.session_state["ai_lp_messages"] = []
        st.rerun()

    st.markdown("---")

    form_submitted_text = None
    chat_box_text = None

    if not has_started:
        with st.form("ai_lp_initial_form", clear_on_submit=True):
            user_question = st.text_area(
                "請輸入您的線性規劃 / 整數規劃應用問題：",
                height=135,
                placeholder=""
            )
            submitted = st.form_submit_button("🚀 轉換為數學模型、求解並由 AI 解釋 (Send)", type="primary", use_container_width=True)
            if submitted and user_question.strip():
                form_submitted_text = user_question.strip()
    else:
        for idx, msg in enumerate(st.session_state["ai_lp_messages"]):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                parsed = msg.get("parsed_model")
                if msg["role"] == "assistant" and parsed and parsed.get("can_solve") and parsed.get("c") and parsed.get("constraints"):
                    st.markdown("---")
                    st.subheader("⚙️ 系統自動求解與驗證結果 (SciPy HiGHS)")
                    opt_t = parsed.get("opt_type", "Max")
                    c_vec = [float(v) for v in parsed["c"]]
                    cons_list = [
                        {"a": [float(x) for x in c["a"]], "op": c["op"], "rhs": float(c["rhs"])}
                        for c in parsed["constraints"]
                    ]
                    n_neg = bool(parsed.get("non_neg", True))
                    i_int = bool(parsed.get("is_int", False))
                    i_bin = bool(parsed.get("is_binary", False))
                    var_descs = parsed.get("var_descriptions", [f"決策變數 x{j+1}" for j in range(len(c_vec))])

                    ok, s_data, err = solve_lp_highs(opt_t, c_vec, cons_list, n_neg, i_int, i_bin)
                    if not ok:
                        st.error(f"求解器驗證結果：此模型無可行解或為無界解 ({err})。")
                    else:
                        sol_v, best_z_v, sp_v = s_data["x"], s_data["z"], s_data["shadow_prices"]
                        m1, m2 = st.columns(2)
                        m1.metric("精確最佳目標值 (Z*)", fmt_num(best_z_v))
                        sol_lines = "<br>".join([
                            f"x{j+1} = {fmt_num(sol_v[j])} <span style='font-size:13px;color:#64748b;'>({var_descs[j] if j < len(var_descs) else ''})</span>"
                            for j in range(len(c_vec))
                        ])
                        m2.markdown(f"""
                        <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:10px 18px;border-radius:8px;">
                            <div style="font-size:13px;color:#64748b;margin-bottom:4px;">最佳決策變數解</div>
                            <div style="font-size:18px;font-weight:600;color:#0f172a;line-height:1.6;">{sol_lines}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        c_rows = []
                        for k, con in enumerate(parsed["constraints"]):
                            lhs_v = sum(float(con["a"][j]) * sol_v[j] for j in range(len(c_vec)))
                            is_b = abs(lhs_v - float(con["rhs"])) < 1e-4
                            slk = 0.0 if is_b else abs(float(con["rhs"]) - lhs_v)
                            row_d = {
                                "限制式": con.get("name", f"限制式 ({k+1})"),
                                "左側值 (LHS)": fmt_num(lhs_v),
                                "符號": con["op"],
                                "右側常數 (RHS)": fmt_num(float(con["rhs"])),
                                "鬆弛/剩餘量": fmt_num(slk),
                                "狀態": "Binding (緊束瓶頸)" if is_b else "Nonbinding (有餘量)"
                            }
                            if not (i_int or i_bin):
                                row_d["影子價格"] = fmt_num(sp_v[k])
                            c_rows.append(row_d)
                        st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)

                        if msg.get("ai_explanation"):
                            st.subheader("💡 AI 最佳決策方案與敏感度管理意涵解讀")
                            st.info(msg["ai_explanation"])

                        if len(c_vec) == 2 and abs(c_vec[0]) + abs(c_vec[1]) > 1e-10:
                            st.subheader("📊 二維座標平面視覺化與互動等利潤線")
                            render_interactive_2d_plot(c_vec, cons_list, sol_v, best_z_v, n_neg, i_int, i_bin)
                            render_2d_vertices_table(c_vec, cons_list, sol_v, opt_t, n_neg, i_int, i_bin, sec_prefix="📐")

        chat_box_text = st.chat_input("")

    final_prompt = (
        sample_prompt
        if sample_prompt
        else (form_submitted_text if form_submitted_text else (chat_box_text.strip() if chat_box_text and chat_box_text.strip() else None))
    )

    if final_prompt:
        gemini_key = get_gemini_api_key()
        if not gemini_key:
            st.error("尚未偵測到後端 GEMINI_API_KEY 設定。請在專案目錄下的 `.streamlit/secrets.toml` 檔案中填入 `GEMINI_API_KEY = \"你的金鑰\"`。")
        else:
            st.session_state["ai_lp_messages"].append({"role": "user", "content": final_prompt})
            with st.spinner("AI 正在建立數學模型、呼叫求解器計算並撰寫管理意涵解讀..."):
                try:
                    raw_reply = call_gemini_lp_parser(
                        gemini_key,
                        st.session_state["ai_lp_messages"]
                    )
                    clean_reply, parsed_json = extract_lp_json_and_clean_text(raw_reply)
                    ai_exp = None

                    if parsed_json and parsed_json.get("can_solve") and parsed_json.get("c") and parsed_json.get("constraints"):
                        opt_t = parsed_json.get("opt_type", "Max")
                        c_vec = [float(v) for v in parsed_json["c"]]
                        cons_list = [
                            {"a": [float(x) for x in c["a"]], "op": c["op"], "rhs": float(c["rhs"])}
                            for c in parsed_json["constraints"]
                        ]
                        n_neg = bool(parsed_json.get("non_neg", True))
                        i_int = bool(parsed_json.get("is_int", False))
                        i_bin = bool(parsed_json.get("is_binary", False))
                        var_descs = parsed_json.get("var_descriptions", [f"決策變數 x{j+1}" for j in range(len(c_vec))])

                        ok, s_data, _ = solve_lp_highs(opt_t, c_vec, cons_list, n_neg, i_int, i_bin)
                        if ok:
                            sol_v, best_z_v, sp_v = s_data["x"], s_data["z"], s_data["shadow_prices"]
                            summary_lines = [
                                f"【原始商業問題】：{final_prompt}",
                                f"【最佳化類型】：{opt_t}（{'0-1二元整數規劃' if i_bin else ('整數規劃' if i_int else '連續型線性規劃')}）",
                                f"【最佳目標函數值 Z*】：{fmt_num(best_z_v)}",
                                "【最佳決策變數解】："
                            ]
                            for j in range(len(c_vec)):
                                desc = var_descs[j] if j < len(var_descs) else f"變數 x{j+1}"
                                summary_lines.append(f"  - x{j+1} ({desc}) = {fmt_num(sol_v[j])}")
                            summary_lines.append("【限制式與敏感度分析數據】：")
                            for k, con in enumerate(parsed_json["constraints"]):
                                lhs_v = sum(float(con["a"][j]) * sol_v[j] for j in range(len(c_vec)))
                                is_b = abs(lhs_v - float(con["rhs"])) < 1e-4
                                slk = 0.0 if is_b else abs(float(con["rhs"]) - lhs_v)
                                sp_str = f"，影子價格 = {fmt_num(sp_v[k])}" if not (i_int or i_bin) else ""
                                summary_lines.append(
                                    f"  - {con.get('name', f'限制式({k+1})')}: LHS={fmt_num(lhs_v)} {con['op']} RHS={fmt_num(float(con['rhs']))}，"
                                    f"鬆弛/剩餘量={fmt_num(slk)}，狀態={'Binding(緊束瓶頸)' if is_b else 'Nonbinding(有餘量)'}{sp_str}"
                                )
                            ai_exp = generate_ai_management_insight(gemini_key, "\n".join(summary_lines))

                    st.session_state["ai_lp_messages"].append({
                        "role": "assistant",
                        "content": clean_reply,
                        "parsed_model": parsed_json,
                        "ai_explanation": ai_exp
                    })
                    st.rerun()
                except urllib.error.HTTPError as e:
                    if st.session_state["ai_lp_messages"] and st.session_state["ai_lp_messages"][-1]["role"] == "user":
                        st.session_state["ai_lp_messages"].pop()
                    if e.code == 503:
                        st.warning("⚠️ 目前 Google AI 伺服器遇到暫時性流量尖峰（HTTP 503 忙碌中），請稍等 3～5 秒後再送出一次即可。")
                    else:
                        err_detail = e.read().decode("utf-8", errors="ignore")
                        st.error(f"AI 服務連線失敗 (HTTP {e.code})：請確認後端設定的 API Key 是否完整有效。\n詳細訊息：{err_detail}")
                except Exception as e:
                    if st.session_state["ai_lp_messages"] and st.session_state["ai_lp_messages"][-1]["role"] == "user":
                        st.session_state["ai_lp_messages"].pop()
                    st.error(f"連線發生錯誤：{e}")

# -------------------------------------------------------------------------------------------------
# 頁面一：通用線性規劃計算器
# -------------------------------------------------------------------------------------------------
elif app_mode == "線性規劃計算器":
    st.title("線性規劃計算器 (LP Calculator)")
    st.caption("支援多決策變數建模、整數規劃限制、即時數學模型顯示、AI 管理意涵與敏感度分析、2D 互動拖曳等利潤線與解點")

    st.subheader("1. 模型基本設定與目標函數")
    c1, c2 = st.columns(2)
    opt_type = c1.selectbox("最佳化類型", ["Max", "Min"])
    num_vars = c2.number_input("決策變數數量", min_value=1, max_value=6, value=2, step=1)
    is_int = st.checkbox("啟用整數限制 (xᵢ ∈ ℤ)", value=False)

    st.markdown("**請輸入目標函數係數 (Z)：**")
    obj_cols = st.columns(num_vars)
    default_c = [20.0, 15.0, 10.0, 10.0, 10.0, 10.0]
    obj_c = [obj_cols[i].number_input(f"x{i+1} 係數", value=default_c[i], step=1.0, key=f"obj_{i}") for i in range(num_vars)]

    st.markdown("---")
    st.subheader("2. 限制式條件設定")
    non_neg = st.checkbox("啟用非負限制 (xᵢ ≥ 0)", value=True)
    num_constraints = st.number_input("限制式數量", min_value=1, max_value=15, value=2, step=1)

    constraints_input = []
    default_A, default_rhs = [[2.0, 1.0], [1.0, 3.0]], [20.0, 35.0]
    for k in range(num_constraints):
        st.markdown(f"**限制式 ({k+1})：**")
        c_cols = st.columns(num_vars + 2)
        row_a = [
            c_cols[j].number_input(
                f"x{j+1}",
                value=float(default_A[k][j] if (k < 2 and j < 2) else 1.0),
                step=1.0,
                key=f"c_{k}_{j}"
            )
            for j in range(num_vars)
        ]
        op = c_cols[num_vars].selectbox("符號", ["<=", ">=", "="], key=f"op_{k}")
        rhs_val = c_cols[num_vars + 1].number_input("右端常數 (RHS)", value=float(default_rhs[k] if k < 2 else 10.0), step=1.0, key=f"rhs_{k}")
        constraints_input.append({'a': row_a, 'op': op, 'rhs': rhs_val})

    st.markdown("---")
    st.subheader("3. 數學模型即時預覽")
    obj_latex = " + ".join([f"{fmt_num(obj_c[i])}x_{{{i+1}}}" for i in range(num_vars)])
    vars_str = ", ".join([f"x_{{{j+1}}}" for j in range(num_vars)])
    con_latex_list = []
    for con in constraints_input:
        lhs_terms = " + ".join([f"{fmt_num(con['a'][j])}x_{{{j+1}}}" for j in range(num_vars)])
        sym = "\\le" if con['op'] == '<=' else ("\\ge" if con['op'] == '>=' else "=")
        con_latex_list.append(f"{lhs_terms} &{sym} {fmt_num(con['rhs'])}")
    if non_neg:
        con_latex_list.append(f"{vars_str} &\\ge 0")
    if is_int:
        con_latex_list.append(f"{vars_str} &\\in \\mathbb{{Z}}")
    st.latex(f"\\begin{{aligned}} \\text{{{opt_type}}}\\quad Z &= {obj_latex} \\\\\n\\text{{subject to}}\\quad " + " \\\\\n".join(con_latex_list) + "\n\\end{aligned}")

    if st.button("求解最佳解 (Solve)", type="primary"):
        st.session_state["lp_solved"] = True

    if st.session_state.get("lp_solved", False):
        success, sol_data, err_msg = solve_lp_highs(opt_type, obj_c, constraints_input, non_neg, is_int)
        if not success:
            st.error(f"求解失敗：此模型無可行解 (Infeasible) 或解為無界解 (Unbounded)。詳細訊息: {err_msg}")
        else:
            sol, best_z, s_prices = sol_data['x'], sol_data['z'], sol_data['shadow_prices']
            st.success(f"求解成功！已找到最佳{'整數' if is_int else ''}決策方案。")

            res_c1, res_c2 = st.columns(2)
            res_c1.metric("最佳目標函數值 (Z*)", fmt_num(best_z))
            sol_multiline = "<br>".join([f"x{j+1} = {fmt_num(sol[j])}" for j in range(num_vars)])
            res_c2.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:10px 18px;border-radius:8px;">
                <div style="font-size:13px;color:#64748b;margin-bottom:4px;">最佳決策變數方案</div>
                <div style="font-size:20px;font-weight:600;color:#0f172a;line-height:1.5;">{sol_multiline}</div>
            </div>
            """, unsafe_allow_html=True)

            table_data = []
            summary_lines = [
                f"【最佳化類型】：{opt_type}（{'整數規劃 MILP' if is_int else '線性規劃 LP'}）",
                f"【目標函數】：Z = {' + '.join([f'{fmt_num(obj_c[j])}x{j+1}' for j in range(num_vars)])}",
                f"【最佳目標函數值 Z*】：{fmt_num(best_z)}",
                "【最佳決策變數解】：" + "，".join([f"x{j+1} = {fmt_num(sol[j])}" for j in range(num_vars)]),
                "【限制式與敏感度數據】："
            ]

            for k, con in enumerate(constraints_input):
                lhs_val = sum(con['a'][j] * sol[j] for j in range(num_vars))
                is_binding = abs(lhs_val - con['rhs']) < 1e-4
                slack = 0.0 if is_binding else abs(con['rhs'] - lhs_val)
                sp = s_prices[k]
                status_str = "Binding (緊束瓶頸)" if is_binding else "Nonbinding (有餘量)"
                sp_text = f"，影子價格 = {fmt_num(sp)}" if (not is_int and sp is not None) else ""
                lhs_expr = " + ".join([f"{fmt_num(con['a'][j])}x{j+1}" for j in range(num_vars)])
                summary_lines.append(
                    f"  - 限制式 ({k+1}) [{lhs_expr} {con['op']} {fmt_num(con['rhs'])}]: "
                    f"LHS={fmt_num(lhs_val)}，鬆弛/剩餘量={fmt_num(slack)}，狀態={status_str}{sp_text}"
                )

                row_dict = {
                    "限制式": f"限制式 ({k+1})", "左側值 (LHS)": fmt_num(lhs_val), "運算符號": con['op'],
                    "右側常數 (RHS)": fmt_num(con['rhs']), "鬆弛/剩餘量 (Slack/Surplus)": fmt_num(slack),
                    "狀態判定": status_str
                }
                if not is_int:
                    row_dict["影子價格 (Shadow Price)"] = fmt_num(sp)
                table_data.append(row_dict)

            st.subheader("4. AI 最佳決策方案與管理意涵解讀")
            gemini_key = get_gemini_api_key()
            cache_key = json.dumps({"m": "lp", "opt": opt_type, "c": obj_c, "cons": constraints_input, "nn": non_neg, "int": is_int}, sort_keys=True)
            if gemini_key:
                if st.session_state.get("lp_ai_cache_key") != cache_key:
                    with st.spinner("AI 正在針對求解結果與敏感度數據撰寫管理意涵解讀..."):
                        try:
                            st.session_state["lp_ai_cache_text"] = generate_ai_management_insight(gemini_key, "\n".join(summary_lines))
                            st.session_state["lp_ai_cache_key"] = cache_key
                        except Exception as e:
                            st.session_state["lp_ai_cache_text"] = f"⚠️ AI 解讀暫時無法連線（{e}），請參考下方數據表。"
                st.info(st.session_state["lp_ai_cache_text"])
            else:
                meaning_parts = [f"生產/配置決策變數 x{j+1} 為 **{fmt_num(sol[j])}** 單位" for j in range(num_vars)]
                st.info(f"根據{'整數' if is_int else '線性'}規劃求解結果，建議之最佳執行方案為：{'； '.join(meaning_parts)}。在此資源分配條件下，預計可達成最佳目標總值 (Z*) 為 **{fmt_num(best_z)}**。")

            st.subheader("5. 限制式與敏感度分析數據表")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            if is_int:
                st.info("ℹ️ 已啟用整數規劃限制 (MILP)：整數規劃模型不具連續型對偶影子價格 (Shadow Price)。")
            else:
                st.warning("⚠️ **作業研究理論提醒**：上述影子價格僅在當前最佳基底之右端項允許變動範圍 (Allowable Range) 內有效。")

            if num_vars == 2:
                st.markdown("---")
                st.subheader("6. 二維座標平面視覺化與互動等利潤線 (2D Graphic Analysis)")
                st.caption("陰影區域顯示了同時滿足所有不等式的地方；按住圖中的 **紅色星星（解點）** 或 **紫色虛線（等利潤線）** 拖曳，上方方程式的 Z 值與座標會即時跟著改變。")
                if abs(obj_c[0]) + abs(obj_c[1]) < 1e-10:
                    st.info("目前目標函數係數皆為 0，沒有可拖曳的等利潤線。")
                else:
                    render_interactive_2d_plot(obj_c, constraints_input, sol, best_z, non_neg, is_int)

                st.markdown("---")
                render_2d_vertices_table(obj_c, constraints_input, sol, opt_type, non_neg, is_int)

# -------------------------------------------------------------------------------------------------
# 頁面二：0-1 整數規劃計算器 (變數只能是 0 或 1)
# -------------------------------------------------------------------------------------------------
elif app_mode == "0-1 整數規劃計算器":
    st.title("0-1 整數規劃計算器 (Binary Integer Programming)")
    st.caption("所有決策變數皆限制為 0 或 1 (xᵢ ∈ {0, 1})，適用於專案選取、背包問題、指派與決策是非題")

    st.subheader("1. 模型基本設定與目標函數")
    c1, c2 = st.columns(2)
    bip_opt_type = c1.selectbox("最佳化類型", ["Max", "Min"], key="bip_opt")
    bip_num_vars = c2.number_input("決策變數數量 (xᵢ ∈ {0, 1})", min_value=1, max_value=10, value=4, step=1, key="bip_nvars")

    st.markdown("**請輸入目標函數係數 (Z)：**")
    bip_obj_cols = st.columns(min(bip_num_vars, 5))
    bip_default_c = [16.0, 22.0, 12.0, 8.0, 10.0, 14.0, 9.0, 11.0, 7.0, 15.0]
    bip_obj_c = [
        bip_obj_cols[i % 5].number_input(f"x{i+1} 係數", value=bip_default_c[i], step=1.0, key=f"bip_obj_{i}")
        for i in range(bip_num_vars)
    ]

    st.markdown("---")
    st.subheader("2. 限制式條件設定")
    bip_num_cons = st.number_input("限制式數量", min_value=1, max_value=15, value=2, step=1, key="bip_ncons")

    bip_cons_input = []
    bip_default_A = [[5.0, 7.0, 4.0, 3.0], [1.0, 1.0, 1.0, 1.0]]
    bip_default_rhs = [14.0, 3.0]
    for k in range(bip_num_cons):
        st.markdown(f"**限制式 ({k+1})：**")
        c_cols = st.columns(bip_num_vars + 2)
        row_a = [
            c_cols[j].number_input(
                f"x{j+1}",
                value=float(bip_default_A[k][j] if (k < 2 and j < 4) else 1.0),
                step=1.0,
                key=f"bip_c_{k}_{j}"
            )
            for j in range(bip_num_vars)
        ]
        op = c_cols[bip_num_vars].selectbox("符號", ["<=", ">=", "="], key=f"bip_op_{k}")
        rhs_val = c_cols[bip_num_vars + 1].number_input("右端常數 (RHS)", value=float(bip_default_rhs[k] if k < 2 else 2.0), step=1.0, key=f"bip_rhs_{k}")
        bip_cons_input.append({'a': row_a, 'op': op, 'rhs': rhs_val})

    st.markdown("---")
    st.subheader("3. 0-1 整數規劃數學模型即時預覽")
    obj_latex = " + ".join([f"{fmt_num(bip_obj_c[i])}x_{{{i+1}}}" for i in range(bip_num_vars)])
    vars_str = ", ".join([f"x_{{{j+1}}}" for j in range(bip_num_vars)])
    con_latex_list = []
    for con in bip_cons_input:
        lhs_terms = " + ".join([f"{fmt_num(con['a'][j])}x_{{{j+1}}}" for j in range(bip_num_vars)])
        sym = "\\le" if con['op'] == '<=' else ("\\ge" if con['op'] == '>=' else "=")
        con_latex_list.append(f"{lhs_terms} &{sym} {fmt_num(con['rhs'])}")
    con_latex_list.append(f"{vars_str} &\\in \\{{0, 1\\}}")
    st.latex(f"\\begin{{aligned}} \\text{{{bip_opt_type}}}\\quad Z &= {obj_latex} \\\\\n\\text{{subject to}}\\quad " + " \\\\\n".join(con_latex_list) + "\n\\end{aligned}")

    if st.button("求解 0-1 最佳決策方案 (Solve Binary IP)", type="primary", key="bip_solve_btn"):
        st.session_state["bip_solved"] = True

    if st.session_state.get("bip_solved", False):
        success, sol_data, err_msg = solve_lp_highs(bip_opt_type, bip_obj_c, bip_cons_input, non_neg=True, is_binary=True)
        if not success:
            st.error(f"求解失敗：此 0-1 模型無可行解 (Infeasible)。詳細訊息: {err_msg}")
        else:
            sol, best_z = sol_data['x'], sol_data['z']
            st.success("求解成功！已求得全域最佳 0-1 二元決策組合。")

            res_c1, res_c2 = st.columns(2)
            res_c1.metric("最佳目標函數值 (Z*)", fmt_num(best_z))
            sol_multiline = "<br>".join([
                f"x{j+1} = {int(round(sol[j]))}  ({'✅ 選取 / 執行 (1)' if round(sol[j]) == 1 else '⬜ 不選取 (0)'})"
                for j in range(bip_num_vars)
            ])
            res_c2.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;padding:10px 18px;border-radius:8px;">
                <div style="font-size:13px;color:#64748b;margin-bottom:4px;">最佳 0-1 決策變數方案</div>
                <div style="font-size:18px;font-weight:600;color:#0f172a;line-height:1.5;">{sol_multiline}</div>
            </div>
            """, unsafe_allow_html=True)

            table_data = []
            summary_lines = [
                f"【最佳化類型】：{bip_opt_type}（0-1 二元整數規劃 Binary IP）",
                f"【目標函數】：Z = {' + '.join([f'{fmt_num(bip_obj_c[j])}x{j+1}' for j in range(bip_num_vars)])}",
                f"【最佳目標函數值 Z*】：{fmt_num(best_z)}",
                "【最佳 0-1 決策變數解】：" + "，".join([f"x{j+1}={int(round(sol[j]))}" for j in range(bip_num_vars)]),
                "【限制式資源使用情況】："
            ]
            for k, con in enumerate(bip_cons_input):
                lhs_val = sum(con['a'][j] * sol[j] for j in range(bip_num_vars))
                is_binding = abs(lhs_val - con['rhs']) < 1e-4
                slack = 0.0 if is_binding else abs(con['rhs'] - lhs_val)
                status_str = "Binding (緊束瓶頸)" if is_binding else "Nonbinding (有餘量)"
                lhs_expr = " + ".join([f"{fmt_num(con['a'][j])}x{j+1}" for j in range(bip_num_vars)])
                summary_lines.append(
                    f"  - 限制式 ({k+1}) [{lhs_expr} {con['op']} {fmt_num(con['rhs'])}]: "
                    f"LHS={fmt_num(lhs_val)}，鬆弛/剩餘量={fmt_num(slack)}，狀態={status_str}"
                )
                table_data.append({
                    "限制式": f"限制式 ({k+1})", "左側值 (LHS)": fmt_num(lhs_val), "運算符號": con['op'],
                    "右側常數 (RHS)": fmt_num(con['rhs']), "鬆弛/剩餘量 (Slack/Surplus)": fmt_num(slack),
                    "狀態判定": status_str
                })

            st.subheader("4. AI 最佳決策方案與管理意涵解讀")
            gemini_key = get_gemini_api_key()
            bip_cache_key = json.dumps({"m": "bip", "opt": bip_opt_type, "c": bip_obj_c, "cons": bip_cons_input}, sort_keys=True)
            if gemini_key:
                if st.session_state.get("bip_ai_cache_key") != bip_cache_key:
                    with st.spinner("AI 正在針對 0-1 決策組合與資源使用情況撰寫管理意涵解讀..."):
                        try:
                            st.session_state["bip_ai_cache_text"] = generate_ai_management_insight(gemini_key, "\n".join(summary_lines))
                            st.session_state["bip_ai_cache_key"] = bip_cache_key
                        except Exception as e:
                            st.session_state["bip_ai_cache_text"] = f"⚠️ AI 解讀暫時無法連線（{e}），請參考下方數據表。"
                st.info(st.session_state["bip_ai_cache_text"])
            else:
                chosen = [f"x{j+1}" for j in range(bip_num_vars) if round(sol[j]) == 1]
                chosen_str = "、".join(chosen) if chosen else "無任何項目（全為 0）"
                st.info(f"根據 0-1 二元整數規劃求解結果，建議選取（值為 1）之決策項目為：**{chosen_str}**，其餘項目為 0；預計可達成最佳目標總值 (Z*) 為 **{fmt_num(best_z)}**。")

            st.subheader("5. 限制式資源使用與鬆弛量分析")
            st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

            if bip_num_vars == 2:
                st.markdown("---")
                st.subheader("6. 二維 0-1 座標平面視覺化與互動等利潤線 (2D Binary Graphic Analysis)")
                st.caption("圖中黑點為 (0,0)、(1,0)、(0,1)、(1,1) 四個二元候選點，可拖曳星星或紫色虛線觀察目標值變化。")
                if abs(bip_obj_c[0]) + abs(bip_obj_c[1]) < 1e-10:
                    st.info("目前目標函數係數皆為 0，沒有可拖曳的等利潤線。")
                else:
                    render_interactive_2d_plot(bip_obj_c, bip_cons_input, sol, best_z, non_neg=True, is_binary=True)

                st.markdown("---")
                render_2d_vertices_table(bip_obj_c, bip_cons_input, sol, bip_opt_type, non_neg=True, is_binary=True)

# -------------------------------------------------------------------------------------------------
# 頁面三：運輸問題排程系統
# -------------------------------------------------------------------------------------------------
elif app_mode == "運輸問題排程系統":
    st.title("運輸問題排程決策系統 (Transportation Problem)")
    st.caption("經典作業研究應用：多產地與多銷地間的供需平衡與物流最低成本配置模型")

    c1, c2 = st.columns(2)
    num_sources = c1.number_input("供應地數量 (Sources / 供給地)", min_value=1, max_value=6, value=3, step=1)
    num_dests = c2.number_input("需求地數量 (Destinations / 需求地)", min_value=1, max_value=6, value=4, step=1)

    st.markdown("**各供給地供應量 (Supply)：**")
    supp_cols = st.columns(num_sources)
    supplies = [supp_cols[i].number_input(f"供應地 S{i+1}", min_value=0.0, value=200.0, step=10.0, key=f"supp_{i}") for i in range(num_sources)]

    st.markdown("**各需求地目標需求量 (Demand)：**")
    dem_cols = st.columns(num_dests)
    demands = [dem_cols[j].number_input(f"需求地 D{j+1}", min_value=0.0, value=150.0, step=10.0, key=f"dem_{j}") for j in range(num_dests)]

    st.markdown("**單位運費矩陣 (Unit Cost Matrix)：**")
    df_cost_input = pd.DataFrame(
        [[float((i + 1) * 3 + (j + 1) * 2) for j in range(num_dests)] for i in range(num_sources)],
        index=[f"供應地 S{i+1}" for i in range(num_sources)],
        columns=[f"需求地 D{j+1}" for j in range(num_dests)]
    )
    edited_costs = st.data_editor(df_cost_input, use_container_width=True)

    expand_all_latex = st.checkbox("完整展開目標函數所有變數項", value=(num_sources * num_dests <= 8))
    with st.expander("點擊查看系統自動生成之完整 LP 數學模型", expanded=True):
        cost_mat = edited_costs.values
        total_terms = num_sources * num_dests
        obj_terms = [f"{fmt_num(cost_mat[i][j])}x_{{{i * num_dests + j + 1}}}" for i in range(num_sources) for j in range(num_dests)]
        if expand_all_latex or total_terms <= 8:
            obj_str = " \\\\\n&\\quad + ".join([" + ".join(obj_terms[idx:idx + 4]) for idx in range(0, len(obj_terms), 4)])
        else:
            obj_str = " + ".join(obj_terms[:6]) + " + \\cdots + " + obj_terms[-1]

        con_lines = [f"{' + '.join([f'x_{{{i * num_dests + j + 1}}}' for j in range(num_dests)])} &\\le {fmt_num(supplies[i])}" for i in range(num_sources)]
        con_lines += [f"{' + '.join([f'x_{{{i * num_dests + j + 1}}}' for i in range(num_sources)])} &\\ge {fmt_num(demands[j])}" for j in range(num_dests)]
        con_lines.append(f"{', '.join([f'x_{{{m+1}}}' for m in range(total_terms)])} &\\ge 0" if total_terms <= 6 else f"x_k &\\ge 0, \\quad \\forall k=1,\\dots,{total_terms}")
        st.latex(f"\\begin{{aligned}} \\min\\quad Z &= {obj_str} \\\\\n\\text{{subject to}}\\quad " + " \\\\\n".join(con_lines) + "\n\\end{aligned}")

    if st.button("求解最佳調度方案 (Solve LP)", type="primary"):
        costs_matrix = edited_costs.values.tolist()
        num_vars_tp = num_sources * num_dests
        c_tp = [costs_matrix[i][j] for i in range(num_sources) for j in range(num_dests)]
        A_ub_tp, b_ub_tp = [], []
        for i in range(num_sources):
            row = [0.0] * num_vars_tp
            for j in range(num_dests): row[i * num_dests + j] = 1.0
            A_ub_tp.append(row); b_ub_tp.append(supplies[i])
        for j in range(num_dests):
            row = [0.0] * num_vars_tp
            for i in range(num_sources): row[i * num_dests + j] = -1.0
            A_ub_tp.append(row); b_ub_tp.append(-demands[j])

        res_tp = linprog(c_tp, A_ub=A_ub_tp, b_ub=b_ub_tp, bounds=[(0, None)] * num_vars_tp, method='highs')
        if not res_tp.success:
            st.error(f"運輸模型求解失敗：無法滿足供需條件（{res_tp.message}）。")
        else:
            sol_x = res_tp.x.reshape((num_sources, num_dests))
            st.success("計算完成！已求得全域最佳成本調度方案。")
            m1, m2, m3 = st.columns(3)
            m1.metric("最低總運輸成本 (Z*)", f"${int(round(res_tp.fun)):,} 元")
            m2.metric("總供應總量", f"{fmt_num(sum(supplies))} 單位")
            m3.metric("總需求總量", f"{fmt_num(sum(demands))} 單位")

            records = [{
                "供應來源地": f"供應地 S{i+1}", "配送目的地": f"需求地 D{j+1}",
                "每單位運費": f"${fmt_num(costs_matrix[i][j])}",
                "最佳調度數量": f"{fmt_num(sol_x[i][j])} 單位",
                "運費小計": f"${fmt_num(sol_x[i][j] * costs_matrix[i][j])} 元"
            } for i in range(num_sources) for j in range(num_dests) if sol_x[i][j] > 1e-3]
            st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

            fig_heat = go.Figure(data=go.Heatmap(
                z=sol_x, x=[f"需求地 D{j+1}" for j in range(num_dests)], y=[f"供應地 S{i+1}" for i in range(num_sources)],
                colorscale='Blues', text=[[fmt_num(sol_x[i][j]) for j in range(num_dests)] for i in range(num_sources)],
                texttemplate="%{text}", textfont={"size": 14}
            ))
            fig_heat.update_layout(height=340)
            st.plotly_chart(fig_heat, use_container_width=True)

# -------------------------------------------------------------------------------------------------
# 頁面四：薪資最佳化決策支援系統
# -------------------------------------------------------------------------------------------------
elif app_mode == "薪資最佳化決策支援系統":
    st.title("薪資最佳化決策支援系統 (DSS)")
    st.caption("進階管理科學問題範本：個人財務與多目標資源分配模型")

    salary = st.sidebar.slider("每月實領總薪資 (元)", 30000, 120000, 50000, 1000)
    fixed_cost = st.sidebar.slider("每月固定支出 (元)", 5000, 40000, 12000, 500)
    target_gap = st.sidebar.number_input("目標緊急預備金總額 (元)", value=300000, step=10000)
    current_gap = st.sidebar.number_input("目前已存預備金 (元)", value=280000, step=5000)
    persona = st.sidebar.selectbox("偏好情境 (目標函數權重)", ["均衡穩健型", "資產衝刺型", "生活品質型", "自訂權重模式"])

    B, gap = max(0, salary - fixed_cost), max(0, target_gap - current_gap)
    weights_map = {
        "均衡穩健型": [1.0, 1.3, 1.3, 2.5, 3.0, 1.6, 1.4],
        "資產衝刺型": [1.0, 1.0, 2.2, 3.0, 2.0, 1.5, 1.0],
        "生活品質型": [1.2, 2.0, 1.2, 1.5, 1.5, 1.2, 2.5]
    }
    var_names = ["基本生活 (x1)", "社交休閒 (x2)", "自我進修 (x3)", "長期投資 (x4)", "緊急預備金 (x5)", "固定儲蓄 (x6)", "個人嗜好 (x7)"]

    if persona == "自訂權重模式":
        default_custom = [1.0, 1.5, 1.5, 2.5, 2.0, 1.5, 1.0]
        active_weights = [st.sidebar.slider(f"{name} 權重", 0.1, 5.0, default_custom[i], 0.1, key=f"cw_{i}") for i, name in enumerate(var_names)]
    else:
        active_weights = weights_map[persona]

    st.markdown(f"**當前套用之目標函數：** $\\max Z = {' + '.join([f'{w:.1f}x_{{{i+1}}}' for i, w in enumerate(active_weights)])}$")

    def solve_salary_dss(budget, reserve_gap, w):
        A = [
            [1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 1, 0, 0],
            [0, 0, -1, -1, 0, 0, 1], [0, -1, 0, 0.2, 0, 0, -1], [0, 0, 1, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0]
        ]
        b = [budget, 0.65 * budget, reserve_gap, 0, 0, 0.15 * budget, -0.07 * budget, -9000, -10000, -0.05 * budget, -0.10 * budget]
        res = linprog([-val for val in w], A_ub=A, b_ub=b, bounds=[(0, None)] * 7, method='highs')
        return (True, res.x, -res.fun) if res.success else (False, [], 0)

    success, sol, max_z = solve_salary_dss(B, gap, active_weights)
    if not success:
        st.error("求解狀態：無可行解 (Infeasible)。最低生活與投資底限總和超出可支配預算。")
    else:
        asset_sum = sol[3] + sol[4] + sol[5]
        c1, c2, c3 = st.columns(3)
        c1.metric("可支配預算 (B)", f"{int(B):,} 元")
        c2.metric("最大目標函數值 (Max Z)", f"{int(round(max_z)):,}", f"情境：{persona}")
        c3.metric("資產累積總額 (x4+x5+x6)", f"{int(round(asset_sum)):,} 元", f"佔預算 {asset_sum/B*100:.1f}%")

        col_t, col_c = st.columns(2)
        col_t.dataframe(pd.DataFrame({
            "決策變數": [f"x{i+1}" for i in range(7)],
            "分配項目": ["基本生活開銷", "社交/休閒娛樂", "自我進修/學習", "長期投資配置", "緊急預備金提撥", "固定儲蓄提撥", "個人愛好支出"],
            "最佳解金額": [f"{int(round(v)):,} 元" for v in sol],
            "預算佔比": [f"{v/B*100:.1f}%" for v in sol]
        }), use_container_width=True, hide_index=True)

        fig_pie = go.Figure(data=[go.Pie(labels=var_names, values=sol, hole=0.55, textinfo='label+percent')])
        fig_pie.update_layout(title="最佳化支出結構佔比", height=300, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
        col_c.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("敏感度情境模擬 (What-if Analysis)")
        sc1, sc2 = st.columns(2)
        sim_B = max(0, sc1.slider("情境模擬：總薪資 (元)", 30000, 120000, salary + 6000, 1000) - sc2.slider("情境模擬：固定支出 (元)", 5000, 40000, fixed_cost, 500))
        sim_ok, sim_sol, sim_z = solve_salary_dss(sim_B, gap, active_weights)
        if sim_ok:
            st.dataframe(pd.DataFrame({
                "項目": var_names + ["可支配預算 (B)", "最大目標函數值 (Max Z)"],
                "基準方案": [f"{int(round(v)):,} 元" for v in sol] + [f"{int(B):,} 元", f"{int(round(max_z)):,}"],
                "模擬情境": [f"{int(round(v)):,} 元" for v in sim_sol] + [f"{int(sim_B):,} 元", f"{int(round(sim_z)):,}"],
                "差異 (Δ)": [f"{int(round(s - b)):+,} 元" for s, b in zip(sim_sol, sol)] + [f"{int(sim_B - B):+,} 元", f"{int(round(sim_z - max_z)):+,}"]
            }), use_container_width=True, hide_index=True)

# -------------------------------------------------------------------------------------------------
# 頁面五：可自訂限制式參數之薪資決策系統
# -------------------------------------------------------------------------------------------------
else:
    st.title("可自訂限制式參數之薪資決策系統 (Customizable Constraints DSS)")

    p1, p2, p3, p4 = st.columns(4)
    param_salary = p1.number_input("每月實領總薪資 (元)", 20000, 200000, 50000, 1000)
    param_fixed = p2.number_input("每月固定支出 (元)", 0, 80000, 12000, 500)
    param_tg = p3.number_input("目標緊急預備金總額 (元)", value=300000, step=10000)
    param_cg = p4.number_input("目前已存預備金 (元)", value=280000, step=5000)

    B_val, gap_val = max(0.0, float(param_salary - param_fixed)), max(0.0, float(param_tg - param_cg))
    st.info(f"💡 **可支配預算 (B)**：**{int(B_val):,} 元** ｜ **緊急預備金缺口 (gap)**：**{int(gap_val):,} 元**")

    c1, c2 = st.columns(2)
    with c1:
        param_x1_min = st.number_input("基本生活保障下限 (x₁ ≥ ? 元)", value=9000.0, step=500.0)
        param_x4_min = st.number_input("核心長期投資底限 (x₄ ≥ ? 元)", value=10000.0, step=1000.0)
        param_bal = st.number_input("投資連動生活平衡係數 (係數*x₄ - x₂ - x₇ ≤ 0)", value=0.20, step=0.05, min_value=0.0, max_value=1.0)
    with c2:
        r_asset = st.slider("資產累積上限比例 (% × B)", 30, 90, 65, 5) / 100.0
        r_x3_max = st.slider("自我進修上限比例 (% × B)", 5, 40, 15, 1) / 100.0
        r_x3_min = st.slider("自我進修下限比例 (% × B)", 0, 20, 7, 1) / 100.0
        r_x2_min = st.slider("社交休閒保障比例 (% × B)", 0, 20, 5, 1) / 100.0
        r_x6_min = st.slider("固定儲蓄提撥比例 (% × B)", 0, 30, 10, 1) / 100.0

    var_list_3 = ["基本生活 (x1)", "社交休閒 (x2)", "自我進修 (x3)", "長期投資 (x4)", "緊急預備金 (x5)", "固定儲蓄 (x6)", "個人嗜好 (x7)"]
    w_cols = st.columns(7)
    default_w3 = [1.0, 1.3, 1.3, 2.5, 3.0, 1.6, 1.4]
    weights_3 = [w_cols[i].number_input(f"x{i+1} 權重", value=default_w3[i], step=0.1, key=f"w3_{i}") for i in range(7)]

    constraints_custom = [
        {"name": "總預算上限", "a": [1, 1, 1, 1, 1, 1, 1], "op": "<=", "rhs": B_val},
        {"name": "資產累積上限", "a": [0, 0, 0, 1, 1, 1, 0], "op": "<=", "rhs": r_asset * B_val},
        {"name": "緊急預備金水位", "a": [0, 0, 0, 0, 1, 0, 0], "op": "<=", "rhs": gap_val},
        {"name": "嗜好連動約束", "a": [0, 0, -1, -1, 0, 0, 1], "op": "<=", "rhs": 0.0},
        {"name": "生活平衡約束", "a": [0, -1, 0, param_bal, 0, 0, -1], "op": "<=", "rhs": 0.0},
        {"name": "進修上限", "a": [0, 0, 1, 0, 0, 0, 0], "op": "<=", "rhs": r_x3_max * B_val},
        {"name": "進修下限", "a": [0, 0, 1, 0, 0, 0, 0], "op": ">=", "rhs": r_x3_min * B_val},
        {"name": "基本生活保障", "a": [1, 0, 0, 0, 0, 0, 0], "op": ">=", "rhs": param_x1_min},
        {"name": "核心投資底限", "a": [0, 0, 0, 1, 0, 0, 0], "op": ">=", "rhs": param_x4_min},
        {"name": "社交休閒底限", "a": [0, 1, 0, 0, 0, 0, 0], "op": ">=", "rhs": r_x2_min * B_val},
        {"name": "固定儲蓄底限", "a": [0, 0, 0, 0, 0, 1, 0], "op": ">=", "rhs": r_x6_min * B_val}
    ]

    with st.expander("點擊展開查看動態生成之 LP 模型", expanded=True):
        obj_str3 = " + ".join([f"{fmt_num(weights_3[i])}x_{{{i+1}}}" for i in range(7)])
        latex_lines3 = [f"\\begin{{aligned}} \\max\\quad Z &= {obj_str3} \\\\\n\\text{{subject to}}\\quad "]
        for con in constraints_custom:
            terms = [
                f"x_{{{j+1}}}" if abs(c - 1.0) < 1e-5 else (f"-x_{{{j+1}}}" if abs(c + 1.0) < 1e-5 else f"{fmt_num(c)}x_{{{j+1}}}")
                for j, c in enumerate(con['a']) if c != 0
            ]
            sym = "\\le" if con['op'] == '<=' else ("\\ge" if con['op'] == '>=' else "=")
            latex_lines3.append(f"{' + '.join(terms).replace('+ -', '- ')} &{sym} {fmt_num(con['rhs'])}")
        latex_lines3.append("x_i &\\ge 0, \\quad \\forall i=1,\\dots,7")
        st.latex(" \\\\\n".join(latex_lines3) + "\n\\end{aligned}")

    if st.button("依自訂限制式參數求解 (Solve Custom LP)", type="primary"):
        ok_3, data_3, err_3 = solve_lp_highs("Max", weights_3, constraints_custom, non_neg=True)
        if not ok_3:
            st.error(f"求解失敗：無可行解 ({err_3})。請調降最低保障金額或提高薪資。")
        else:
            sol_3, best_z_3, sp_3 = data_3['x'], data_3['z'], data_3['shadow_prices']
            st.success("計算成功！已求得最佳財務資源配置方案。")
            r1, r2, r3 = st.columns(3)
            r1.metric("可支配預算 (B)", f"{int(B_val):,} 元")
            r2.metric("最佳目標函數值 (Max Z*)", fmt_num(best_z_3))
            asset_3 = sol_3[3] + sol_3[4] + sol_3[5]
            r3.metric("資產累積總額 (x₄+x₅+x₆)", f"{int(round(asset_3)):,} 元", f"佔預算 {asset_3/B_val*100:.1f}%")

            ct3, cp3 = st.columns(2)
            ct3.dataframe(pd.DataFrame({
                "決策變數": [f"x{j+1}" for j in range(7)],
                "項目名稱": ["基本生活開銷", "社交/休閒娛樂", "自我進修/學習", "長期投資配置", "緊急預備金提撥", "固定儲蓄提撥", "個人嗜好支出"],
                "配置金額": [f"{int(round(v)):,} 元" for v in sol_3],
                "佔預算比例": [f"{v/B_val*100:.1f}%" if B_val > 0 else "0%" for v in sol_3]
            }), use_container_width=True, hide_index=True)

            fig_p3 = go.Figure(data=[go.Pie(labels=var_list_3, values=sol_3, hole=0.55, textinfo='label+percent')])
            fig_p3.update_layout(title="自訂參數最佳支出結構佔比", height=300, margin=dict(l=10, r=10, t=35, b=10), showlegend=False)
            cp3.plotly_chart(fig_p3, use_container_width=True)

            st.dataframe(pd.DataFrame([{
                "限制式名稱": con['name'],
                "左側計算值 (LHS)": fmt_num(sum(con['a'][j] * sol_3[j] for j in range(7))),
                "符號": con['op'],
                "右側常數 (RHS)": fmt_num(con['rhs']),
                "鬆弛量 (Slack)": fmt_num(0.0 if abs(sum(con['a'][j] * sol_3[j] for j in range(7)) - con['rhs']) < 1e-4 else abs(con['rhs'] - sum(con['a'][j] * sol_3[j] for j in range(7)))),
                "狀態判定": "Binding (緊束瓶頸)" if abs(sum(con['a'][j] * sol_3[j] for j in range(7)) - con['rhs']) < 1e-4 else "Nonbinding (有餘量)",
                "影子價格 (Shadow Price)": fmt_num(sp_3[k])
            } for k, con in enumerate(constraints_custom)]), use_container_width=True, hide_index=True)