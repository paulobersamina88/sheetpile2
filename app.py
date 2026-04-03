import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Sheet Pile Design App', layout='wide')

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def deg(x):
    return math.radians(x)


def rankine_active(phi_deg: float, beta_deg: float = 0.0) -> float:
    phi = deg(phi_deg)
    beta = deg(beta_deg)
    if abs(beta_deg) < 1e-9:
        return math.tan(math.radians(45) - phi / 2) ** 2
    c = math.cos(beta)
    val = c * (c - math.sqrt(max(c * c - math.cos(phi) ** 2, 1e-12))) / (c + math.sqrt(max(c * c - math.cos(phi) ** 2, 1e-12)))
    return max(val, 1e-6)


def rankine_passive(phi_deg: float) -> float:
    phi = deg(phi_deg)
    return math.tan(math.radians(45) + phi / 2) ** 2


def mononobe_okabe_delta_ka(phi_deg: float, kh: float, kv: float = 0.0) -> float:
    """
    Simplified implementation used for engineering screening.
    Returns increment in active pressure coefficient.
    Conservative fallback when exact M-O geometry is unstable.
    """
    phi = deg(phi_deg)
    # Simplified inertial angle
    psi = math.atan2(kh, max(1.0 - kv, 1e-6))
    ka = rankine_active(phi_deg)
    try:
        # Simplified form for vertical wall, level backfill, wall friction = 0
        num = math.cos(phi - psi) ** 2
        den = math.cos(psi) * math.cos(psi) * (
            1 + math.sqrt(max(math.sin(phi) * math.sin(phi - psi) / max(math.cos(psi) ** 2, 1e-9), 0.0))
        ) ** 2
        kae = num / max(den, 1e-9)
        return max(kae - ka, 0.0)
    except Exception:
        return max(0.75 * kh, 0.0)


def trapezoid_resultant(p_top: float, p_bot: float, height: float):
    """Resultant and centroid from top for a trapezoid."""
    w = 0.5 * (p_top + p_bot) * height
    if abs(p_top + p_bot) < 1e-12:
        y = height / 2
    else:
        y = height * (p_top + 2 * p_bot) / (3 * (p_top + p_bot))
    return w, y


def net_pressures(gamma, H, D, Ka, Kp, q=0.0, delta_ka=0.0):
    y1 = np.linspace(0, H, 400)
    p_above = Ka * gamma * y1 + Ka * q + delta_ka * gamma * y1

    z = np.linspace(0, D, 400)
    p_below = Ka * gamma * (H + z) + Ka * q + delta_ka * gamma * (H + z) - Kp * gamma * z
    return y1, p_above, z, p_below


def force_moment_above_dredge(gamma, H, Ka, q=0.0, delta_ka=0.0):
    p_top = Ka * q
    p_bot = Ka * gamma * H + Ka * q + delta_ka * gamma * H
    P, ybar = trapezoid_resultant(p_top, p_bot, H)
    M = P * (H - ybar)  # moment about dredge line
    return P, M, p_top, p_bot


def force_moment_below_dredge(gamma, H, D, Ka, Kp, q=0.0, delta_ka=0.0):
    # integrate numerically for robustness
    z = np.linspace(0, D, 2000)
    p = Ka * gamma * (H + z) + Ka * q + delta_ka * gamma * (H + z) - Kp * gamma * z
    F = np.trapz(p, z)
    M = np.trapz(p * z, z)
    return F, M


def solve_embedment(gamma, H, Ka, Kp, q=0.0, delta_ka=0.0, fs=1.3):
    P_top, M_top, _, _ = force_moment_above_dredge(gamma, H, Ka, q, delta_ka)
    # free earth support style screening solution based on moment equilibrium about tip
    def f(D):
        F_b, M_b = force_moment_below_dredge(gamma, H, D, Ka, Kp, q, delta_ka)
        # moment about tip: top load contributes P_top*(D + lever from dredge)
        Mtop_tip = M_top + P_top * D
        Mb_tip = F_b * D - M_b
        return Mb_tip - Mtop_tip

    Ds = np.linspace(max(0.5, 0.2 * H), max(12.0, 3.5 * H), 1200)
    vals = [f(d) for d in Ds]
    root = None
    for i in range(len(Ds) - 1):
        if vals[i] == 0:
            root = Ds[i]
            break
        if vals[i] * vals[i + 1] < 0:
            a, b = Ds[i], Ds[i + 1]
            for _ in range(60):
                c = 0.5 * (a + b)
                fc = f(c)
                if f(a) * fc <= 0:
                    b = c
                else:
                    a = c
            root = 0.5 * (a + b)
            break
    if root is None:
        root = 0.7 * H

    D_req = fs * root
    return root, D_req


def shear_moment_distribution(gamma, H, D, Ka, Kp, q=0.0, delta_ka=0.0):
    x = np.linspace(0, H + D, 1400)  # from top
    p = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= H:
            p[i] = Ka * gamma * xi + Ka * q + delta_ka * gamma * xi
        else:
            z = xi - H
            p[i] = Ka * gamma * (H + z) + Ka * q + delta_ka * gamma * (H + z) - Kp * gamma * z

    # integrate from top with V(0)=0, M(0)=0 then shift so tip moment = 0 approximately
    V = np.zeros_like(x)
    M = np.zeros_like(x)
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        V[i] = V[i - 1] + 0.5 * (p[i] + p[i - 1]) * dx
        M[i] = M[i - 1] + 0.5 * (V[i] + V[i - 1]) * dx
    # Make bending at tip zero by subtracting linear correction
    M_corr = M - (M[-1] / max(x[-1], 1e-9)) * x
    V_corr = np.gradient(M_corr, x)
    return x, p, V_corr, M_corr


def aisc_compression_strength(Fy, E, A, rx, ry, KLx, KLy):
    rmin = max(min(rx, ry), 1e-6)
    slender = max(KLx / max(rx, 1e-6), KLy / max(ry, 1e-6))
    Fe = (math.pi ** 2 * E) / (slender ** 2)
    if Fy / Fe <= 2.25:
        Fcr = (0.658 ** (Fy / Fe)) * Fy
    else:
        Fcr = 0.877 * Fe
    return Fcr * A, slender, Fe, Fcr


def aisc_h1_interaction(Pu, Mux, Muy, phiPn, phiMnx, phiMny, LRFD=True):
    # Simplified AISC H1-1 screening for doubly symmetric members in compression and flexure
    if phiPn <= 0 or phiMnx <= 0 or phiMny <= 0:
        return np.nan
    ratioP = Pu / phiPn
    if ratioP < 0.2:
        return ratioP / 2 + Mux / phiMnx + Muy / phiMny
    return ratioP + (8.0 / 9.0) * (Mux / phiMnx + Muy / phiMny)


def format_float(v, n=3):
    try:
        return f"{float(v):,.{n}f}"
    except Exception:
        return "-"


# -------------------------------------------------
# Data loading
# -------------------------------------------------
DATA = Path(__file__).parent / 'data' / 'sample_sheet_piles.csv'
if DATA.exists():
    sections_df = pd.read_csv(DATA)
else:
    sections_df = pd.DataFrame(columns=['section', 'type', 'A_mm2', 'Sx_mm3', 'Zx_mm3', 'Ix_mm4', 'rx_mm', 'ry_mm', 'Fy_MPa', 'weight_kg_per_m'])

# -------------------------------------------------
# UI
# -------------------------------------------------

st.title('Steel Sheet Pile Design – Streamlit App')
st.caption('IBC 2024 / AISC 360-22 oriented screening tool with optional NSCP-compatible workflow. Verify all results with project geotechnical report, manufacturer sheet pile tables, and local code requirements.')

with st.expander('Design basis and current scope', expanded=False):
    st.markdown(
        """
        - Lateral soil pressure: Rankine active/passive earth pressure.
        - Uniform surcharge: converted to constant lateral pressure using active coefficient.
        - Seismic ground shaking: optional Mononobe-Okabe-style increment for screening.
        - Embedment: free-earth-support style cantilever screening approach.
        - Section force extraction: pressure, shear, and moment diagrams with focus at dredge/cantilever bottom.
        - Section selection: built-in editable CSV starter library.
        - Steel check: simplified AISC combined compression and bending interaction.

        This app is intended for preliminary design and education. It does **not** replace a full geotechnical and structural design package.
        """
    )

col1, col2, col3 = st.columns(3)
with col1:
    H = st.number_input('Retained height above dredge line, H (m)', min_value=0.5, value=5.0, step=0.1)
    gamma = st.number_input('Effective soil unit weight, γ (kN/m³)', min_value=5.0, value=18.0, step=0.1)
    phi = st.number_input('Soil friction angle, φ (deg)', min_value=10.0, max_value=50.0, value=30.0, step=0.5)
    beta = st.number_input('Backfill slope β (deg, 0 for level)', min_value=0.0, max_value=20.0, value=0.0, step=0.5)
    q = st.number_input('Uniform surcharge q (kPa)', min_value=0.0, value=10.0, step=1.0)

with col2:
    use_seismic = st.checkbox('Include seismic increment', value=True)
    kh = st.number_input('Horizontal seismic coefficient kh', min_value=0.0, max_value=1.0, value=0.15, step=0.01)
    kv = st.number_input('Vertical seismic coefficient kv', min_value=-0.5, max_value=0.5, value=0.0, step=0.01)
    FS_embed = st.number_input('Embedment safety multiplier', min_value=1.0, value=1.3, step=0.05)
    K_factor = st.number_input('Effective length factor K for steel member', min_value=0.5, value=1.0, step=0.1)

with col3:
    E = st.number_input('Steel modulus E (MPa)', min_value=100000.0, value=200000.0, step=1000.0)
    unbraced = st.number_input('Unbraced length for steel check (m)', min_value=0.5, value=H + 1.0, step=0.1)
    design_method = st.selectbox('Steel design format', ['LRFD', 'ASD'])
    phi_c = 0.9 if design_method == 'LRFD' else 1 / 1.67
    phi_b = 0.9 if design_method == 'LRFD' else 1 / 1.67

Ka = rankine_active(phi, beta)
Kp = rankine_passive(phi)
delta_ka = mononobe_okabe_delta_ka(phi, kh, kv) if use_seismic else 0.0

D_raw, D_req = solve_embedment(gamma, H, Ka, Kp, q=q, delta_ka=delta_ka, fs=FS_embed)
P_top, M_top, p_top, p_bot = force_moment_above_dredge(gamma, H, Ka, q=q, delta_ka=delta_ka)

x, p, V, M = shear_moment_distribution(gamma, H, D_req, Ka, Kp, q=q, delta_ka=delta_ka)
ix_dredge = np.argmin(np.abs(x - H))
Mmax = float(np.max(np.abs(M)))
Vmax = float(np.max(np.abs(V)))
Mdredge = float(M[ix_dredge])
Vdredge = float(V[ix_dredge])

st.subheader('1) Soil pressure and embedment summary')
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric('Ka', format_float(Ka, 3))
s2.metric('Kp', format_float(Kp, 3))
s3.metric('ΔKa,seismic', format_float(delta_ka, 3))
s4.metric('Trial embedment D0 (m)', format_float(D_raw, 3))
s5.metric('Required embedment Dreq (m)', format_float(D_req, 3))

c1, c2 = st.columns([1.1, 1])
with c1:
    st.markdown('**Loads above dredge line**')
    st.write({
        'Top lateral pressure at grade (kPa)': round(p_top, 3),
        'Bottom pressure at dredge line (kPa)': round(p_bot, 3),
        'Resultant above dredge P (kN/m)': round(P_top, 3),
        'Moment at dredge due to above-grade load (kN·m/m)': round(M_top, 3),
        'Shear at dredge from integrated diagram Vd (kN/m)': round(Vdredge, 3),
        'Moment at dredge from integrated diagram Md (kN·m/m)': round(Mdredge, 3),
        'Maximum moment anywhere Mmax (kN·m/m)': round(Mmax, 3),
        'Maximum shear anywhere Vmax (kN/m)': round(Vmax, 3),
    })

with c2:
    st.markdown('**Approximate bottom / cantilever checks**')
    st.write({
        'Depth to tip below dredge (m)': round(D_req, 3),
        'Estimated moment at cantilever bottom / dredge line (kN·m/m)': round(abs(Mdredge), 3),
        'Estimated shear at cantilever bottom / dredge line (kN/m)': round(abs(Vdredge), 3),
    })

st.subheader('2) Section selection and AISC combined check')

if sections_df.empty:
    st.warning('No section library found. Add data/sample_sheet_piles.csv with section properties.')
else:
    sect_name = st.selectbox('Select sheet pile section', sections_df['section'].tolist())
    sect = sections_df.loc[sections_df['section'] == sect_name].iloc[0]

    A = float(sect['A_mm2'])
    Sx = float(sect['Sx_mm3'])
    Zx = float(sect.get('Zx_mm3', Sx))
    Ix = float(sect['Ix_mm4'])
    rx = float(sect['rx_mm'])
    ry = float(sect['ry_mm'])
    Fy = float(sect['Fy_MPa'])

    axial_kN_per_m = st.number_input('Factored / service axial compression on pile, P (kN per m wall)', min_value=0.0, value=50.0, step=5.0)
    tributary_width = st.number_input('Section spacing / tributary width used for one pile (m)', min_value=0.1, value=1.0, step=0.1)

    P_member = axial_kN_per_m * tributary_width
    Mu_member = Mmax * tributary_width
    Vu_member = Vmax * tributary_width

    Pn_N, slender, Fe, Fcr = aisc_compression_strength(Fy, E, A, rx, ry, K_factor * unbraced * 1000, K_factor * unbraced * 1000)
    Pn_kN = Pn_N / 1000.0
    Mn_Nmm = Fy * Zx
    Mn_kNm = Mn_Nmm / 1e6

    phiPn = phi_c * Pn_kN
    phiMn = phi_b * Mn_kNm
    interaction = aisc_h1_interaction(P_member, Mu_member, 0.0, phiPn, phiMn, 1e18)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric('Selected section', sect_name)
    r2.metric('φPn or allowable P (kN)', format_float(phiPn, 1))
    r3.metric('φMn or allowable M (kN·m)', format_float(phiMn, 1))
    r4.metric('H1 interaction ratio', format_float(interaction, 3))

    st.dataframe(pd.DataFrame([
        {
            'Section': sect_name,
            'Type': sect['type'],
            'A (mm²)': A,
            'Sx (mm³)': Sx,
            'Zx (mm³)': Zx,
            'Ix (mm⁴)': Ix,
            'rx (mm)': rx,
            'ry (mm)': ry,
            'Fy (MPa)': Fy,
            'Weight (kg/m)': sect['weight_kg_per_m'],
            'Steel slenderness KL/r': round(slender, 2),
            'Euler stress Fe (MPa)': round(Fe, 2),
            'Critical stress Fcr (MPa)': round(Fcr, 2),
            'Pile force Pused (kN)': round(P_member, 2),
            'Pile moment Mused (kN·m)': round(Mu_member, 2),
            'Pile shear Vused (kN)': round(Vu_member, 2),
            'Interaction ratio': round(interaction, 3),
            'Status': 'OK' if interaction <= 1.0 else 'NG'
        }
    ]), use_container_width=True)

st.subheader('3) Charts')
chart_df = pd.DataFrame({
    'Depth from grade (m)': x,
    'Lateral pressure (kPa)': p,
    'Shear (kN/m)': V,
    'Moment (kN·m/m)': M,
})

pc1, pc2, pc3 = st.columns(3)
with pc1:
    st.line_chart(chart_df.set_index('Depth from grade (m)')[['Lateral pressure (kPa)']])
with pc2:
    st.line_chart(chart_df.set_index('Depth from grade (m)')[['Shear (kN/m)']])
with pc3:
    st.line_chart(chart_df.set_index('Depth from grade (m)')[['Moment (kN·m/m)']])

st.subheader('4) Calculation notes')
st.markdown(
    f"""
    - Active earth pressure uses Rankine coefficient **Ka = {Ka:.3f}**.
    - Passive resistance uses Rankine coefficient **Kp = {Kp:.3f}**.
    - Uniform surcharge contributes **Ka × q = {Ka*q:.3f} kPa** as constant lateral pressure.
    - Seismic increment uses a simplified Mononobe-Okabe style increment **ΔKa = {delta_ka:.3f}**.
    - Required embedment is obtained from an equilibrium-based cantilever screening method and then multiplied by the user-selected safety multiplier **{FS_embed:.2f}**.
    - Steel strength check is a simplified AISC beam-column interaction screening using axial compression and major-axis bending.
    - Final design should also evaluate groundwater, corrosion allowance, constructability, interlock strength, serviceability/deflection, passive reduction factors, and project-specific geotechnical recommendations.
    """
)

st.subheader('5) Download design summary CSV')
summary = pd.DataFrame([
    {
        'H_m': H,
        'gamma_kN_m3': gamma,
        'phi_deg': phi,
        'beta_deg': beta,
        'q_kPa': q,
        'kh': kh if use_seismic else 0.0,
        'kv': kv if use_seismic else 0.0,
        'Ka': Ka,
        'Kp': Kp,
        'delta_ka': delta_ka,
        'embedment_raw_m': D_raw,
        'embedment_required_m': D_req,
        'P_top_kN_per_m': P_top,
        'M_dredge_kNm_per_m': Mdredge,
        'V_dredge_kN_per_m': Vdredge,
        'Mmax_kNm_per_m': Mmax,
        'Vmax_kN_per_m': Vmax,
    }
])

st.download_button(
    'Download summary CSV',
    summary.to_csv(index=False).encode('utf-8'),
    file_name='sheet_pile_design_summary.csv',
    mime='text/csv'
)
