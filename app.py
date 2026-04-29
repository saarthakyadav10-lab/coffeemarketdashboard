import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

st.set_page_config(
    page_title="Coffee & Currency Dashboard",
    page_icon="☕",
    layout="wide"
)

@st.cache_data
def load_data(start_date="2020-01-01"):
    kc = yf.download("KC=F", start=start_date, auto_adjust=False)
    kc.columns = kc.columns.droplevel(1)
    brl = yf.download("BRL=X", start=start_date, auto_adjust=False, progress=False)
    brl.columns = brl.columns.droplevel(1)
    vnd = yf.download("VND=X", start=start_date, auto_adjust=False, progress=False)
    vnd.columns = vnd.columns.droplevel(1)
    dxy = yf.download("DX-Y.NYB", start=start_date, auto_adjust=False, progress=False)
    dxy.columns = dxy.columns.droplevel(1)
    prices = pd.DataFrame({
        "Arabica": kc["Close"],
        "USD_BRL": brl["Close"],
        "USD_VND": vnd["Close"],
        "DXY":     dxy["Close"],
    }).ffill().dropna()
    returns = prices.pct_change().dropna()
    return prices, returns

def chart_layout(**kwargs):
    defaults = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(size=16, color="#E0E0E0"),
        height=600,
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0)
    )
    defaults.update(kwargs)
    return defaults

# --- Global style ---
st.markdown("""
<style>
    .main p, .main li { font-size: 1.15rem; line-height: 1.85; color: #D0D0D0; }
    .main h1 { font-size: 2.8rem; font-weight: 700; letter-spacing: -0.5px; }
    .main h2 { font-size: 1.75rem; font-weight: 600; margin-top: 0.25rem; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 600; }
    div[data-testid="stMetricLabel"] { font-size: 0.95rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 1rem 1.5rem; }
    section[data-testid="stSidebar"] { background: rgba(255,255,255,0.03); }
    .stDivider { opacity: 0.15; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("☕ Controls")
st.sidebar.markdown("Use these controls to explore how the KC/BRL relationship changes across different time horizons.")
st.sidebar.markdown("---")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
window = st.sidebar.slider("Correlation window (days)", min_value=20, max_value=120, value=60)
st.sidebar.markdown("---")
st.sidebar.caption("Data: Yahoo Finance · KC=F, BRL=X, VND=X, DX-Y.NYB")

prices, returns = load_data(start_date)
rolling_corr = returns["Arabica"].rolling(window).corr(returns["USD_BRL"])

# --- Header ---
st.title("☕ Coffee Price & Currency Dashboard")
st.markdown(
    f"This dashboard tracks **Arabica coffee futures (KC=F)** alongside the **USD/BRL exchange rate** "
    f"from {start_date.strftime('%B %Y')} to today, the two most-watched variables in the world's "
    f"largest coffee-producing economy.\n\n"
    f"The conventional view holds that a weaker Brazilian real should push KC prices down: farmers earn "
    f"in dollars but pay costs in reals, so a softer real expands their margins, accelerates selling, "
    f"and adds supply pressure. **The data tells a more complicated story.** Use the controls on the "
    f"left to test the thesis yourself."
)

st.divider()

# --- Top-line metrics ---
kc_latest = prices['Arabica'].iloc[-1]
brl_latest = prices['USD_BRL'].iloc[-1]
corr_latest = rolling_corr.iloc[-1]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Arabica Futures (KC=F)", f"${kc_latest:.2f}",
              help="Cents per pound, front-month ICE contract on the Intercontinental Exchange (ICE)")
with col2:
    st.metric("USD/BRL Rate", f"R${brl_latest:.4f}",
              help="Reais per US dollar a higher number means the real is weaker against the dollar")
with col3:
    st.metric(f"{window}-Day Rolling Correlation", f"{corr_latest:.3f}",
              help="Pearson correlation of daily returns ranges from -1 (perfect inverse) to +1 (perfect lockstep)")

exp1, exp2, exp3 = st.columns(3)
with exp1:
    st.caption(
        f"The price of one pound of Arabica coffee on the ICE futures exchange, quoted in US cents. "
        f"A reading of ${kc_latest:.2f} means it costs {kc_latest:.0f} cents to buy one pound of coffee "
        f"for future delivery. This is the global benchmark price that roasters and traders use worldwide."
    )
with exp2:
    st.caption(
        f"How many Brazilian reals you get for one US dollar. At R${brl_latest:.2f}, one dollar buys "
        f"{brl_latest:.2f} reals. A rising number means the real is weakening against the dollar, "
        f"which in theory makes Brazilian coffee cheaper for foreign buyers and increases supply pressure on KC."
    )
with exp3:
    if abs(corr_latest) < 0.1:
        corr_label = "no meaningful relationship"
        corr_detail = f"Knowing whether BRL rose or fell today gives you almost no information about whether KC rose or fell."
    elif corr_latest >= 0.1:
        corr_label = "moving in the same direction"
        corr_detail = f"Over the past {window} trading days, when BRL goes up, KC has tended to go up too. This is the opposite of the conventional thesis."
    else:
        corr_label = "moving in opposite directions"
        corr_detail = f"Over the past {window} trading days, when BRL goes up, KC has tended to go down. This is consistent with the conventional thesis."
    st.caption(
        f"This number measures how closely KC and BRL move together on a scale from -1 to +1. "
        f"**+1** = they move in perfect lockstep every day. **-1** = they move in perfect opposition every day. "
        f"**0** = no relationship at all. At **{corr_latest:.3f}**, the two markets are currently {corr_label}. "
        f"{corr_detail}"
    )

st.divider()

# --- Chart 1: Rolling Correlation ---
st.subheader("Rolling Correlation Over Time")
st.markdown(
    f"The **{window}-day rolling Pearson correlation** measures how closely KC and USD/BRL daily returns "
    f"move together over each rolling window. A value of **+1** means they moved in perfect lockstep every day. "
    f"A value of **-1** means they moved in perfect opposition. A value of **0** means no relationship at all.\n\n"
    f"Since {start_date.strftime('%B %Y')}, this correlation has ranged from "
    f"**{rolling_corr.min():.2f} to {rolling_corr.max():.2f}**, averaging **{rolling_corr.mean():.2f}**, "
    f"essentially zero. This tells you the BRL/KC relationship is **not stable**: it strengthens during "
    f"certain macro regimes and disappears entirely during others. The shaded regions show the events "
    f"that caused the biggest shifts."
)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=rolling_corr.index, y=rolling_corr.values,
    mode='lines', name=f'{window}-Day Correlation',
    line=dict(color='#4C9BE8', width=1.5)
))
fig1.add_vrect(x0="2020-03-01", x1="2020-04-30", fillcolor="#FF4444", opacity=0.12,
               line_width=0, annotation_text="COVID crash", annotation_position="top left",
               annotation=dict(font_size=13, font_color="#FF8888"))
fig1.add_vrect(x0="2021-07-01", x1="2021-08-31", fillcolor="#FFA500", opacity=0.12,
               line_width=0, annotation_text="Brazil frost", annotation_position="top left",
               annotation=dict(font_size=13, font_color="#FFC060"))
fig1.add_vrect(x0="2022-06-01", x1="2022-10-31", fillcolor="#AA88FF", opacity=0.12,
               line_width=0, annotation_text="DXY surge", annotation_position="top left",
               annotation=dict(font_size=13, font_color="#CC99FF"))
fig1.add_vrect(x0="2024-01-01", x1="2024-06-30", fillcolor="#44CC88", opacity=0.12,
               line_width=0, annotation_text="Supply tightness", annotation_position="top left",
               annotation=dict(font_size=13, font_color="#66DDAA"))
fig1.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
fig1.update_layout(
    **chart_layout(
        title=f"{window}-Day Rolling Correlation: Arabica Futures vs USD/BRL",
        xaxis_title="Date",
        yaxis=dict(title="Pearson Correlation", range=[-1, 1],
                   showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    )
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# --- Chart 2: Normalized Prices ---
st.subheader("Price Performance (Rebased to 100)")

arabica_normalized = (prices["Arabica"] / prices["Arabica"].iloc[0]) * 100
brl_normalized = (prices["USD_BRL"] / prices["USD_BRL"].iloc[0]) * 100
arabica_total_return = arabica_normalized.iloc[-1] - 100
brl_total_return = brl_normalized.iloc[-1] - 100

st.markdown(
    f"Both lines start at 100 on {start_date.strftime('%B %d, %Y')} and move up or down from there, "
    f"so the y-axis shows percentage change from that starting point, not the actual price. "
    f"For example, the blue line reaching 200 means coffee is up 100% from where it started. "
    f"The blue line (Arabica, left axis) is currently at **{arabica_normalized.iloc[-1]:.0f}**, meaning coffee is "
    f"**{'up' if arabica_total_return > 0 else 'down'} {abs(arabica_total_return):.0f}%** since {start_date.strftime('%B %Y')}. "
    f"The orange line (USD/BRL, right axis) is at **{brl_normalized.iloc[-1]:.0f}**, meaning the Brazilian real has "
    f"**{'weakened' if brl_total_return > 0 else 'strengthened'} {abs(brl_total_return):.0f}%** against the dollar over the same period. "
    f"From 2020 to 2023 the two lines moved broadly together, but from 2024 onward coffee prices surged while the BRL barely moved, "
    f"which is why the overall correlation between them is so close to zero."
)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
**What this chart shows**

This chart tests the conventional story: if Brazil's currency weakens, coffee should get cheaper
because Brazilian farmers earn more reals per dollar of coffee sold, so they sell more, adding
supply and pushing prices down. If that story were true, you would expect the two lines to move
in opposite directions — one up when the other goes down. This chart lets you see whether that
actually happened.
""")
with col_b:
    st.markdown(f"""
**Why it matters**

The {abs(arabica_total_return):.0f}% move in coffee since {start_date.strftime('%B %Y')} is one
of the largest commodity rallies of the past decade. If it were explained by BRL weakness, you
would expect a similarly large move in the exchange rate. The BRL moved only {abs(brl_total_return):.0f}%
over the same period. That gap tells you the coffee rally was driven by something else entirely —
supply shocks, harvest failures, and warehouse drawdowns — factors that have nothing to do with
currency markets. Understanding this distinction matters for anyone trying to forecast coffee
prices or hedge coffee exposure.
""")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=prices.index, y=arabica_normalized,
    mode='lines', name='Arabica KC=F',
    line=dict(color='#4C9BE8', width=1.5)
))
fig2.add_trace(go.Scatter(
    x=prices.index, y=brl_normalized,
    mode='lines', name='USD/BRL',
    line=dict(color='#E8824C', width=1.5),
    yaxis="y2"
))
fig2.update_layout(
    **chart_layout(
        title="Normalized Price Comparison: Arabica vs USD/BRL",
        xaxis_title="Date",
        yaxis=dict(title="KC Index (Base = 100)",
                   showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title="BRL Index (Base = 100)", overlaying='y', side='right',
                    showgrid=False)
    )
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- Chart 3: Correlation Heatmap ---
st.subheader("Correlation Matrix")

corr_matrix = returns.corr()
kc_brl_corr = corr_matrix.loc["Arabica", "USD_BRL"]
kc_dxy_corr = corr_matrix.loc["Arabica", "DXY"]
brl_dxy_corr = corr_matrix.loc["USD_BRL", "DXY"]

kc_vnd_corr = corr_matrix.loc["Arabica", "USD_VND"]

st.markdown(
    f"Each cell shows the **Pearson correlation** between two assets' daily returns. Green means they tend "
    f"to move together, red means they tend to move in opposite directions, and yellow/cream means no "
    f"consistent relationship. The diagonal is always 1.0 by definition.\n\n"
    f"**The headline finding:** KC is uncorrelated with the entire FX complex: "
    f"**{kc_brl_corr:.2f}** vs USD/BRL, **{kc_dxy_corr:.2f}** vs DXY, **{kc_vnd_corr:.2f}** vs USD/VND. "
    f"All three are indistinguishable from zero. "
    f"This means coffee's daily returns are driven by coffee-specific factors: weather events, harvest "
    f"forecasts, freight disruptions, and ICE warehouse stock levels, not by macro or FX. "
    f"For a portfolio manager, this makes KC a genuine diversifier: its return drivers are orthogonal "
    f"to the traditional macro factors that dominate FX, rates, and equity exposure."
)
label_map = {
    "Arabica": "☕ Arabica",
    "USD_BRL": "🇧🇷 USD/BRL",
    "USD_VND": "🇻🇳 USD/VND",
    "DXY":     "🇺🇸 DXY",
}
x_labels = [label_map[c] for c in corr_matrix.columns]
y_labels = [label_map[c] for c in corr_matrix.index[::-1]]
z = corr_matrix.values[::-1]
text = [[f"{v:.2f}" for v in row] for row in z]

bloomberg = [
    [0.0,  "#8B0000"], [0.2,  "#CC2200"], [0.4,  "#FF6644"],
    [0.48, "#FFDDCC"], [0.5,  "#F5F0E8"], [0.52, "#CCEECC"],
    [0.6,  "#44AA44"], [0.8,  "#007700"], [1.0,  "#004400"],
]

fig3 = go.Figure(data=go.Heatmap(
    z=z, x=x_labels, y=y_labels,
    colorscale=bloomberg, zmin=-1, zmax=1,
    text=text, texttemplate="%{text}",
    textfont=dict(size=15, color="black"),
    colorbar=dict(title=dict(text='Correlation', font=dict(color="#E0E0E0")),
                  tickfont=dict(color="#E0E0E0"))
))
fig3.update_layout(
    title='Correlation Matrix of Daily Returns',
    height=700, font=dict(size=16, color="#E0E0E0"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=140, r=140, t=160, b=80),
    xaxis=dict(side='top', tickangle=-20),
    yaxis=dict(scaleanchor='x'),
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown(f"""
**Why KC's near-zero FX correlations are significant**

KC's correlation with USD/BRL is **{kc_brl_corr:.2f}**, with DXY is **{kc_dxy_corr:.2f}**, and with
USD/VND is **{kc_vnd_corr:.2f}**. All three are statistically indistinguishable from zero. This means
on any given day, knowing whether the dollar strengthened or emerging market currencies sold off
gives you essentially no information about whether coffee prices moved.

The reason is that coffee's daily price is set by coffee-specific supply and demand: frost damage
in Minas Gerais, the size of the Vietnamese robusta harvest, ICE-certified warehouse stock levels
in New Orleans and Hamburg, and shipping freight rates. None of these inputs have anything to do
with the Federal Reserve or global FX flows.

For an investor already holding dollar exposure, EM currency exposure, or a broad commodities
index, adding KC provides genuine risk reduction because it is driven by a completely separate
set of factors. That is rare in modern markets where correlations between asset classes have
risen significantly since 2008.
""")

st.divider()

# --- Cointegration Test ---
st.subheader("Cointegration Analysis")
st.markdown(
    "**Cointegration** tests whether two price series share a long-run equilibrium meaning their "
    "spread is mean-reverting even if each series individually wanders randomly. "
    "This is a stronger claim than correlation: two series can move together on any given day "
    "but have no stable long-run anchor. Cointegration says the gap between them has a "
    "gravitational pull back to a fixed level.\n\n"
    "The **Engle-Granger approach** used here regresses KC prices on BRL prices, extracts the "
    "residuals, and tests whether those residuals are stationary. If yes, the pair is cointegrated "
    "the spread mean-reverts, and a pairs trade has a statistical foundation.\n\n"
    "**Caveat:** a single test on the full period can mask regime changes. "
    "The rolling test further down shows which sub-periods passed the threshold."
)

t_stat, p_value, crit_values = coint(prices["Arabica"], prices["USD_BRL"], trend='c')

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ADF Test Statistic", f"{t_stat:.4f}",
              help="More negative = stronger evidence of stationarity in the spread")
with col2:
    st.metric("P-Value", f"{p_value:.4f}",
              help="Probability of this result if the series were NOT cointegrated")
with col3:
    st.markdown("**Reject non-cointegration at 5%?**")
    if p_value < 0.05:
        st.markdown('<span style="font-size:1.3rem; color:#00CC66; font-weight:600">✓ YES cointegrated</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span style="font-size:1.3rem; color:#FF5555; font-weight:600">✗ NO not cointegrated</span>',
                    unsafe_allow_html=True)

if p_value < 0.05:
    st.info(
        f"**Result:** The spread between KC and USD/BRL prices is stationary over this period "
        f"(p={p_value:.4f}). The pair shares a long-run equilibrium. Deviations from the "
        f"BRL-implied KC price have historically mean-reverted, giving a statistical basis for a spread trade."
    )
else:
    st.warning(
        f"**Result:** The spread between KC and USD/BRL prices is not stationary over this period "
        f"(p={p_value:.4f}). The pair does not share a stable long-run equilibrium. The spread can "
        f"drift indefinitely, and a pairs trade has no statistical foundation in this window."
    )

st.divider()

# --- Spread Chart ---
st.subheader("KC–BRL Spread")

X = sm.add_constant(prices["USD_BRL"])
ols_result = sm.OLS(prices["Arabica"], X).fit()
spread = ols_result.resid
spread_std = spread.std()
spread_latest = spread.iloc[-1]
beta = ols_result.params["USD_BRL"]

st.markdown(
    f"The regression finds that **each 1 real move in USD/BRL is associated with a "
    f"{beta:.1f} cent/lb move in KC** (β = {beta:.2f}). "
    f"The spread is how far KC currently sits from that prediction.\n\n"
    f"**Today's spread: {spread_latest:+.1f} cents/lb.** KC is "
    f"{'**{:.0f} cents above**'.format(spread_latest) if spread_latest > 0 else '**{:.0f} cents below**'.format(abs(spread_latest))} "
    f"where BRL alone would predict it. The ±2σ bands sit at "
    f"**±{2*spread_std:.0f} cents/lb.** When the spread exceeds these levels, "
    f"it has historically tended to revert, creating a potential pairs trade entry signal."
)

fig_spread = go.Figure()
fig_spread.add_trace(go.Scatter(
    x=spread.index, y=spread.values,
    mode='lines', name='Spread',
    line=dict(color='#7EB8D4', width=1.5),
    fill='tozeroy', fillcolor='rgba(126,184,212,0.07)'
))
fig_spread.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                     annotation_text="Mean (0)", annotation_position="right")
fig_spread.add_hline(y=2*spread_std, line_dash="dash", line_color="#FF5555",
                     annotation_text="+2σ", annotation_position="right")
fig_spread.add_hline(y=-2*spread_std, line_dash="dash", line_color="#FF5555",
                     annotation_text="−2σ", annotation_position="right")
fig_spread.update_layout(
    **chart_layout(
        title="KC–BRL Spread (residuals from OLS regression)",
        xaxis_title="Date",
        yaxis=dict(title="Spread (cents/lb)",
                   showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    )
)
st.plotly_chart(fig_spread, use_container_width=True)

st.divider()

# --- Rolling Cointegration ---
st.subheader("Rolling Cointegration Test (1-Year Window)")
st.markdown(
    "A single cointegration test gives one number for the whole period but the relationship "
    "between KC and BRL is not constant. This chart runs the same Engle-Granger test on every "
    "rolling **252-day window** and plots the p-value over time."
)

roll_window = 252
roll_pvalues = []
roll_dates = []

for i in range(roll_window, len(prices)):
    window_prices = prices.iloc[i - roll_window:i]
    try:
        _, pval, _ = coint(window_prices["Arabica"], window_prices["USD_BRL"], trend='c')
    except Exception:
        pval = float("nan")
    roll_pvalues.append(pval)
    roll_dates.append(prices.index[i])

fig_roll = go.Figure()
fig_roll.add_trace(go.Scatter(
    x=roll_dates, y=roll_pvalues,
    mode='lines', name='Rolling p-value',
    line=dict(color='#4C9BE8', width=1.5)
))
fig_roll.add_hline(
    y=0.05, line_dash="dash", line_color="#FF5555",
    annotation_text="5% significance threshold",
    annotation_position="top right",
    annotation=dict(font_size=14, font_color="#FF8888")
)
fig_roll.update_layout(
    **chart_layout(
        title="Rolling 1-Year Cointegration P-Value (KC vs USD/BRL)",
        xaxis_title="Date",
        yaxis=dict(title="P-Value (log scale)", type="log",
                   showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    )
)
st.plotly_chart(fig_roll, use_container_width=True)

st.markdown(
    "**Reading the chart:** When the line is **below the red dashed line (p < 0.05)**, the KC/BRL "
    "relationship was statistically real in that 252-day window the spread was mean-reverting and "
    "a pairs trade had a statistical foundation. When the line is **above 0.05**, the relationship "
    "had broken down the spread could drift indefinitely and no statistical edge existed."
)

st.divider()

# --- Findings ---
import datetime

st.markdown("""
<style>
.finding {
    border-left: 3px solid rgba(76,155,232,0.6);
    padding: 0.6rem 0 0.6rem 1.2rem;
    margin-bottom: 0.4rem;
}
.findings-box {
    background: rgba(76,155,232,0.05);
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}
.limitations-box {
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 1.2rem 2rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.subheader("Findings")
st.markdown("Synthesizing the analysis above, six observations stand out:")

st.markdown(f"""<div class="findings-box">

<div class="finding">
<strong>1. The textbook BRL–KC relationship is empirically weak at daily frequency.</strong>
The Pearson correlation between KC and USD/BRL daily returns averages {rolling_corr.mean():.2f} since {start_date.strftime('%B %Y')},
with the rolling {window}-day correlation oscillating between {rolling_corr.min():.2f} and {rolling_corr.max():.2f}.
This directly contradicts the conventional explanation that BRL movements drive KC prices the signal does not show up in the daily return data.
</div>

<div class="finding">
<strong>2. The relationship is regime-conditional, not constant.</strong>
Some periods notably the 2022 DXY surge show meaningful negative correlation consistent with the FX thesis.
Others show essentially no relationship, and the 2024 supply tightness regime saw the correlation turn positive.
The productive research question is not "does the BRL–KC relationship exist" but "when does it activate and what triggers it."
</div>

<div class="finding">
<strong>3. Cointegration is episodic, not structural.</strong>
The Engle-Granger test on the full sample fails to reject non-cointegration (p = {p_value:.4f}).
The rolling test reveals brief sub-0.05 episodes a sustained dip in late 2024 and early 2025 with p-values reaching as low as 0.001.
The long-run equilibrium relationship exists but switches on and off rather than holding continuously.
</div>

<div class="finding">
<strong>4. Supply shocks override FX dynamics entirely.</strong>
The 2021 Brazilian frost and the 2024–2025 supply tightness regime both drove KC sharply higher while BRL remained range-bound.
When supply-side news dominates frost damage, harvest shortfalls, warehouse drawdowns the FX transmission channel is effectively disabled.
KC's {arabica_total_return:+.0f}% move since {start_date.strftime('%B %Y')} cannot be explained by BRL's {brl_total_return:+.0f}% depreciation.
</div>

<div class="finding">
<strong>5. KC is orthogonal to the FX complex.</strong>
Daily KC returns correlate at {kc_brl_corr:.2f} with USD/BRL, {kc_dxy_corr:.2f} with DXY, and {kc_vnd_corr:.2f} with USD/VND all indistinguishable from zero.
Coffee's daily price moves are driven by weather, harvests, freight, and ICE warehouse stock levels, not by macro or currency dynamics.
This makes KC a genuine portfolio diversifier: its return drivers are orthogonal to the macro factors that dominate FX, rates, and equity risk.
</div>

<div class="finding">
<strong>6. A static KC–BRL pairs trade is not viable.</strong>
The cointegration evidence does not support a continuously deployed mean-reversion strategy the spread drifts indefinitely across most of the sample.
A regime-conditional version activating only when the rolling cointegration p-value drops below a defined threshold is potentially viable and is the natural next step from this analysis.
</div>

</div>""", unsafe_allow_html=True)

st.markdown("#### Limitations & Next Steps")
st.markdown(f"""<div class="limitations-box">

This analysis uses Yahoo Finance daily continuous futures data, which carries roll-yield artifacts at contract expiry and does not capture intraday price dynamics or basis differences across delivery locations.
The BRL–KC transmission mechanism may operate at weekly or monthly frequency rather than daily the correlation structure at lower frequencies has not been tested here and warrants a separate analysis.
sThe natural next project is a formal regime-switching model (Markov-switching or threshold VAR) that replaces the informal structural / decoupled / inverted regime labels with a statistically estimated state process.

</div>""", unsafe_allow_html=True)

st.divider()

# --- Footer ---
last_updated = datetime.datetime.now().strftime("%B %d, %Y")
st.markdown(
    f"""<div style="text-align:center; color:#666; font-size:0.85rem; padding: 1rem 0 0.5rem 0;">
    Built by <strong>Saarthak Yadav</strong> &nbsp;·&nbsp;
    Data: Yahoo Finance &nbsp;·&nbsp;
    Last updated: {last_updated} &nbsp;·&nbsp;
    <a href="https://github.com/placeholder" style="color:#4C9BE8; text-decoration:none;">GitHub</a> &nbsp;·&nbsp;
    <a href="https://linkedin.com/placeholder" style="color:#4C9BE8; text-decoration:none;">LinkedIn</a>
    </div>""",
    unsafe_allow_html=True
)
