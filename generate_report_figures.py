"""
Generate all figures for the FYP report from actual CSV result files.
Run this script to produce PNG files in the figures/ directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results", "experiments")
PRIME_DIR = os.path.join(BASE, "results", "prime")
SWITCH_DIR = os.path.join(BASE, "results", "switch_modelling")
FIGURES_DIR = os.path.join(BASE, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

COLORS = {
    'scratch':       '#2196F3',
    'transfer':      '#F44336',
    'frozen':        '#4CAF50',
    'multitransfer': '#FF9800',
    'prime':         '#9C27B0',
    'ensemble':      '#795548',
}
WEEKS = [1, 2, 4, 8, 16, 32, 64, 104]

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})


def load_csv(path):
    try:
        df = pd.read_csv(path)
        df = df[pd.to_numeric(df['mae'], errors='coerce').notna()].copy()
        df['weeks'] = df['weeks'].astype(int)
        return df
    except Exception:
        return None


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, name), dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved {name}")


# ── FIG 1: Rat/Education (Colin → Denise) ────────────────────────────────────
print("Generating fig01 …")
fig, ax = plt.subplots(figsize=(8, 5))
exp = os.path.join(RESULTS, "rat_education")
scratch  = load_csv(os.path.join(exp, "data_efficiency_pretransfer.csv"))
transfer = load_csv(os.path.join(exp, "data_efficiency_transfer.csv"))
frozen   = load_csv(os.path.join(exp, "data_efficiency_frozen.csv"))
ax.plot(scratch['weeks'],  scratch['mae'],  'o-', color=COLORS['scratch'],   label='Scratch',         lw=2, ms=6)
ax.plot(transfer['weeks'], transfer['mae'], 's-', color=COLORS['transfer'],  label='Full Fine-Tuning', lw=2, ms=6)
ax.plot(frozen['weeks'],   frozen['mae'],   '^-', color=COLORS['frozen'],    label='Frozen Backbone',  lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh)')
ax.set_title('Rat/Education: Colin → Denise — Data Efficiency')
ax.legend(); ax.grid(True, alpha=0.3)
savefig('fig01_rat_education_efficiency.png')

# ── FIG 2: Rat/Education New (Theo → Lee) ────────────────────────────────────
print("Generating fig02 …")
fig, ax = plt.subplots(figsize=(8, 5))
exp = os.path.join(RESULTS, "rat_education_new")
scratch  = load_csv(os.path.join(exp, "data_efficiency_pretransfer.csv"))
transfer = load_csv(os.path.join(exp, "data_efficiency_transfer.csv"))
frozen   = load_csv(os.path.join(exp, "data_efficiency_frozen.csv"))
ax.plot(scratch['weeks'],  scratch['mae'],  'o-', color=COLORS['scratch'],   label='Scratch',         lw=2, ms=6)
ax.plot(transfer['weeks'], transfer['mae'], 's-', color=COLORS['transfer'],  label='Full Fine-Tuning', lw=2, ms=6)
ax.plot(frozen['weeks'],   frozen['mae'],   '^-', color=COLORS['frozen'],    label='Frozen Backbone',  lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh)')
ax.set_title('Rat/Education: Theo → Lee — Data Efficiency')
ax.legend(); ax.grid(True, alpha=0.3)
savefig('fig02_rat_education_new_efficiency.png')

# ── FIG 3: Eagle/Education — KEY FIGURE (dual panel) ─────────────────────────
print("Generating fig03 …")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
exp = os.path.join(RESULTS, "eagle_education")
scratch  = load_csv(os.path.join(exp, "data_efficiency_pretransfer.csv"))
transfer = load_csv(os.path.join(exp, "data_efficiency_transfer.csv"))
frozen   = load_csv(os.path.join(exp, "data_efficiency_frozen.csv"))
for ax in (ax1, ax2):
    ax.semilogy(scratch['weeks'],  scratch['mae'],  'o-', color=COLORS['scratch'],  label='Scratch',         lw=2, ms=6)
    ax.semilogy(transfer['weeks'], transfer['mae'], 's-', color=COLORS['transfer'], label='Full Fine-Tuning', lw=2, ms=6)
    ax.semilogy(frozen['weeks'],   frozen['mae'],   '^-', color=COLORS['frozen'],   label='Frozen Backbone',  lw=2, ms=6)
    ax.set_xscale('log', base=2)
    ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
    ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh, log scale)')
    ax.legend(); ax.grid(True, alpha=0.3, which='both')
ax1.set_title('Eagle/Education: Samantha → Brooke\n(Full Scale — log y-axis)')
ax2.set_ylim(10, 1000)
ax2.set_title('Eagle/Education: Samantha → Brooke\n(Zoomed — log y-axis)')
fig.suptitle('Full Fine-Tuning catastrophically fails on Eagle/Brooke; Frozen Backbone excels',
             fontweight='bold', y=1.02)
savefig('fig03_eagle_education_efficiency.png')

# ── FIG 4: Lamb/Education ─────────────────────────────────────────────────────
print("Generating fig04 …")
fig, ax = plt.subplots(figsize=(8, 5))
exp = os.path.join(RESULTS, "lamb_education")
scratch  = load_csv(os.path.join(exp, "data_efficiency_pretransfer.csv"))
transfer = load_csv(os.path.join(exp, "data_efficiency_transfer.csv"))
frozen   = load_csv(os.path.join(exp, "data_efficiency_frozen.csv"))
ax.plot(scratch['weeks'],  scratch['mae'],  'o-', color=COLORS['scratch'],  label='Scratch',         lw=2, ms=6)
ax.plot(transfer['weeks'], transfer['mae'], 's-', color=COLORS['transfer'], label='Full Fine-Tuning', lw=2, ms=6)
ax.plot(frozen['weeks'],   frozen['mae'],   '^-', color=COLORS['frozen'],   label='Frozen Backbone',  lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh)')
ax.set_title('Lamb/Education: Lucas → Mae — Data Efficiency')
ax.legend(); ax.grid(True, alpha=0.3)
savefig('fig04_lamb_education_efficiency.png')

# ── FIG 5: Office ─────────────────────────────────────────────────────────────
print("Generating fig05 …")
fig, ax = plt.subplots(figsize=(8, 5))
exp = os.path.join(RESULTS, "office_any")
scratch  = load_csv(os.path.join(exp, "data_efficiency_pretransfer.csv"))
transfer = load_csv(os.path.join(exp, "data_efficiency_transfer.csv"))
frozen   = load_csv(os.path.join(exp, "data_efficiency_frozen.csv"))
ax.plot(scratch['weeks'],  scratch['mae'],  'o-', color=COLORS['scratch'],  label='Scratch',         lw=2, ms=6)
ax.plot(transfer['weeks'], transfer['mae'], 's-', color=COLORS['transfer'], label='Full Fine-Tuning', lw=2, ms=6)
ax.plot(frozen['weeks'],   frozen['mae'],   '^-', color=COLORS['frozen'],   label='Frozen Backbone',  lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh)')
ax.set_title('Office: Miriam → Denita — Data Efficiency')
ax.legend(); ax.grid(True, alpha=0.3)
savefig('fig05_office_efficiency.png')

# ── FIG 6: Lodging ────────────────────────────────────────────────────────────
print("Generating fig06 …")
fig, ax = plt.subplots(figsize=(8, 5))
exp = os.path.join(RESULTS, "lodging_any")
scratch  = load_csv(os.path.join(exp, "data_efficiency_pretransfer.csv"))
transfer = load_csv(os.path.join(exp, "data_efficiency_transfer.csv"))
frozen   = load_csv(os.path.join(exp, "data_efficiency_frozen.csv"))
ax.plot(scratch['weeks'],  scratch['mae'],  'o-', color=COLORS['scratch'],  label='Scratch',         lw=2, ms=6)
ax.plot(transfer['weeks'], transfer['mae'], 's-', color=COLORS['transfer'], label='Full Fine-Tuning', lw=2, ms=6)
ax.plot(frozen['weeks'],   frozen['mae'],   '^-', color=COLORS['frozen'],   label='Frozen Backbone',  lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh)')
ax.set_title('Lodging: Celia → Oliva — Data Efficiency')
ax.legend(); ax.grid(True, alpha=0.3)
savefig('fig06_lodging_efficiency.png')

# ── FIG 7: 8-Week Snapshot Bar Chart ─────────────────────────────────────────
print("Generating fig07 …")
labels = ['Rat/Colin→Denise', 'Rat/Theo→Lee', 'Eagle/Sam→Brooke',
          'Lamb/Lucas→Mae',   'Hog/Miriam→Denita', 'Robin/Celia→Oliva']
scratch_8  = [18.24, 71.84, 77.71, 38.12, 27.62, 13.11]
ft_8       = [15.35, 67.29, 335.36, 27.08, 20.80, 12.47]
frozen_8   = [17.21, 69.03,  50.87, 32.29, 24.22, 12.65]
x = np.arange(len(labels)); w = 0.25
fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - w, scratch_8, w, label='Scratch',         color=COLORS['scratch'],  alpha=0.85)
ax.bar(x,     [min(v, 200) for v in ft_8], w, label='Full Fine-Tuning (clipped at 200)', color=COLORS['transfer'], alpha=0.85)
ax.bar(x + w, frozen_8,  w, label='Frozen Backbone', color=COLORS['frozen'],   alpha=0.85)
ax.text(2, 205, 'FT=335 kWh ↑', ha='center', fontsize=8, color='red', fontweight='bold')
ax.set_ylim(0, 230)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel('MAE (kWh) at 8 weeks')
ax.set_title('8-Week Snapshot: MAE across All Six Core Experiments\n(Eagle/Brooke Full Fine-Tuning clipped at 200 kWh for display)')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
savefig('fig07_8week_snapshot.png')

# ── FIG 8: Transfer Benefit Heatmap ──────────────────────────────────────────
print("Generating fig08 …")
week_labels = ['1w','2w','4w','8w','16w','32w','64w']
exp_labels  = ['Rat/Colin→Denise','Rat/Theo→Lee','Eagle/Sam→Brooke',
               'Lamb/Lucas→Mae','Hog/Miriam→Denita','Robin/Celia→Oliva']
# Benefit = (Scratch - FT) / Scratch * 100
benefits = np.array([
    [-21.9,  9.1,  4.4, 15.8, -5.5, -9.9,  0.5],
    [ 67.5, 55.3,  9.0,  6.3,  6.9, -7.6, -7.4],
    [ 28.2,  0.7,-1255.7,-331.4,16.6,11.2,  6.2],
    [-42.5, 20.2, 21.7, 29.0,  7.9,  2.8, -5.1],
    [  4.4,  1.3,  5.2, 24.7,  5.8,  5.5,-10.9],
    [  7.2,  0.9,  4.3,  4.9,  2.5,  1.4,  1.6],
])
clipped = np.clip(benefits, -60, 60)
fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(clipped, cmap='RdYlGn', vmin=-60, vmax=60, aspect='auto')
ax.set_xticks(range(7)); ax.set_xticklabels(week_labels)
ax.set_yticks(range(6)); ax.set_yticklabels(exp_labels)
plt.colorbar(im, ax=ax, label='Benefit (%) [green = better than Scratch]')
ax.set_title('Full Fine-Tuning Benefit (%) over Scratch [clipped to ±60%]\nEagle/Brooke 4w actual: −1256%; 8w actual: −331%')
for i in range(6):
    for j in range(7):
        v = benefits[i, j]
        txt = f'{v:.0f}' if abs(v) > 60 else f'{v:.1f}'
        ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                color='black' if abs(clipped[i,j]) < 40 else 'white')
savefig('fig08_transfer_benefit_heatmap.png')

# ── FIG 9: N-Source Ablation ──────────────────────────────────────────────────
print("Generating fig09 …")
ns = [1, 2, 3, 4, 5, 10, 15]
mae_4 = []; mae_8 = []
for n in ns:
    df = load_csv(os.path.join(RESULTS,"multitransfer_ablation",f"data_efficiency_multitransfer_n{n}.csv"))
    mae_4.append(df[df['weeks']==4]['mae'].values[0] if df is not None else np.nan)
    mae_8.append(df[df['weeks']==8]['mae'].values[0] if df is not None else np.nan)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for ax, vals, sc, fr, wk in [(ax1, mae_4, 40.20, 48.47, 4), (ax2, mae_8, 77.71, 50.87, 8)]:
    ax.plot(ns, vals, 'D-', color=COLORS['multitransfer'], label='Multi-Source FT', lw=2, ms=8)
    ax.axhline(sc, color=COLORS['scratch'], ls='--', lw=2, label=f'Scratch ({sc:.1f} kWh)')
    ax.axhline(fr, color=COLORS['frozen'],  ls='--', lw=2, label=f'Frozen Backbone ({fr:.1f} kWh)')
    ax.set_xlabel('Number of Source Buildings (N)'); ax.set_ylabel('MAE (kWh)')
    ax.set_title(f'N-Source Ablation: Eagle/Brooke\n({wk}-week training)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(ns)
fig.suptitle('N-Source Ablation: Multi-Source FT vs Baselines\n'
             'Frozen Backbone outperforms all Multi-Source FT configurations at both data regimes',
             fontweight='bold')
savefig('fig09_nsource_ablation.png')

# ── FIG 10: Multi-Transfer on Eagle/Brooke ────────────────────────────────────
print("Generating fig10 …")
exp_mt    = os.path.join(RESULTS, "multi_transfer")
scratch   = load_csv(os.path.join(exp_mt, "data_efficiency_pretransfer.csv"))
single_ft = load_csv(os.path.join(exp_mt, "data_efficiency_transfer.csv"))
multi_ft  = load_csv(os.path.join(exp_mt, "data_efficiency_multitransfer.csv"))
frozen_e  = load_csv(os.path.join(RESULTS, "eagle_education", "data_efficiency_frozen.csv"))
fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogy(scratch['weeks'],   scratch['mae'],   'o-', color=COLORS['scratch'],       label='Scratch',                lw=2, ms=6)
ax.semilogy(single_ft['weeks'], single_ft['mae'], 's-', color=COLORS['transfer'],      label='Single-Source FT (Samantha)',lw=2, ms=6)
ax.semilogy(multi_ft['weeks'],  multi_ft['mae'],  'D-', color=COLORS['multitransfer'], label='Multi-Source FT (N=5)',   lw=2, ms=6)
ax.semilogy(frozen_e['weeks'],  frozen_e['mae'],  '^-', color=COLORS['frozen'],        label='Frozen Backbone (single-src)',lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh, log scale)')
ax.set_title('Multi-Source Transfer vs Alternatives: Eagle/Brooke\n'
             '(Log scale; both FT approaches fail at low data; Frozen Backbone best)')
ax.legend(); ax.grid(True, alpha=0.3, which='both')
savefig('fig10_multi_transfer_eagle.png')

# ── FIG 11: Cross-Type Transfer Comparison ────────────────────────────────────
print("Generating fig11 …")
exp_ct    = os.path.join(RESULTS, "cross_type_transfer")
scratch   = load_csv(os.path.join(exp_ct, "data_efficiency_pretransfer.csv"))
same_site = load_csv(os.path.join(exp_ct, "data_efficiency_transfer_samesite.csv"))
same_type = load_csv(os.path.join(exp_ct, "data_efficiency_transfer_sametype.csv"))
cross_t   = load_csv(os.path.join(exp_ct, "data_efficiency_transfer_crosstype.csv"))
frozen_e  = load_csv(os.path.join(RESULTS, "eagle_education", "data_efficiency_frozen.csv"))
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(scratch['weeks'],   scratch['mae'],   'o-', color=COLORS['scratch'],  label='Scratch',             lw=2, ms=6)
ax.semilogy(same_site['weeks'], same_site['mae'], 's-', color='#E91E63',          label='Same-site/Same-type FT (Samantha)', lw=2, ms=6)
ax.semilogy(same_type['weeks'], same_type['mae'], 'D-', color='#FF9800',          label='Cross-site/Same-type FT (Colin)',   lw=2, ms=6)
ax.semilogy(cross_t['weeks'],   cross_t['mae'],   '^-', color='#9C27B0',          label='Cross-site/Cross-type FT (Miriam)', lw=2, ms=6)
ax.semilogy(frozen_e['weeks'],  frozen_e['mae'],  'v-', color=COLORS['frozen'],   label='Frozen Backbone (same-site)',        lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh, log scale)')
ax.set_title('Domain Distance: Cross-Type Transfer to Eagle/Brooke\n'
             '(All Full FT approaches fail at <16 weeks; Frozen Backbone recovers by 2 weeks)')
ax.legend(); ax.grid(True, alpha=0.3, which='both')
savefig('fig11_cross_type_transfer.png')

# ── FIG 12: Multi-Transfer Generalisation (Easy Target) ──────────────────────
print("Generating fig12 …")
exp_gen   = os.path.join(RESULTS, "multitransfer_generalisation")
scratch   = load_csv(os.path.join(exp_gen, "data_efficiency_pretransfer.csv"))
single_ft = load_csv(os.path.join(exp_gen, "data_efficiency_transfer.csv"))
multi_ft  = load_csv(os.path.join(exp_gen, "data_efficiency_multitransfer.csv"))
frozen_r  = load_csv(os.path.join(RESULTS, "rat_education", "data_efficiency_frozen.csv"))
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(scratch['weeks'],   scratch['mae'],   'o-', color=COLORS['scratch'],       label='Scratch',              lw=2, ms=6)
ax.plot(single_ft['weeks'], single_ft['mae'], 's-', color=COLORS['transfer'],      label='Single-Source FT',     lw=2, ms=6)
ax.plot(multi_ft['weeks'],  multi_ft['mae'],  'D-', color=COLORS['multitransfer'], label='Multi-Source FT (N=5)', lw=2, ms=6)
ax.plot(frozen_r['weeks'],  frozen_r['mae'],  '^-', color=COLORS['frozen'],        label='Frozen Backbone',       lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh)')
ax.set_title('Multi-Transfer Generalisation: Easy Target (Rat/Denise)\n'
             'All transfer strategies outperform Scratch at 4–16 weeks')
ax.legend(); ax.grid(True, alpha=0.3)
savefig('fig12_generalisation.png')

# ── FIG 13: Switch Modelling ──────────────────────────────────────────────────
print("Generating fig13 …")
categories = ['Oracle\n(best in hindsight)', 'Auto-Switch', 'Always Scratch', 'Always Transfer']
rmse_vals  = [22.70, 22.72, 22.84, 25.45]
bar_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(categories, rmse_vals, color=bar_colors, alpha=0.85, edgecolor='white', lw=1.5)
for bar, v in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.06,
            f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylim(21.5, 27)
ax.set_ylabel('Mean RMSE (kWh)')
ax.set_title('Auto-Switch Model Selection Results\n'
             'Achieves 99.9% of Oracle Performance; 10.7% better than Always-Transfer')
ax.annotate('', xy=(1.0, 22.72), xytext=(3.0, 25.45),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(2.0, 24.3, '10.7% improvement', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
savefig('fig13_switch_modelling.png')

# ── FIG 14: PRIME Comparison ──────────────────────────────────────────────────
print("Generating fig14 …")
prime_sw  = os.path.join(PRIME_DIR, "Eagle_education_Brooke_sweep")
# data_efficiency_prime.csv has combined columns (prime_mae, pretransfer_mae)
prime_raw = pd.read_csv(os.path.join(prime_sw, "data_efficiency_prime.csv"))
prime_weeks = prime_raw['weeks'].astype(int).tolist()
prime_mae_vals = prime_raw['prime_mae'].tolist()
sc_prime_mae_vals = prime_raw['pretransfer_mae'].tolist()
ensemble  = load_csv(os.path.join(RESULTS, "ensemble_transfer", "data_efficiency_ensembletransfer.csv"))
frozen_e  = load_csv(os.path.join(RESULTS, "eagle_education", "data_efficiency_frozen.csv"))
fig, ax = plt.subplots(figsize=(9, 6))
ax.semilogy(prime_weeks, sc_prime_mae_vals, 'o-', color=COLORS['scratch'],   label='Scratch',                    lw=2, ms=6)
ax.semilogy(prime_weeks, prime_mae_vals,    's-', color=COLORS['prime'],     label='PRIME (Eagle Education only, N=5)', lw=2, ms=6)
ax.semilogy(ensemble['weeks'], ensemble['mae'],   'D-', color=COLORS['ensemble'],  label='Ensemble Transfer (weight avg)',lw=2, ms=6)
ax.semilogy(frozen_e['weeks'], frozen_e['mae'],   '^-', color=COLORS['frozen'],    label='Frozen Backbone',             lw=2, ms=6)
ax.set_xscale('log', base=2)
ax.set_xticks(WEEKS); ax.set_xticklabels([str(w) for w in WEEKS])
ax.set_xlabel('Training Data (weeks)'); ax.set_ylabel('MAE (kWh, log scale)')
ax.set_title('PRIME vs Alternatives: Eagle/Brooke Target\n'
             '(PRIME sources: Eagle Education only — same site, same type)')
ax.legend(); ax.grid(True, alpha=0.3, which='both')
savefig('fig14_prime_comparison.png')

# ── FIG 15: Eagle/Brooke — All Strategies Summary ─────────────────────────────
print("Generating fig15 …")
exp_e   = os.path.join(RESULTS, "eagle_education")
scratch = load_csv(os.path.join(exp_e, "data_efficiency_pretransfer.csv"))
transfer= load_csv(os.path.join(exp_e, "data_efficiency_transfer.csv"))
frozen  = load_csv(os.path.join(exp_e, "data_efficiency_frozen.csv"))
multi_ft= load_csv(os.path.join(RESULTS, "multi_transfer","data_efficiency_multitransfer.csv"))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# MAE log
for src, lab, col, mk in [(scratch,'Scratch',COLORS['scratch'],'o'),
                           (transfer,'Full FT (single-source)',COLORS['transfer'],'s'),
                           (frozen,'Frozen Backbone',COLORS['frozen'],'^'),
                           (multi_ft,'Multi-Source FT (N=5)',COLORS['multitransfer'],'D')]:
    ax1.semilogy(src['weeks'], src['mae'], f'{mk}-', color=col, label=lab, lw=2, ms=6)
ax1.set_xscale('log',base=2); ax1.set_xticks(WEEKS); ax1.set_xticklabels([str(w) for w in WEEKS])
ax1.set_xlabel('Training Data (weeks)'); ax1.set_ylabel('MAE (kWh, log scale)')
ax1.set_title('Eagle/Brooke: MAE Comparison (log scale)'); ax1.legend(); ax1.grid(True,alpha=0.3,which='both')
# R²
for src, lab, col, mk in [(scratch,'Scratch',COLORS['scratch'],'o'),
                           (transfer,'Full FT (single-source)',COLORS['transfer'],'s'),
                           (frozen,'Frozen Backbone',COLORS['frozen'],'^')]:
    ax2.plot(src['weeks'], src['r2'], f'{mk}-', color=col, label=lab, lw=2, ms=6)
ax2.axhline(0, color='black', ls='--', alpha=0.5, label='R²=0 threshold')
ax2.set_xscale('log',base=2); ax2.set_xticks(WEEKS); ax2.set_xticklabels([str(w) for w in WEEKS])
ax2.set_xlabel('Training Data (weeks)'); ax2.set_ylabel('R²')
ax2.set_title('Eagle/Brooke: R² Progression\n(Frozen Backbone achieves positive R² earliest)')
ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim(-3, 1)
fig.suptitle('Eagle/Brooke: Frozen Backbone is the Best Strategy at Low Data', fontweight='bold', y=1.02)
savefig('fig15_eagle_strategies_summary.png')

print("\nAll 15 figures generated in:", FIGURES_DIR)
