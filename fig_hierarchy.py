"""
Figure C: Experience-hierarchy schematic.

Conceptual 2D placement of phenomenological states in the
(alpha_t, |eps^CL_self|) plane. Not a numerical computation:
it is a visual summary of Corollary prop:dissolution and the
experience hierarchy (sec:memoryless-mindfulness).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Cleaner sans-serif font (Helvetica/Arial-style); matplotlib will fall
# back if Helvetica is unavailable. Use mathtext for math symbols.
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         13,
    'axes.titlesize':    15,
    'axes.labelsize':    14,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'mathtext.fontset':  'cm',
})

fig, ax = plt.subplots(figsize=(10, 7))

XMAX, YMAX = 4.6, 4.4
ax.set_xlim(-0.5, XMAX)
ax.set_ylim(-0.5, YMAX)
ax.set_xlabel(r'Self-agency $\alpha_t$  (belief-to-action coupling)')
ax.set_ylabel(r'Self-model complexity $|\varepsilon^{CL}_{\mathrm{self}}|$')
ax.set_title(r'Experience hierarchy in the $(\alpha_t,\; |\varepsilon^{CL}_{\mathrm{self}}|)$ plane')

for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
ax.axhline(0, color='k', lw=0.6)
ax.axvline(0, color='k', lw=0.6)

# Region shading (Level 3 / normal)
ax.add_patch(plt.Rectangle((2.3, 2.3), XMAX - 2.3, YMAX - 2.3,
                           facecolor='lightblue', alpha=0.20, zorder=0))
ax.text(XMAX - 0.15, YMAX - 0.2, 'Level 3\n(full self)',
        fontsize=12, ha='right', va='top', color='navy',
        style='italic', alpha=0.8)

# Phenomenological states: (x, y, label, color, (dx, dy), ha)
# Ego dissolution label goes BELOW the point so the orange arrow does not cross it.
states = [
    (3.5, 3.5,  'Normal',            'navy',        ( 0.22,  0.28), 'left'),
    (1.6, 2.2,  'Mindfulness',       'purple',      ( 0.22,  0.28), 'left'),
    (0.15, 3.2, 'Level 1 / zombie',  'crimson',     ( 0.28,  0.10), 'left'),
    (0.15, 1.0, 'Ego dissolution',   'darkorange',  ( 0.28, -0.32), 'left'),
    (0.05, 0.10, 'Pure awareness',   'darkgreen',   ( 0.28,  0.00), 'left'),
]

for x, y, label, color, (dx, dy), ha in states:
    ax.plot(x, y, 'o', color=color, markersize=13,
            markeredgecolor='black', markeredgewidth=0.8, zorder=3)
    ax.annotate(label, (x, y), xytext=(x + dx, y + dy),
                fontsize=13, color=color, fontweight='bold',
                ha=ha, zorder=4)

# Trajectories. Endpoints offset so they do not collide with point markers.
trajectories = [
    {'from': (3.4, 3.4), 'to': (0.30, 1.05),
     'color': 'darkorange', 'ls': '--', 'label': 'psychedelic contraction'},
    {'from': (3.4, 3.4), 'to': (0.20, 0.18),
     'color': 'darkgreen',  'ls': ':',  'label': 'deep meditation'},
    {'from': (3.4, 3.4), 'to': (0.30, 3.18),
     'color': 'crimson',    'ls': '-.', 'label': 'depersonalisation'},
]
legend_handles = []
for t in trajectories:
    ax.annotate('', xy=t['to'], xytext=t['from'],
                arrowprops=dict(arrowstyle='->', color=t['color'],
                                lw=2.4, ls=t['ls']),
                zorder=2)
    legend_handles.append(
        mpatches.Patch(color=t['color'], label=t['label'])
    )

ax.legend(handles=legend_handles, loc='upper center',
          bbox_to_anchor=(0.5, -0.18), ncol=3,
          frameon=False, fontsize=15, handlelength=4.0,
          handleheight=2.0, handletextpad=0.8, columnspacing=2.5)

# Note about the origin (left-aligned, starting just to the right of the
# (0,0) tick so the leading "At" sits near the origin)
ax.text(0.20, -0.32,
        r'At $(0,0)$ all internal states vanish but $\mathcal{I}^{CL}$ remains a well-defined channel.',
        fontsize=12, style='italic', color='dimgray', ha='left')

# Ticks
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['0', '', '', '', 'high'])
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['0', '1', '', '', 'high'])
ax.grid(alpha=0.12)

plt.tight_layout()
plt.savefig('fig_hierarchy.pdf', bbox_inches='tight')
plt.savefig('fig_hierarchy.png', dpi=160, bbox_inches='tight')
plt.close()
print('Figure C saved.')
