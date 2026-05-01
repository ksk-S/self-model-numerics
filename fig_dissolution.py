"""
Figure B: Numerical visualisation of Proposition prop:dissolution.

Two independent axes:
  - self memory strength lambda in [0, 1]
      lambda=1: memoryful policy, lambda=0: reactive baseline
  - world informativeness gamma in [0, 1]
      gamma=1: deterministic observation, gamma=0: uniform observation

Plots:
  (a) alpha_t heatmap over (lambda, gamma), with corners labelled
      by the Dissolution-Limits phenomenology
  (b) alpha_t vs lambda at fixed gamma — "ego dissolution" axis (i)
  (c) I(M; Y) vs gamma — "world-collapse" axis (ii); combined
      (0,0) corner is pure awareness (iii)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
# Output directory: <repo>/figures/, written next to this script.
fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(fig_dir, exist_ok=True)


plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':         13,
    'axes.titlesize':    14,
    'axes.labelsize':    13,
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   12,
    'mathtext.fontset':  'cm',
})

np.random.seed(0)

n_mu, n_y, n_a = 3, 2, 2
p_m = np.ones(n_mu) / n_mu

# Fully memoryful policy (depends on both M and Y)
pi_full = np.array([
    [[1.0, 0.0], [0.0, 1.0]],   # M=0
    [[0.0, 1.0], [1.0, 0.0]],   # M=1
    [[0.5, 0.5], [0.5, 0.5]],   # M=2
])


def obs_model(gamma):
    """gamma=1 gives a deterministic Y=f(M); gamma=0 gives uniform Y."""
    base = np.array([[1.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 0.0]])
    uniform = np.full((n_mu, n_y), 1.0 / n_y)
    return gamma * base + (1 - gamma) * uniform


def mixed_policy(lam):
    """pi_lambda = lam * pi_full + (1 - lam) * pi_reflex, where
       pi_reflex(a|y) = sum_m p(m) pi_full(a|m,y) (m-independent)."""
    pi_reflex_ay = np.einsum('m,mya->ya', p_m, pi_full)
    pi_reflex = np.broadcast_to(pi_reflex_ay[None, :, :], pi_full.shape).copy()
    return lam * pi_full + (1 - lam) * pi_reflex


def alpha_from_policy(pi, p_y_given_m):
    p_y = np.einsum('m,my->y', p_m, p_y_given_m)
    p_m_given_y = np.einsum('m,my->my', p_m, p_y_given_m) / p_y[None, :]
    p_a_given_y = np.einsum('my,mya->ya', p_m_given_y, pi)
    tot = 0.0
    for yi in range(n_y):
        for mi in range(n_mu):
            for ai in range(n_a):
                p = pi[mi, yi, ai]
                pb = p_a_given_y[yi, ai]
                if p > 1e-15 and pb > 1e-15:
                    tot += p_y[yi] * p_m_given_y[mi, yi] * p * np.log(p / pb)
    return tot


def world_info(p_y_given_m):
    p_y = np.einsum('m,my->y', p_m, p_y_given_m)
    tot = 0.0
    for mi in range(n_mu):
        for yi in range(n_y):
            p = p_y_given_m[mi, yi]
            if p > 1e-15 and p_y[yi] > 1e-15:
                tot += p_m[mi] * p * np.log(p / p_y[yi])
    return tot


lams = np.linspace(0, 1, 41)
gammas = np.linspace(0, 1, 41)

alpha_grid = np.zeros((len(gammas), len(lams)))
for gi, g in enumerate(gammas):
    pym = obs_model(g)
    for li, lam in enumerate(lams):
        alpha_grid[gi, li] = alpha_from_policy(mixed_policy(lam), pym)

world_info_arr = np.array([world_info(obs_model(g)) for g in gammas])

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

im = axes[0].imshow(alpha_grid, origin='lower', aspect='auto',
                    extent=[lams[0], lams[-1], gammas[0], gammas[-1]],
                    cmap='viridis')
axes[0].set_xlabel(r'Self memory $\lambda$')
axes[0].set_ylabel(r'World informativeness $\gamma$')
axes[0].set_title(r'(a) $\alpha_t$ over $(\lambda,\gamma)$')
cbar = plt.colorbar(im, ax=axes[0])
cbar.set_label(r'$\alpha_t$ (nats)')
axes[0].plot([0], [0], 'wo', ms=10, mec='black', mew=0.8)
axes[0].annotate('pure awareness (iii)', xy=(0.02, 0.02), xytext=(0.10, 0.13),
                 color='white', fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='white', lw=1.0))
axes[0].plot([0], [1], 'wo', ms=10, mec='black', mew=0.8)
axes[0].annotate('ego dissolution (i)', xy=(0.02, 0.98), xytext=(0.10, 0.78),
                 color='white', fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='white', lw=1.0))
axes[0].plot([1], [1], 'wo', ms=10, mec='black', mew=0.8)
axes[0].annotate('normal', xy=(0.98, 0.98), xytext=(0.62, 0.82),
                 color='white', fontsize=11, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='white', lw=1.0))

axes[1].plot(lams, alpha_grid[-1, :], 'b-',  lw=2.4, label=r'$\gamma = 1$ (transparent)')
axes[1].plot(lams, alpha_grid[len(gammas) // 2, :], 'g--', lw=1.9,
             label=r'$\gamma = 0.5$')
axes[1].plot(lams, alpha_grid[0, :],  'r:', lw=1.9, label=r'$\gamma = 0$ (uniform $Y$)')
axes[1].set_xlabel(r'Self memory $\lambda$')
axes[1].set_ylabel(r'$\alpha_t$ (nats)')
axes[1].set_title(r'(b) Self-collapse limit (i)')
axes[1].legend(loc='upper left')
axes[1].grid(alpha=0.25)
for s in ('top', 'right'):
    axes[1].spines[s].set_visible(False)

axes[2].plot(gammas, world_info_arr, 'k-', lw=2.4)
axes[2].set_xlabel(r'World informativeness $\gamma$')
axes[2].set_ylabel(r'$I(M; Y)$ (nats)')
axes[2].set_title(r'(c) World-collapse limit (ii)')
axes[2].grid(alpha=0.25)
axes[2].axvspan(0, 0.02, color='red', alpha=0.18)
axes[2].text(0.03, world_info_arr.max() * 0.5, 'collapse',
             fontsize=11, fontweight='bold', color='darkred', ha='left')
for s in ('top', 'right'):
    axes[2].spines[s].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'fig_dissolution.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(fig_dir, 'fig_dissolution.png'), dpi=160, bbox_inches='tight')
plt.close()
print(f'alpha range: [{alpha_grid.min():.4f}, {alpha_grid.max():.4f}]')
print(f'I(M;Y) range: [{world_info_arr.min():.4f}, {world_info_arr.max():.4f}]')
print('Figure B saved.')
