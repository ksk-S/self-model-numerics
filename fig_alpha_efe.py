"""
Figure A: Numerical verification of Proposition prop:alpha-efe.

Verifies the identity
    alpha_t = I(M^self; A | Y) = E_Y[BS(Y)]
across a sweep of Boltzmann policy temperatures beta.

BS(y) = E_mu[ KL(pi(.|mu,y) || pi_bar(.|y)) ]
pi_bar(a|y) = sum_mu p(mu) pi(a|mu, y)
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

# Setup: 3 beliefs, 2 observations (indep.), 2 actions
n_mu, n_y, n_a = 3, 2, 2
mu_prior = np.ones(n_mu) / n_mu
y_prior = np.ones(n_y) / n_y

# EFE G(mu, y, a) — belief-dependent optimal actions
G = np.zeros((n_mu, n_y, n_a))
G[0, :, :] = [[0.0, 1.5], [0.0, 1.5]]   # mu=1: prefers a=0
G[1, :, :] = [[1.5, 0.0], [1.5, 0.0]]   # mu=2: prefers a=1
G[2, :, :] = [[0.6, 0.4], [0.4, 0.6]]   # mu=3: y-dependent mild preference


def boltzmann_policy(G, beta):
    logits = -beta * G
    logits -= logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=-1, keepdims=True)


def alpha_t(pi):
    """alpha_t = I(mu; A | Y) directly."""
    pi_bar = np.einsum('m,mya->ya', mu_prior, pi)  # (n_y, n_a)
    tot = 0.0
    for yi in range(n_y):
        for mi in range(n_mu):
            for ai in range(n_a):
                p = pi[mi, yi, ai]
                pb = pi_bar[yi, ai]
                if p > 1e-15 and pb > 1e-15:
                    tot += y_prior[yi] * mu_prior[mi] * p * np.log(p / pb)
    return tot


def belief_sensitivity_expected(pi):
    """E_Y[BS(Y)] where BS(y) = E_mu[KL(pi(.|mu,y) || pi_bar(.|y))]."""
    pi_bar = np.einsum('m,mya->ya', mu_prior, pi)
    tot = 0.0
    for yi in range(n_y):
        bs_y = 0.0
        for mi in range(n_mu):
            kl = 0.0
            for ai in range(n_a):
                p = pi[mi, yi, ai]
                pb = pi_bar[yi, ai]
                if p > 1e-15 and pb > 1e-15:
                    kl += p * np.log(p / pb)
            bs_y += mu_prior[mi] * kl
        tot += y_prior[yi] * bs_y
    return tot


def G_variance(pi):
    """Per-y variance of G(a*(mu); mu, y) across mu (greedy proxy)."""
    var_total = 0.0
    for yi in range(n_y):
        vals = np.array([G[mi, yi, :].min() for mi in range(n_mu)])  # optimal G per mu
        var_total += y_prior[yi] * np.var(vals)
    return var_total


betas = np.linspace(0, 12, 241)
alphas = np.array([alpha_t(boltzmann_policy(G, b)) for b in betas])
bss = np.array([belief_sensitivity_expected(boltzmann_policy(G, b)) for b in betas])

err = np.abs(alphas - bss).max()
print(f'Max |alpha_t - E_Y[BS(Y)]| over sweep = {err:.3e}')

# Entropy upper bound
H_mu = -np.sum(mu_prior * np.log(mu_prior + 1e-15))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

ax[0].plot(betas, alphas, 'b-', lw=2.6, label=r'$\alpha_t = I(M^{\mathrm{self}}; A\mid Y)$')
ax[0].plot(betas, bss, 'r--', lw=2.2, label=r'$E_Y[\mathrm{BS}(Y)]$')
ax[0].set_xlabel(r'Inverse temperature $\beta$')
ax[0].set_ylabel('Information (nats)')
ax[0].set_title(r'(a) Identity $\alpha_t = E_Y[\mathrm{BS}(Y)]$')
ax[0].legend(loc='lower right')
ax[0].grid(alpha=0.25)
for s in ('top', 'right'):
    ax[0].spines[s].set_visible(False)

ax[1].plot(betas, alphas, 'b-', lw=2.6, label=r'$\alpha_t$')
ax[1].axhline(H_mu, color='gray', ls=':', lw=1.8,
              label=r'$H(M^{\mathrm{self}}) = \log 3$')
ax[1].set_xlabel(r'Inverse temperature $\beta$')
ax[1].set_ylabel('Information (nats)')
ax[1].set_title(r'(b) Greedy-limit saturation')
ax[1].legend(loc='lower right')
ax[1].grid(alpha=0.25)
for s in ('top', 'right'):
    ax[1].spines[s].set_visible(False)

plt.tight_layout()
out_pdf = os.path.join(fig_dir, 'fig_alpha_efe.pdf')
out_png = os.path.join(fig_dir, 'fig_alpha_efe.png')
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, dpi=160, bbox_inches='tight')
plt.close()
print(f'Saved {out_pdf} and {out_png}')
