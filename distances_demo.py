"""
Демонстрация к вопросам 22–24:
  1) DTW: расстояние между временными рядами разной длины;
  2) Махаланобис: при диагональной Σ — нормированное евклидово, при Σ = I — евклидово;
  3) Линейная регрессия и KNN: экстраполяция и нелинейная зависимость.

Запуск:  python distances_demo.py
Зависимости: numpy, matplotlib, scikit-learn
Результат: числа печатаются в консоль, графики сохраняются в папку figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

RNG = np.random.default_rng(42)
OUT = "figures"
os.makedirs(OUT, exist_ok=True)

ACCENT = "#2B6CB0"   # основной цвет
SECOND = "#C05621"   # второй цвет
GREY = "#718096"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": "#1A202C",
    "xtick.color": GREY, "ytick.color": GREY,
})


# ---------------------------------------------------------------------------
# 1. DTW
# ---------------------------------------------------------------------------
def dtw(a, b, window=None):
    """Расстояние DTW между рядами a (длина n) и b (длина m).

    window — ширина полосы Сакоэ–Чибы (None = без ограничения).
    Возвращает (сырое расстояние, нормированное на длину пути, путь).
    Сложность O(n·m), с полосой — O(n·w).
    """
    n, m = len(a), len(b)
    if window is not None:
        window = max(window, abs(n - m))  # иначе путь из начала в конец не построить
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = 1, m
        if window is not None:
            lo, hi = max(1, i - window), min(m, i + window)
        for j in range(lo, hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # обратный проход: восстанавливаем путь выравнивания
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return D[n, m], D[n, m] / len(path), path


def znorm(x):
    return (x - x.mean()) / x.std()


def interp_to(x, length):
    """Равномерное растяжение ряда до нужной длины (линейная интерполяция)."""
    return np.interp(np.linspace(0, 1, length), np.linspace(0, 1, len(x)), x)


print("=" * 60)
print("1. DTW")
print("=" * 60)

# Пример из теории: разные ряды, а DTW = 0
raw, _, _ = dtw(np.array([1, 2, 3.]), np.array([1, 1, 2, 2, 3.]))
print(f"DTW((1,2,3), (1,1,2,2,3)) = {raw:.2f}  -> DTW не метрика")

# Ряд A: импульс в начале (40 точек); ряд B: тот же импульс позже и медленнее (70 точек)
t_a, t_b = np.linspace(0, 1, 40), np.linspace(0, 1, 70)
A = znorm(np.exp(-((t_a - 0.3) / 0.08) ** 2) + 0.05 * RNG.standard_normal(40))
B = znorm(np.exp(-((t_b - 0.6) / 0.12) ** 2) + 0.05 * RNG.standard_normal(70))
# Ряд C: другая форма (два горба), 55 точек
t_c = np.linspace(0, 1, 55)
C = znorm(np.sin(4 * np.pi * t_c) + 0.05 * RNG.standard_normal(55))

for name, other in [("B (та же форма, сдвиг)", B), ("C (другая форма)", C)]:
    eu = np.sqrt(np.mean((interp_to(A, len(other)) - other) ** 2))
    _, norm_dtw, _ = dtw(A, other)
    print(f"A vs {name:24s}: евклид после растяжения = {eu:.2f}, DTW/длина пути = {norm_dtw:.2f}")

_, _, path = dtw(A, B)
fig, ax = plt.subplots(figsize=(8, 3.6))
off = 4.5
ax.plot(t_a, A + off, color=ACCENT, lw=2.2, label="ряд A, n = 40")
ax.plot(t_b, B, color=SECOND, lw=2.2, label="ряд B, m = 70")
for i, j in path[::2]:
    ax.plot([t_a[i], t_b[j]], [A[i] + off, B[j]], color=GREY, lw=0.6, alpha=0.6)
ax.set_yticks([]); ax.set_xlabel("время (доля длины ряда)")
ax.spines["left"].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/dtw.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Махаланобис
# ---------------------------------------------------------------------------
def mahalanobis(x, y, cov):
    d = x - y
    return float(np.sqrt(d @ np.linalg.inv(cov) @ d))


print("\n" + "=" * 60)
print("2. Махаланобис")
print("=" * 60)
x, y = np.array([2.0, 1.0]), np.array([0.0, 0.0])
cases = {
    "коррелированные, ρ=0.8": np.array([[4.0, 1.6], [1.6, 1.0]]),
    "ρ=0, σ²=(4, 1)":         np.diag([4.0, 1.0]),
    "ρ=0, σ²=1 (Σ = I)":      np.eye(2),
}
d_e = np.linalg.norm(x - y)
for name, S in cases.items():
    d_m = mahalanobis(x, y, S)
    d_norm = np.sqrt(np.sum((x - y) ** 2 / np.diag(S)))
    print(f"{name:24s}: d_M = {d_m:.3f}, нормир. евклид = {d_norm:.3f}, евклид = {d_e:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
g = np.linspace(-4, 4, 300)
X, Y = np.meshgrid(g, g)
P = np.stack([X, Y], -1)
for ax, (name, S) in zip(axes, cases.items()):
    inv = np.linalg.inv(S)
    Z = np.sqrt(np.einsum("...i,ij,...j->...", P, inv, P))
    ax.contour(X, Y, Z, levels=[1, 2, 3], colors=[ACCENT], linewidths=1.6)
    pts = RNG.multivariate_normal([0, 0], S, 300)
    ax.scatter(pts[:, 0], pts[:, 1], s=4, color=GREY, alpha=0.5)
    ax.set_title(name, fontsize=11); ax.set_aspect("equal")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(f"{OUT}/mahalanobis.png", dpi=200); plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Линейная регрессия и KNN
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. Линейная регрессия и KNN")
print("=" * 60)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


# (а) Линейная зависимость, прогноз ЗА пределами обучающего диапазона
x_tr = RNG.uniform(0, 10, 100).reshape(-1, 1)
y_tr = 2 * x_tr.ravel() + 1 + RNG.normal(0, 1.5, 100)
x_te = np.linspace(10, 15, 50).reshape(-1, 1)
y_te = 2 * x_te.ravel() + 1
lr = LinearRegression().fit(x_tr, y_tr)
knn = KNeighborsRegressor(n_neighbors=5).fit(x_tr, y_tr)
print(f"(а) экстраполяция x∈[10,15]: MSE LR = {mse(lr.predict(x_te), y_te):.2f}, "
      f"MSE KNN = {mse(knn.predict(x_te), y_te):.2f}")
print(f"    макс. прогноз KNN = {knn.predict(x_te).max():.2f}, макс. y в обучении = {y_tr.max():.2f}")

# (б) Нелинейная зависимость внутри диапазона
x2 = RNG.uniform(0, 10, 300).reshape(-1, 1)
y2 = np.sin(x2.ravel()) + RNG.normal(0, 0.2, 300)
x2_te = np.linspace(0, 10, 200).reshape(-1, 1)
y2_te = np.sin(x2_te.ravel())
lr2 = LinearRegression().fit(x2, y2)
ks = [1, 3, 5, 10, 20, 50, 100]
cv = [-cross_val_score(KNeighborsRegressor(n_neighbors=k), x2, y2, cv=5,
                       scoring="neg_mean_squared_error").mean() for k in ks]
best_k = ks[int(np.argmin(cv))]
knn2 = KNeighborsRegressor(n_neighbors=best_k).fit(x2, y2)
print(f"(б) y = sin(x): MSE LR = {mse(lr2.predict(x2_te), y2_te):.3f}, "
      f"MSE KNN (K={best_k}) = {mse(knn2.predict(x2_te), y2_te):.3f}")
print("    кросс-валидация K: " + ", ".join(f"K={k}: {c:.3f}" for k, c in zip(ks, cv)))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axes[0]
grid = np.linspace(0, 15, 300).reshape(-1, 1)
ax.scatter(x_tr, y_tr, s=8, color=GREY, alpha=0.6)
ax.plot(grid, lr.predict(grid), color=ACCENT, lw=2.2, label="лин. регрессия")
ax.plot(grid, knn.predict(grid), color=SECOND, lw=2.2, label="KNN, K = 5")
ax.axvspan(10, 15, color=GREY, alpha=0.1)
ax.text(12.5, 3, "экстраполяция", ha="center", color=GREY, fontsize=10)
ax.set_title("линейная зависимость", fontsize=11); ax.legend(frameon=False, fontsize=10)
ax = axes[1]
ax.scatter(x2, y2, s=6, color=GREY, alpha=0.5)
ax.plot(x2_te, lr2.predict(x2_te), color=ACCENT, lw=2.2, label="лин. регрессия")
ax.plot(x2_te, knn2.predict(x2_te), color=SECOND, lw=2.2, label=f"KNN, K = {best_k}")
ax.set_title("нелинейная зависимость y = sin x", fontsize=11); ax.legend(frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/lr_knn.png", dpi=200); plt.close(fig)

print("\nГрафики сохранены в папку", OUT)
