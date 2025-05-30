import pickle
from dataclasses import dataclass
from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from oetils import JaxTqdm

from traces import ppo, tmaze  # pylint: disable=unused-import


class MRP:
    dy: int
    dx: int
    T: jnp.ndarray
    E: jnp.ndarray
    mu: jnp.ndarray
    T_rev: jnp.ndarray

    def __post_init__(self):
        # Stationary distribution
        with jax.default_device(jax.devices('cpu')[0]):
            eigs = jnp.linalg.eig(self.T)  # jnp.linalg.eig is only compatible with CPU.
        eigs = jax.device_put(eigs, device=jax.devices()[0])
        assert abs(eigs[0][0] - 1) < 1e-5
        self.mu = eigs[1][:, jnp.argmax(eigs[0].real)].real
        self.mu /= self.mu.sum()

        # Time-reversed kernel
        self.T_rev = self.T.T * np.outer(self.mu, 1/self.mu)

    def belief(self, *ys):
        """Compute belief state: p(x_t = • | y_t, y_{t-1}, ...)"""
        if not ys:
            return self.mu
        b = self.T @ self.belief(*ys[1:]) * self.E[ys[0]]
        return b / b.sum()

    def emission_rev(self, x, *ys):
        """Compute reverse emission probability: p(y_t, y_{t-1}, ... | x_t)"""
        e = 0
        for y in ys:
            e += jnp.log(self.E[y] @ x)
            x = self.T_rev @ x
        return jnp.exp(e)


@dataclass
class Sutton(MRP):
    """Sutton's random walk MRP with state aggregation."""
    gam: float = 0.99  # Discount factor
    dx: int = 1001     # Number of states
    dy: int = 11       # Number of aggregated states
    eps: float = 0.5   # Stochasticity of state aggregation sensor
    d: int = 100       # Maximum step size in random walk

    def __post_init__(self):
        xs = jnp.arange(self.dx)

        # Transition kernel
        self.T = jnp.zeros((self.dx, self.dx))
        for i in range(-self.d, self.d+1):
            self.T = self.T.at[jnp.maximum(0, xs + i), xs].set(1 / (2*self.d + 1))
        self.T = self.T.at[self.dx // 2, xs].add(1 - self.T.sum(0))

        # Emission kernel
        self.E = jnp.ones((self.dy, self.dx)) * self.eps / self.dy
        self.E = self.E.at[self.g_(xs), xs].add(1 - self.eps)
        super().__post_init__()

    def f(self, x, rng):
        x += jax.random.randint(rng, (), -self.d, self.d + 1)
        r = jax.lax.select((x < 0) | (x >= self.dx), jnp.sign(x), 0)
        x = jax.lax.select((x < 0) | (x >= self.dx), self.dx // 2, x)
        return x, r

    def reset(self, _):
        return self.dx // 2

    def g_(self, x):
        """State aggregation"""
        return jnp.floor(self.dy * x / self.dx).astype(int)

    def g(self, x, rng):
        y = self.g_(x)
        keys = jax.random.split(rng, 2)
        z = jax.random.bernoulli(keys[0], 1 - self.eps)
        return jax.lax.select(z, y, jax.random.randint(keys[1], (), 0, self.dy))

    def ve_ml(self, w, v):
        """Value error of memoryless estimator"""
        @partial(jax.vmap, in_axes=(0, None))
        @partial(jax.vmap, in_axes=(None, 0))
        def e(x, y):
            return self.mu[x] * (self.eps/self.dy + (1 - self.eps) * (y == self.g_(x))) * (v[x] - w[..., y])**2
        return jnp.sum(e(jnp.arange(self.dx), jnp.arange(self.dy)), (0, 1))

    def ev(self, v):
        """Expected value of memoryless estimator: E[v(y) | x]"""
        xs = jnp.arange(self.dx)[:, None]
        ys = jnp.arange(self.dy)
        return ((self.eps / self.dy + (1 - self.eps) * (self.g_(xs) == ys)) * v[ys]).sum(1)

    def opt_ml(self, v):
        """Explicitly construct optimal memoryless value function (plus suboptimal illustrative)"""
        # Construct optimal memoryless value function
        v_mlo = jnp.zeros(self.dy)
        xs = jnp.arange(self.dx)
        for i in range(self.dy):
            p_i = self.mu * (self.eps / self.dy + (1 - self.eps) * (self.g_(xs) == i))
            p_i /= p_i.sum()
            v_mlo = v_mlo.at[i].set(v @ p_i)

        # Construct suboptimal (illustrative) memoryless value function
        xms = ((jnp.arange(self.dy) + 1/2) * self.dx / self.dy).astype(int)[:, None]
        ys = jnp.arange(self.dy)[None, :]
        pyx = self.eps / self.dy + (1 - self.eps) * (self.g_(xms) == ys)
        mask = self.g_(xs[:, None]) == ys
        b = (v[:, None] * mask).sum(0) / mask.sum(0)
        v_mlso = jnp.linalg.solve(pyx, b)
        return v_mlo, v_mlso

    def rtdp_eval_oracle(self, rng, T=2_000_000):
        """RTDP evaluation (fully observed)"""
        @JaxTqdm(T).loop
        def dp_update(t, x):
            x, v, c, p, rng = x
            y = x + jnp.arange(-self.d, self.d + 1)
            r = jnp.mean(y >= self.dx) - jnp.mean(y < 0)
            x_ = jnp.where((y < 0) | (y >= self.dx), self.dx // 2, y)
            c = c.at[x_].add(1)
            v_ = r + self.gam * v[x_].mean()
            p = p.at[t].set(v_ - v[x])
            v = v.at[x].set(v_)
            rng, key = jax.random.split(rng)
            x, _ = self.f(x, key)
            return x, v, c, p, rng

        @jax.jit
        def dp(rng):
            v = jnp.zeros(self.dx)
            c = jnp.zeros(self.dx, dtype=int)
            p = jnp.zeros(T)
            rng, key = jax.random.split(rng)
            x = self.reset(key)
            _, v, c, p, _ = jax.lax.fori_loop(0, T, dp_update, (x, v, c, p, rng))
            return v, c, p

        v, c, p_dp = dp(rng)
        mu = c / (T * (2 * self.d + 1))
        print(f"RTDP evaluation (oracle): No update for last {100 * (T - (np.arange(T)[p_dp != 0][-1] + 1)) / T:.1f}% "
            "of iterations.")
        return v, mu

    def rtdp_eval(self, rng, T=200_000):
        """RTDP evaluation (memoryless)"""
        # Precomputations
        xs = jnp.arange(self.dx)
        f_ = xs[:, None] + jnp.arange(-self.d, self.d + 1)
        r = jnp.mean(f_ >= self.dx, 1) - jnp.mean(f_ < 0, 1)
        x_ = jnp.where((f_ < 0) | (f_ >= self.dx), self.dx // 2, f_)

        @JaxTqdm(T).loop
        def dp_update(t, x):
            x, v, p, rng = x
            rng, *keys = jax.random.split(rng, 3)
            y = self.g(x, keys[0])
            px = self.mu * (self.eps / self.dy + (1 - self.eps) * (self.g_(xs) == y))
            px /= px.sum()
            v_ = px @ (r + self.gam *
                ((1 - self.eps) * v[self.g_(x_)] + self.eps/self.dy * sum(v[y_] for y_ in range(self.dy))).mean(1))
            p = p.at[t].set(v_ - v[y])
            v = v.at[y].set(v_)
            x, _ = self.f(x, keys[1])
            return x, v, p, rng

        @jax.jit
        def dp(rng):
            v = jnp.zeros(self.dy)
            p = jnp.zeros(T)
            rng, key = jax.random.split(rng)
            x = self.reset(key)
            _, v, p, _ = jax.lax.fori_loop(0, T, dp_update, (x, v, p, rng))
            return v, p

        v_sa_dp, p_sa_dp = dp(rng)
        print("RTDP evaluation (memoryless): No update for last "
            f"{100 * (T - (np.arange(T)[p_sa_dp != 0][-1] + 1)) / T:.1f}% of iterations.")
        return v_sa_dp


class Memory:
    pass


@dataclass
class Window(Memory):
    mrp: MRP
    m: int = 2
    compact: bool = True  # Compact is faster but not compatible with eligibility traces
    concat: bool = False  # Alternative window definition (simpy concatenate observations)

    def __post_init__(self):
        if self.compact and self.concat:
            raise ValueError("The `compact` and `concat` representations cannot be used together.")

    def oracle(self, v, *ys):
        """Compute optimal value estimate"""
        return v @ self.mrp.belief(*ys)

    def opt_w(self, v):
        f = lambda *y: self.oracle(v, *y[::-1])
        for i in range(self.m + 1):
            in_axes = [None] * (self.m + 1)
            in_axes[i] = 0
            f = jax.vmap(f, in_axes=in_axes)
        return f(*[jnp.arange(self.mrp.dy)] * (self.m + 1))

    def opt_w_scan(self, v):
        T = self.mrp.dy ** (self.m + 1)

        @jax.jit
        @JaxTqdm(T).loop
        def f(i, w):
            # Compute next y sequence
            y = []
            for _ in range(self.m + 1):
                y.insert(0, i % self.mrp.dy)
                i //= self.mrp.dy

            # Compute value of y sequence
            w = w.at[*y].set(self.oracle(v, *y))
            return w

        return jax.lax.fori_loop(0, T, f, jnp.zeros((self.m + 1) * (self.mrp.dy,)))

    def ev(self, w, x):
        """Expected value of estimator: E[v(y_t, y_{t-1}, ...) | x_t]"""
        ys = jnp.array(list(product(*((self.m + 1) * [np.arange(self.mrp.dy)]))))
        return jax.jit(jax.vmap(lambda y: self.mrp.emission_rev(x, *y) * self(w, self.enc(*y))))(ys).sum()

    def ve(self, w, v):
        ys = jnp.array(list(product(*((self.m + 1) * [np.arange(self.mrp.dy)]))))
        return jax.jit(jax.vmap(lambda y: self.mrp.mu @ jax.vmap(lambda x:
            self.mrp.emission_rev(x, *y) * (self(w, self.enc(*y)) - v @ x)**2)(jnp.eye(self.mrp.dx))))(ys).sum()

    def ve_scan(self, w, v):
        T = self.mrp.dy ** (self.m + 1)

        @jax.jit
        @JaxTqdm(T).loop
        def f(i, x):
            tot, c = x

            # Compute next y sequence
            y = []
            for _ in range(self.m + 1):
                y.insert(0, i % self.mrp.dy)
                i //= self.mrp.dy

            # Compute value error term
            x = self.mrp.mu @ jax.vmap(
                lambda x: self.mrp.emission_rev(x, *y) * (self(w, self.enc(*y)) - v @ x)**2)(jnp.eye(self.mrp.dx))

            # Kahan summation
            y = x - c
            t = tot + y
            c = (t - tot) - y
            return t, c

        return jax.lax.fori_loop(0, T, f, (0, 0))[0]

    def init(self, y):
        if self.compact:
            phi = (self.m + 1) * (y,)
            w = jnp.zeros((self.m + 1) * (self.mrp.dy,))
            return phi, w
        if self.concat:
            phi = jnp.zeros((self.m + 1, self.mrp.dy), bool).at[:, y].set(True)
            w = jnp.zeros((self.m + 1, self.mrp.dy))
            return phi, w
        phi = jnp.zeros((self.m + 1) * (self.mrp.dy,), bool).at[(self.m + 1) * (y,)].set(True).ravel()
        w = jnp.zeros((self.m + 1) * (self.mrp.dy,)).ravel()
        return phi, w

    def update(self, phi, y):
        if self.compact:
            return phi[1:] + (y,)
        if self.concat:
            return phi.at[:-1, :].set(phi[1:, :]).at[-1, :].set(False).at[-1, y].set(True)
        cur = jnp.unravel_index(jnp.argmax(phi), (self.m + 1, self.mrp.dy))
        new = jnp.ravel_multi_index(cur[1:] + (y,), (self.m + 1, self.mrp.dy), 'clip')
        return jnp.zeros_like(phi).at[new].set(True)

    def update_w(self, w, phi, delta):
        if self.compact:
            return w.at[phi].add(delta)
        return w + delta * phi

    def enc(self, *y):
        if self.compact:
            return y
        if self.concat:
            return jnp.zeros((self.m + 1, self.mrp.dy), bool).at[jnp.arange(len(y)) + self.m + 1 - len(y), y].set(True)
        raise NotImplementedError  # return jnp.zeros((self.m + 1) * (self.mrp.dy,), bool).at[*y].set(True)

    def __call__(self, w, phi):
        if self.compact:
            return w[phi]
        if self.concat:
            return (w * phi).sum()
        return w @ phi


@dataclass
class Trace:
    mrp: MRP
    lam: int = 0.9

    def __post_init__(self):
        T, E, mu, T_rev, lam, I = self.mrp.T, self.mrp.E, self.mrp.mu, self.mrp.T_rev, self.lam, jnp.eye(self.mrp.dx)
        H = jnp.linalg.solve(I - lam * T, T @ jnp.diag(mu))

        self.w2v = (1 - lam) * E @ jnp.linalg.inv(I - lam * T_rev)  # Weight to value
        self.epp = (1 - lam) / (1 + lam) * (jnp.diag(E @ mu) + lam * E @ (H + H.T) @ E.T)  # E[phi @ phi.T]

    def oracle(self, v):
        """Compute optimal weight vector"""
        return jnp.linalg.solve(self.epp, self.w2v @ jnp.diag(self.mrp.mu) @ v)

    def ve(self, w, v):
        return w @ self.epp @ w - 2 * w @ self.w2v @ jnp.diag(self.mrp.mu) @ v + v @ jnp.diag(self.mrp.mu) @ v

    def ev(self, w):
        return w @ self.w2v

    def init(self, y):
        phi = jnp.zeros(self.mrp.dy).at[y].set(1)
        w = jnp.zeros(self.mrp.dy)
        return phi, w

    def update(self, phi, y):
        return (self.lam * phi).at[y].add(1 - self.lam)

    def update_w(self, w, phi, delta):
        return w + delta * phi

    def __call__(self, w, phi):
        return w @ phi


def td0_ml(mrp, rng, N=100, T=100_000, alpha=0.01, dt=1):
    @JaxTqdm(T).loop
    def td_update(t, x):
        x, y, v, vs, rng = x
        rng, *keys = jax.random.split(rng, 3)
        x, r = mrp.f(x, keys[0])
        y_ = mrp.g(x, keys[1])
        v = v.at[y].add(alpha * (r + mrp.gam * v[y_] - v[y]))
        vs = vs.at[(dt / T * t).astype(int)].set(v)
        return x, y_, v, vs, rng

    @jax.jit
    @jax.vmap
    def td(rng):
        v = jnp.zeros(mrp.dy)
        vs = jnp.zeros((dt, mrp.dy))
        rng, *keys = jax.random.split(rng, 3)
        x = mrp.reset(keys[0])
        y = mrp.g(x, keys[1])
        *_, vs, _ = jax.lax.fori_loop(0, T, td_update, (x, y, v, vs, rng))
        return vs

    rng = jax.random.split(rng, N)
    return td(rng)


def td_ml(mrp, rng, lam=0.99, N=100, T=100_000, alpha=0.01, dt=1):
    @JaxTqdm(T).loop
    def td_update(t, x):
        x, y, z, v, vs, rng = x
        rng, *keys = jax.random.split(rng, 3)
        x, r = mrp.f(x, keys[0])
        y_ = mrp.g(x, keys[1])
        v += alpha * (r + mrp.gam * v[y_] - v[y]) * z
        vs = vs.at[(dt / T * t).astype(int)].set(v)
        z = (lam * z).at[y_].add(1 - lam)
        return x, y_, z, v, vs, rng

    @jax.jit
    @jax.vmap
    def td(rng):
        v = jnp.zeros(mrp.dy)
        vs = jnp.zeros((dt, mrp.dy))
        rng, *keys = jax.random.split(rng, 3)
        x = mrp.reset(keys[0])
        y = mrp.g(x, keys[1])
        z = jnp.zeros(mrp.dy).at[y].set(1)
        *_, vs, rng = jax.lax.fori_loop(0, T, td_update, (x, y, z, v, vs, rng))
        return vs

    rng = jax.random.split(rng, N)
    return td(rng)


def td0(mrp, mem, rng, N=100, T=100_000, alpha=0.01, dt=1):
    @JaxTqdm(T).loop
    def td_update(t, x):
        x, phi, w, ws, rng = x
        rng, *keys = jax.random.split(rng, 3)
        x, r = mrp.f(x, keys[0])
        y = mrp.g(x, keys[1])
        phi_ = mem.update(phi, y)
        w = mem.update_w(w, phi, alpha * (r + mrp.gam * mem(w, phi_) - mem(w, phi)))
        ws = ws.at[(dt / T * t).astype(int)].set(w)
        return x, phi_, w, ws, rng

    @jax.jit
    @jax.vmap
    def td(rng):
        rng, *keys = jax.random.split(rng, 3)
        x = mrp.reset(keys[0])
        y = mrp.g(x, keys[1])
        phi, w = mem.init(y)
        ws = jnp.zeros((dt, *w.shape))
        *_, ws, rng = jax.lax.fori_loop(0, T, td_update, (x, phi, w, ws, rng))
        return ws

    rng = jax.random.split(rng, N)
    return td(rng)


def w2v(w, mrp, mem, rng, N=100, T=100_000):
    @JaxTqdm(T).loop
    def value_update(_, x):
        x, phi, v, c, rng = x
        c = c.at[x].add(1)
        v = v.at[x].add(mem(w, phi))
        rng, *keys = jax.random.split(rng, 3)
        x, _ = mrp.f(x, keys[0])
        y = mrp.g(x, keys[1])
        phi = mem.update(phi, y)
        return x, phi, v, c, rng

    @jax.jit
    @jax.vmap
    def value(rng):
        v = jnp.zeros(mrp.dx)
        c = jnp.zeros(mrp.dx, dtype=int)
        rng, *keys = jax.random.split(rng, 3)
        x = mrp.reset(keys[0])
        y = mrp.g(x, keys[1])
        phi, _ = mem.init(y)
        *_, v, c, _ = jax.lax.fori_loop(0, T, value_update, (x, phi, v, c, rng))
        v /= c
        return v

    rng = jax.random.split(rng, N)
    return value(rng)


def mcve(w, v, mrp, mem, rng, N=100, T=100_000):
    """Monte Carlo estimation of value error"""
    @JaxTqdm(T).loop
    def ve_update(_, x):
        x, phi, s, c, rng = x
        c += 1
        s += (v[x] - mem(w, phi))**2
        rng, *keys = jax.random.split(rng, 3)
        x, _ = mrp.f(x, keys[0])
        y = mrp.g(x, keys[1])
        phi = mem.update(phi, y)
        return x, phi, s, c, rng

    @jax.jit
    @jax.vmap
    def ve(rng):
        s = 0
        c = 0
        rng, *keys = jax.random.split(rng, 3)
        x = mrp.reset(keys[0])
        y = mrp.g(x, keys[1])
        phi, _ = mem.init(y)
        *_, s, c, _ = jax.lax.fori_loop(0, T, ve_update, (x, phi, s, c, rng))
        return s / c

    rng = jax.random.split(rng, N)
    return ve(rng)


def ve_fwd(w, v, mrp, mem, rng, N):
    """Monte Carlo estimation of value error"""
    @jax.jit
    @jax.vmap
    def ve(rng):
        rng, key = jax.random.split(rng)
        x = jax.random.choice(key, mrp.dx, p=mrp.mu)
        ys = jnp.zeros((mem.m + 1, mrp.dy), bool)
        for i in range(mem.m + 1):
            rng, *keys = jax.random.split(rng, 3)
            ys = ys.at[i, mrp.g(x, keys[0])].set(True)
            x, _ = mrp.f(x, keys[1])
        return (v[x] - mem(w, ys))**2

    rng = jax.random.split(rng, N)
    return ve(rng).sum() / N


def sutton(seed, sutton_params, td_params, trace_params, window_params, path):
    rng = jax.random.key(seed)
    N, T, lam = td_params.N, td_params.T, td_params.lam
    del td_params['N'], td_params['T'], td_params['lam']

    mrp = Sutton(**sutton_params)
    v, mu = mrp.rtdp_eval_oracle(rng, T=20*T)
    v_mlo, v_mlso = mrp.opt_ml(v)
    v_sa_dp = mrp.rtdp_eval(rng, T=2*T)
    v_sa = td0_ml(mrp, rng, N=N, T=T, **td_params)[:, -1]
    v_et = td_ml(mrp, rng, lam=lam, N=10*N, T=T, **td_params)[:, -1]

    window = Window(mrp, **window_params)
    rngs = jax.random.split(rng)
    w_fh_opt = window.opt_w(v)
    w_fh = td0(mrp, window, rngs[0], N=N, T=10*T, **td_params)[:, -1]

    trace = Trace(mrp, **trace_params)
    rngs = jax.random.split(rng)
    w_ot_opt = trace.oracle(v)
    w_ot = td0(mrp, trace, rngs[0], N=N, T=T, **td_params)[:, -1]

    print("Optimal memoryless value error:    ", mrp.ve_ml(v_mlo, v))
    print("Suboptimal memoryless value error: ", mrp.ve_ml(v_mlso, v))
    print("Value error memoryless DP:         ", mrp.ve_ml(v_sa_dp, v))
    print("Value error memoryless TD(0):      ", mrp.ve_ml(v_sa.mean(0), v))
    print("Value error memoryless TD(λ):      ", mrp.ve_ml(v_et.mean(0), v))
    print("Optimal fixed memory value error:  ", window.ve(w_fh_opt, v))
    print("Value error fixed memory TD(0):    ", window.ve(w_fh.mean(0), v))
    print("Optimal trace memory value error:  ", trace.ve(w_ot_opt, v))
    print("Value error trace memory TD(0):    ", trace.ve(w_ot.mean(0), v))
    jnp.savez(path / 'results.npz', v=v, mu=mu, v_mlo=v_mlo, v_mlso=v_mlso, v_sa_dp=v_sa_dp, v_sa=v_sa, v_et=v_et,
        w_fh_opt=w_fh_opt, w_fh=w_fh, w_ot_opt=w_ot_opt, w_ot=w_ot)


def sutton_window_opt(seed, sutton_params, window_params, path):
    mrp = Sutton(**sutton_params)
    v, _ = mrp.rtdp_eval_oracle(jax.random.key(seed))
    mem = Window(mrp, **window_params)
    w = mem.opt_w_scan(v)
    jnp.save(path / 'w.npy', w)
    ve = mem.ve_scan(w, v)
    jnp.save(path / 've.npy', ve)
    print(f"Optimal window (m = {window_params.m}) value error: {ve}")


def gpu_test(seed, N, T, path):
    @JaxTqdm(T).loop
    def f(_, x):
        x, rng = x
        rng, key = jax.random.split(rng)
        x = x + jax.random.uniform(key, (N, N))
        x = jnp.linalg.inv(x)
        return x, rng

    @jax.jit
    def loop(rng):
        x = jnp.zeros((N, N))
        x, rng = jax.lax.fori_loop(0, T, f, (x, rng))
        return x

    import time  # pylint: disable=import-outside-toplevel
    t0 = time.time()
    rng = jax.random.key(seed)
    x = loop(rng)
    print(f"Final matrix: min = {x.min()}, max = {x.max()}, mean = {x.mean()}")
    print(f"Elapsed time: {time.time() - t0} s")
    jnp.save(path / 'x.npy', x)


def sutton_td(seed, sutton_params, td_params, params, path):
    rng = jax.random.key(seed)
    mrp = Sutton(**sutton_params)
    rng, key = jax.random.split(rng)
    v, _ = mrp.rtdp_eval_oracle(key)

    mem = None
    if params.memory == 'trace':
        mem = Trace(mrp, **params.trace_params)
    if params.memory == 'window':
        mem = Window(mrp, **params.window_params)

    if mem:
        assert td_params.lam == 0
        del td_params['lam']
        ws = td0(mrp, mem, rng, **td_params)
        jnp.save(path / 'w.npy', ws[:, -1])
        jnp.save(path / 'ws.npy', ws)
        if params.get('mcve'):
            ves = jax.jit(jax.vmap(jax.vmap(lambda w: ve_fwd(w, v, mrp, mem, rng, **params.mcve_params).mean())))(ws)
        elif params.get('jax_scan'):
            ves = jax.jit(jax.vmap(jax.vmap(lambda w: mem.ve_scan(w, v))))(ws)
        else:
            ves = jax.jit(jax.vmap(jax.vmap(lambda w: mem.ve(w, v))))(ws)
        jnp.save(path / 'ves.npy', ves)
    elif td_params.lam == 0:
        del td_params['lam']
        ws = td0_ml(mrp, rng, **td_params)
        jnp.save(path / 'w.npy', ws[:, -1])
        jnp.save(path / 'ws.npy', ws)
        ves = jax.jit(jax.vmap(jax.vmap(lambda w: mrp.ve_ml(w, v))))(ws)
        jnp.save(path / 'ves.npy', ves)
    else:
        ws = td_ml(mrp, rng, **td_params)
        jnp.save(path / 'w.npy', ws[:, -1])
        jnp.save(path / 'ws.npy', ws)
        ves = jax.jit(jax.vmap(jax.vmap(lambda w: mrp.ve_ml(w, v))))(ws)
        jnp.save(path / 'ves.npy', ves)


def ppo_minigrid(seed, env, env_params, ppo_params, memory, wab, params, path):
    if wab.log:
        wandb.login(key=wab.key)
        wandb.init(project='trace', group=params.name, config=params, dir=str(path))
    rng = jax.random.key(seed)
    env, env_params, encoding = ppo.make_minigrid(env, **env_params)
    agent = ppo.ActorCritic(encoding.num_actions)
    if memory == 'trace':
        mem = ppo.Trace(encoding, **params.trace_params)
    elif memory == 'window':
        mem = ppo.Window(encoding, **params.window_params)
    elif memory is None:  # Memoryless
        mem = ppo.Window(encoding, 0)
    model, metrics = ppo.ppo(env, env_params, agent, mem, **ppo_params, rng=rng, wblog=wab.log, wblog_all=wab.log_all)
    with open(path / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(path / 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    if wab.log:
        wandb.finish()
