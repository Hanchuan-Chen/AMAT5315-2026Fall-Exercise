The function `estimate_pi(n, seed)` estimates pi by generating `n` random points in the unit square and calculating the fraction that falls within distance 1 of the origin. The `seed` makes the result repeatable. The implementation is considered correct when:

abs(estimate_pi(1_000_000, seed=2026) - math.pi) < 1e-2
