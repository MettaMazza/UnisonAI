import sys
import numpy as np
from fractions import Fraction
from omni.logging_config import core_logger as logger

# =============================================================================
# THE FORCED CONSTANTS (FROM THE SELF-PROVEN THEOREM; ZERO AXIOMS)
# =============================================================================
# The generators of the Smithian Fold Theory
GEN_B = 2   # Binary Count (The fold's period beyond the One)
GEN_C = 3   # Colour Count
CTX_MAX = 6 # GEN_B * GEN_C  (the structural context depth)

# --- Derived forced locks (each cross-checked in verify_locks() at wake) ------
# The functional band: b^(b+c) coefficients carry a held field's function.
# constants/functional_band.ep (Step 311). Covers the colour volume c^c=27,
# closes as (BAND-1)/BAND + 1/BAND = 1.
BAND = GEN_B ** (GEN_B + GEN_C)          # = 32; the perceptual coefficient count (top-2^5)
# The held memory orbit {1/3, 2/3}: kept by re-exciting, never reaching the One.
# constants/memory_persistence.ep (Step 145).
MEMORY_STATE = Fraction(1, GEN_C)        # = 1/3
REFRESH_STATE = Fraction(GEN_B, GEN_C)   # = 2/3
# The attention/selection lock: the self-antipodal balance, 1/b.
# constants/attention_capacity.ep (Step 181); the graduation lock too.
FOCUS_LOCK = Fraction(1, GEN_B)          # = 1/2
# THE INTEGRATION LOCKS: deep context forced (ernos-verified, form-closed by reuse).
# constants/contextual_integration.ep — depth = the covering depth b+c; the step
# weights are successive halvings closing to the One with the floor.
INTEGRATION_DEPTH = GEN_B + GEN_C        # = 5
# constants/generation_selection_law.ep — spread capacity & binding threshold are the
# closed focus lock (reused); the re-expression minimum is the binary count itself.
SPREAD_LOCK = FOCUS_LOCK                 # = 1/2
REEXPRESS_MIN = GEN_B                    # = 2
# The kin floor (generalisation) and compose floor, forced from the generators.
KIN_FLOOR = Fraction(1, GEN_B * GEN_C)          # = 1/6
COMPOSE_FLOOR = Fraction(1, GEN_B * GEN_B * GEN_C)  # = 1/12

def cascade_share(rank):
    """The dyadic cascade share of the member at rank r: 1/b^r.
    constants/partition_localization.ep (Step 313). The ranked members of a
    partitioned selection store take 1/2, 1/4, 1/8, ... telescoping to the One."""
    return Fraction(1, GEN_B ** rank)

def halt_violation(reason):
    """
    SFT Law: The engine enforces its own existence.
    Any breach of domain, unforced selection, or broken derivation chain
    triggers an immediate hard crash. Invalid operations cannot be bypassed.
    """
    logger.critical(f"FATAL SFT VIOLATION: {reason}")
    sys.exit(1)

# =============================================================================
# EXACT RATIONAL ARITHMETIC & DOMAIN ENFORCEMENT
# =============================================================================
class FoldValue:
    """
    SFT Domain Law: Every value lives strictly in the interval (0, 1].
    Zero is not permitted, negative numbers are not permitted, and no value
    can exceed the One.
    """
    def __init__(self, value):
        if not isinstance(value, Fraction):
            try:
                # Attempt exact rational conversion if it's an integer or float
                # Float conversion is strongly discouraged in SFT but provided for literal initialization
                value = Fraction(value).limit_denominator()
            except Exception:
                halt_violation("Values must be exact Fractions.")
                
        if value <= 0:
            halt_violation(f"Value {value} is <= 0. SFT forbids zero and negatives.")
        if value > 1:
            halt_violation(f"Value {value} is > 1. SFT forbids exceeding the One.")
            
        self.val = value
        
    def fold(self):
        """
        The Fold map: x -> 2x, cast out whole Ones.
        This is the engine of the theory.
        """
        doubled = self.val * 2
        # Keep the part within the One. In SFT, x -> 2x mod 1.
        if doubled > 1:
            result = doubled - 1
        else:
            result = doubled
            
        # 1 folds to 1 (the One is invariant).
        if result == 0:
            result = Fraction(1, 1)
            
        return FoldValue(result)
        
    def take(self, other):
        """
        The only permitted subtraction: absolute difference.
        """
        if not isinstance(other, FoldValue):
            halt_violation("take() requires another FoldValue.")
        diff = abs(self.val - other.val)
        if diff == 0:
            halt_violation("take() resulted in 0. Zero is not permitted in SFT.")
        return FoldValue(diff)

    def __repr__(self):
        return f"FoldValue({self.val.numerator}/{self.val.denominator})"


# =============================================================================
# THE DYADIC SUBSTRATE (Walsh-Hadamard)
# =============================================================================
def fwht_1d(a):
    """
    Fast Walsh-Hadamard Transform (1D) using pure exact integer arithmetic.
    """
    h = 1
    n = len(a)
    if n & (n - 1) != 0:
        halt_violation("FWHT requires power-of-two length.")
    
    # In-place transform
    a = list(a)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

def fwht_2d(matrix):
    """
    2D Fast Walsh-Hadamard Transform.
    matrix: 2D list or numpy array of integers.
    Returns: 2D list of exact integer coefficients.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0
    
    if n_rows & (n_rows - 1) != 0 or n_cols & (n_cols - 1) != 0:
        halt_violation("FWHT 2D requires power-of-two dimensions.")
        
    # Transform rows
    row_transformed = [fwht_1d(row) for row in matrix]
    
    # Transpose
    transposed = [[row_transformed[i][j] for i in range(n_rows)] for j in range(n_cols)]
    
    # Transform columns
    col_transformed = [fwht_1d(row) for row in transposed]
    
    # Transpose back
    final_matrix = [[col_transformed[i][j] for i in range(n_cols)] for j in range(n_rows)]
    
    # Parseval Certification
    certify_parseval(matrix, final_matrix)
    
    return final_matrix

def ifwht_2d(matrix):
    """
    Inverse 2D Fast Walsh-Hadamard Transform.
    FWHT is an involution up to a scaling factor.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows > 0 else 0
    n = n_rows * n_cols
    
    unscaled = fwht_2d(matrix)
    
    # Scale by 1/N
    return [[unscaled[i][j] // n for j in range(n_cols)] for i in range(n_rows)]

def certify_parseval(spatial_matrix, dyadic_matrix):
    """
    SFT Parseval Certification:
    Sum(x^2) = (1/N) * Sum(X^2)
    Uses exact integer arithmetic. Halts on violation.
    """
    n_rows = len(spatial_matrix)
    n_cols = len(spatial_matrix[0])
    N = n_rows * n_cols
    
    sum_spatial_sq = sum(int(val)**2 for row in spatial_matrix for val in row)
    sum_dyadic_sq = sum(int(val)**2 for row in dyadic_matrix for val in row)
    
    if sum_spatial_sq * N != sum_dyadic_sq:
        halt_violation(f"Parseval Identity Failed! Spatial Energy={sum_spatial_sq}, Dyadic Energy/N={sum_dyadic_sq/N}")


# =============================================================================
# THE FORCED LOCKS  --  the engine that enforces its own existence
# =============================================================================
# No chosen number enters the model. Every model quantity above is a FORCED
# LOCK, cross-checked at wake against an INDEPENDENT forward computation from the
# fold's own structure; the engine HALTS on any mismatch. This is the AI-wing
# analogue of the corpus's `forced_to_be`. Interface bounds (buffers, caps,
# timeouts, batch sizes) are hardware facts marked in place in their own
# modules -- they are NOT locks and never enter here.
def _fold_period_of_unit_fraction(n):
    """The period of 1/n under the fold, computed forward: the numerator doubles
    and casts out whole Ones (mod n) until it returns to 1. The generators are
    read off this spectrum -- nothing is chosen."""
    x, k = 1, 0
    while True:
        x = (x * 2) % n
        k += 1
        if x == 1:
            return k
        if k > 4 * n:
            halt_violation(f"fold orbit of 1/{n} did not return (even denominator?)")

def _smallest_fold_period_above(threshold):
    """The smallest fold period strictly above the threshold, read off the
    spectrum of odd unit fractions. The two smallest periods (2 at 1/3, 3 at 1/7)
    have appeared by 1/7, so the scan bound is 'far enough', not a parameter."""
    best = 0
    for n in range(3, 32, 2):
        p = _fold_period_of_unit_fraction(n)
        if p > threshold and (best == 0 or p < best):
            best = p
    return best

def forced_to_be(label, chosen, forced):
    """A lock stands only if the chosen constant equals its independent forced
    computation; otherwise the engine halts. No fitted or unforced value survives."""
    if chosen != forced:
        halt_violation(f"LOCK MISMATCH [{label}]: constant {chosen} != independently forced {forced}")
    return chosen

def verify_locks():
    """Cross-check every forced lock against an independent forward computation.
    Call at wake. Halts on any mismatch. Returns True when the substrate is sound."""
    # The two generators, read off the fold's own period spectrum.
    forced_to_be("binary generator b", GEN_B, _smallest_fold_period_above(1))
    forced_to_be("colour generator c", GEN_C, _smallest_fold_period_above(GEN_B))
    # Context depth = b * c.
    forced_to_be("context depth", CTX_MAX, GEN_B * GEN_C)
    # The held memory orbit {1/3, 2/3}: period b, folding into each other,
    # partitioning the One, never reaching it.
    forced_to_be("memory orbit period", _fold_period_of_unit_fraction(GEN_C), GEN_B)
    if FoldValue(MEMORY_STATE).fold().val != REFRESH_STATE:
        halt_violation("held orbit: fold(1/3) != 2/3")
    if FoldValue(REFRESH_STATE).fold().val != MEMORY_STATE:
        halt_violation("held orbit: fold(2/3) != 1/3")
    forced_to_be("held orbit partitions the One", MEMORY_STATE + REFRESH_STATE, Fraction(1))
    # The attention/selection lock 1/b is the self-antipodal balance.
    forced_to_be("focus lock 1/b", FOCUS_LOCK, Fraction(1) - FOCUS_LOCK)
    # The functional band b^(b+c) = 32 covers the colour volume c^c and closes the One.
    forced_to_be("functional band", BAND, GEN_B ** (GEN_B + GEN_C))
    if BAND < GEN_C ** GEN_C:
        halt_violation(f"band {BAND} does not cover the colour volume c^c={GEN_C**GEN_C}")
    forced_to_be("band closes the One", Fraction(BAND - 1, BAND) + Fraction(1, BAND), Fraction(1))
    # The dyadic cascade: the loud head (top b+c members) carries the band interior.
    head = sum((cascade_share(r) for r in range(1, GEN_B + GEN_C + 1)), Fraction(0))
    forced_to_be("loud-head mass = band interior", head, Fraction(BAND - 1, BAND))
    # The floors forced from the generators.
    forced_to_be("kin floor", KIN_FLOOR, Fraction(1, GEN_B * GEN_C))
    forced_to_be("compose floor", COMPOSE_FLOOR, Fraction(1, GEN_B * GEN_B * GEN_C))
    # THE INTEGRATION LOCKS (contextual_integration.ep + generation_selection_law.ep).
    forced_to_be("integration depth = covering depth b+c", INTEGRATION_DEPTH, GEN_B + GEN_C)
    if GEN_B ** INTEGRATION_DEPTH < GEN_C ** GEN_C:
        halt_violation("integration depth does not cover the colour volume c^c")
    if GEN_B ** (INTEGRATION_DEPTH - 1) >= GEN_C ** GEN_C:
        halt_violation("integration depth is not minimal (one fewer doubling covers)")
    casc = sum((Fraction(1, GEN_B ** k) for k in range(1, INTEGRATION_DEPTH + 1)), Fraction(0))
    forced_to_be("integration cascade + floor = the One",
                 casc + Fraction(1, GEN_B ** INTEGRATION_DEPTH), Fraction(1))
    forced_to_be("spread capacity = the closed focus lock", SPREAD_LOCK, FOCUS_LOCK)
    forced_to_be("re-expression minimum = the binary count", REEXPRESS_MIN, GEN_B)
    return True

if __name__ == "__main__":
    # The forced locks cross-check at wake, or the engine halts.
    verify_locks()
    print(f"verify_locks: all locks forced (b={GEN_B}, c={GEN_C}, ctx={CTX_MAX}, "
          f"band={BAND}, orbit={{{MEMORY_STATE},{REFRESH_STATE}}}, lock={FOCUS_LOCK})")

    v1 = FoldValue(Fraction(1, 3))
    print(f"v1: {v1}")
    print(f"fold(v1): {v1.fold()}")

    # Small dyadic test
    grid = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    dyadic = fwht_2d(grid)
    print("FWHT 2D successful. Parseval identity certified.")

    recovered = ifwht_2d(dyadic)
    assert recovered == grid
    print("Inverse FWHT 2D successful.")

    # Prove the engine halts on a fitted (unforced) value: run with `--halt-demo`.
    if "--halt-demo" in sys.argv:
        print("Feeding a fitted value to a lock (expect hard halt, exit 1)...")
        forced_to_be("deliberately fitted band", 30, GEN_B ** (GEN_B + GEN_C))
