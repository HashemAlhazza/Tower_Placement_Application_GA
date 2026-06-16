import os
import csv
import time
import numpy as np
import rasterio
from rasterio.transform import xy
from rasterio.transform import Affine
from scipy.ndimage import zoom
from pyproj import Transformer
import folium

# ----------------------------
# Input files
# ----------------------------
POP_PATH = "data/egy_pop_2025_CN_1km_R2025A_UA_v1.tif"

# ----------------------------
# Output files
# ----------------------------
OUT_CSV = "/kaggle/working/ga_towers.csv"
OUT_HTML = "/kaggle/working/ga_towers_map.html"
OUT_HISTORY = "/kaggle/working/ga_fitness_history.csv"

# ----------------------------
# Tower settings
# ----------------------------
CAPEX_BUDGET = 10_000_000
TOWER_COST_BASE = 200_000
MAX_TOWERS = 30
TRAIN_DOWNSAMPLE = 2  # downsampled grid used during GA fitness evaluation

# ----------------------------
# GA hyper-parameters
# ----------------------------
N_POP = 40  # number of layouts (chromosomes) per generation
N_GENERATIONS = 100  # total generations
ELITE_SIZE = 4  # top-N layouts copied unchanged to next generation
TOURNAMENT_K = 5  # tournament size for parent selection
CROSSOVER_RATE = 0.85  # probability a pair of parents performs crossover
MUTATION_RATE = 0.15  # probability each tower gene has its position perturbed
MUTATION_SIGMA = 0.06  # std-dev of Gaussian position perturbation (normalised 0-1)
TOGGLE_PROB = 0.10  # probability a mutation flips active/inactive flag
INITIAL_ACTIVE = 20  # towers active in randomly generated chromosomes
PRINT_EVERY = 10  # print progress every N generations

# ----------------------------
# Coverage targets
# ----------------------------
TARGET_POP_COVERAGE = 0.55
TARGET_WEST_COVERAGE = 0.25
TARGET_GEO_COVERAGE = 0.40
TARGET_ZONE_COVERAGE = 0.70
COVERAGE_BONUS = 3.0

# ----------------------------
# Zone grid -- 5x6 = 30 zones
# ----------------------------
ZONE_GRID_ROWS = 5
ZONE_GRID_COLS = 6
ZONE_COV_WEIGHT = 8.0

# ----------------------------
# Radio model settings
# ----------------------------
FREQ_MHZ = 3500.0
TX_POWER_DBM = 43.0
NOISE_DBM = -100.0
MAX_RANGE_KM = 2.5
ANTENNA_GAIN_DB = 15.0
CABLE_LOSS_DB = 3.0
EFFECTIVE_TX_DBM = TX_POWER_DBM + ANTENNA_GAIN_DB - CABLE_LOSS_DB  # 55 dBm

# ----------------------------
# Thresholds & penalties
# ----------------------------
SINR_THRESHOLD_DB = 5.0
MIN_SEP_METERS = 2000.0
ELEV_PENALTY = 0.5

# ----------------------------
# Post-processing repair
# ----------------------------
REPAIR_MAX_ITER = 10
REPAIR_SEARCH_RADII_PIX = [8, 16, 24, 32, 48, 64]
REPAIR_MAX_CANDIDATES_PER_RADIUS = 200

# ----------------------------
# Coordinate transformers
# ----------------------------
_to_wgs84 = Transformer.from_crs("EPSG:32636", "EPSG:4326", always_xy=True)
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32636", always_xy=True)


# ============================================================
# Shared helpers
# ============================================================


def dbm_to_mw(dbm):
    return 10 ** (dbm / 10.0)


def mw_to_dbm(mw):
    return 10 * np.log10(np.maximum(mw, 1e-12))


def urban_macro_path_loss_db(dist_km, freq_mhz):
    d_m = np.maximum(dist_km * 1000.0, 10.0)
    f_ghz = freq_mhz / 1000.0
    shadow = 10.0
    building_pen = 20.0  # dB -- indoor penetration loss, typical urban at 3500 MHz
    return (
        28.0 + 22 * np.log10(d_m) + 20 * np.log10(f_ghz) + 4.0 + shadow + building_pen
    )


def clean_raster(arr, nodata=None):
    arr = arr.astype(np.float32)
    if nodata is not None:
        arr = np.where(arr == nodata, 0.0, arr)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def clean_population(arr, nodata=None):
    arr = arr.astype(np.float32)
    if nodata is not None:
        arr = np.where(arr == nodata, 0.0, arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.where(arr < 0, 0.0, arr)
    return arr


def pixel_to_lonlat(transform, x, y):
    # Raster is EPSG:4326 -- rasterio.transform.xy returns lon/lat directly
    lon, lat = xy(transform, y, x, offset="center")
    return float(lon), float(lat)


def lonlat_to_pixel(transform, lon, lat, width, height):
    # Raster is EPSG:4326 -- apply inverse affine directly on lon/lat
    col, row = ~transform * (lon, lat)
    col = int(np.clip(round(col), 0, width - 1))
    row = int(np.clip(round(row), 0, height - 1))
    return col, row


def tower_cost():
    return TOWER_COST_BASE


def zone_coverage_score(towers_xy, height, width, n_rows=None, n_cols=None):
    if n_rows is None:
        n_rows = ZONE_GRID_ROWS
    if n_cols is None:
        n_cols = ZONE_GRID_COLS
    if not towers_xy:
        return 0.0
    zone_h = height / n_rows
    zone_w = width / n_cols
    occupied = set()
    for x, y in towers_xy:
        zr = int(np.clip(y // zone_h, 0, n_rows - 1))
        zc = int(np.clip(x // zone_w, 0, n_cols - 1))
        occupied.add((zr, zc))
    return len(occupied) / (n_rows * n_cols)


def make_zone_seeds(pop, height, width, n_rows=None, n_cols=None):
    """
    Return one pixel index per zone (the highest-population pixel in that zone,
    or the zone centre if the zone is empty). Used to intelligently seed the
    initial GA population -- this is standard GA warm-starting, not SAC-specific.

    n_rows / n_cols default to the global ZONE_GRID_ROWS/COLS when called from
    the full-raster load_data() path; crop_data_to_bbox passes dynamic values.
    """
    if n_rows is None:
        n_rows = ZONE_GRID_ROWS
    if n_cols is None:
        n_cols = ZONE_GRID_COLS
    zone_h = height / n_rows
    zone_w = width / n_cols
    seeds = []
    for zr in range(n_rows):
        for zc in range(n_cols):
            r0, r1 = int(zr * zone_h), int((zr + 1) * zone_h)
            c0, c1 = int(zc * zone_w), int((zc + 1) * zone_w)
            patch = pop[r0:r1, c0:c1]
            if patch.size > 0 and patch.max() > 0:
                lr, lc = np.unravel_index(np.argmax(patch), patch.shape)
                global_idx = np.ravel_multi_index((r0 + lr, c0 + lc), (height, width))
            else:
                cr = min(r0 + int(zone_h // 2), height - 1)
                cc = min(c0 + int(zone_w // 2), width - 1)
                global_idx = np.ravel_multi_index((cr, cc), (height, width))
            seeds.append(global_idx)
    return np.array(seeds)


def find_conflicting_pairs_xy(towers_xy, transform, min_sep_meters):
    pairs = []
    utm_pts = []
    for x, y in towers_xy:
        lon, lat = pixel_to_lonlat(transform, x, y)
        # Convert to UTM for metric distance calculation
        e, n = _to_utm.transform(lon, lat)
        utm_pts.append((e, n))
    n = len(towers_xy)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(
                (utm_pts[i][0] - utm_pts[j][0]) ** 2
                + (utm_pts[i][1] - utm_pts[j][1]) ** 2
            )
            if d < min_sep_meters:
                pairs.append((i, j, d))
    return pairs


def tower_spacing_ok_xy(candidate_xy, other_xys, transform, min_sep_meters):
    cx, cy = candidate_xy
    clon, clat = pixel_to_lonlat(transform, cx, cy)
    ce, cn = _to_utm.transform(clon, clat)
    for ox, oy in other_xys:
        olon, olat = pixel_to_lonlat(transform, ox, oy)
        oe, on_ = _to_utm.transform(olon, olat)
        if np.sqrt((ce - oe) ** 2 + (cn - on_) ** 2) < min_sep_meters:
            return False
    return True


# ============================================================
# Data loading
# ============================================================


def load_data():
    with rasterio.open(POP_PATH) as src:
        raw = src.read(1)
        print("RAW MIN:", raw.min())
        print("RAW MAX:", raw.max())
        print("RAW SUM:", raw.sum())

        pop = clean_population(src.read(1), src.nodatavals[0])

        print("CLEAN SUM:", pop.sum())
        transform = src.transform
        height, width = src.height, src.width

    rows, cols = np.indices((height, width))
    xs, ys = xy(transform, rows, cols, offset="center")
    easting_grid = np.array(xs, dtype=np.float32).reshape((height, width))
    northing_grid = np.array(ys, dtype=np.float32).reshape((height, width))

    top_pixels = make_zone_seeds(pop, height, width)
    inhabited_mask = pop > 10.0
    inhabited_count = float(inhabited_mask.sum())
    print(f"Inhabited pixels : {inhabited_count:.0f} / {height * width}")
    print(
        f"Zone seeds       : {len(top_pixels)} seeds across "
        f"{ZONE_GRID_ROWS}x{ZONE_GRID_COLS} zones"
    )

    ds = TRAIN_DOWNSAMPLE
    pop_ds = np.array(zoom(pop, 1.0 / ds, order=1), dtype=np.float32)
    easting_ds = np.array(zoom(easting_grid, 1.0 / ds, order=1), dtype=np.float32)
    northing_ds = np.array(zoom(northing_grid, 1.0 / ds, order=1), dtype=np.float32)
    inhabited_ds = zoom(inhabited_mask.astype(np.float32), 1.0 / ds, order=1) > 0.5
    height_ds, width_ds = pop_ds.shape
    transform_ds = Affine(
        transform.a * ds,
        transform.b,
        transform.c,
        transform.d,
        transform.e * ds,
        transform.f,
    )

    return {
        # Full-resolution grids
        "pop": pop,
        "transform": transform,
        "height": height,
        "width": width,
        "easting_grid": easting_grid,
        "northing_grid": northing_grid,
        "inhabited_mask": inhabited_mask,
        "inhabited_count": inhabited_count,
        # Downsampled grids (used during GA fitness evaluation for speed)
        "pop_ds": pop_ds,
        "transform_ds": transform_ds,
        "height_ds": height_ds,
        "width_ds": width_ds,
        "easting_ds": easting_ds,
        "northing_ds": northing_ds,
        "inhabited_ds": inhabited_ds,
        "inhabited_count_ds": float(np.asarray(inhabited_ds).sum()),
        # Zone seeds for population initialisation
        "top_pixels": top_pixels,
        "full_height": height,
        "full_width": width,
    }


# ============================================================
# Radio computation
# ============================================================


def compute_radio_maps(
    towers_lonlat, transform, width, height, easting_grid, northing_grid
):
    if not towers_lonlat:
        zeros = np.full((height, width), -200.0, dtype=np.float32)
        return zeros, zeros, zeros

    # easting_grid / northing_grid are lon/lat in degrees (raster is EPSG:4326).
    # Convert the whole grid to UTM metres once, then compute distances.
    grid_e, grid_n = _to_utm.transform(easting_grid, northing_grid)

    rx_stack_dbm = []
    for tower_lon, tower_lat in towers_lonlat:
        tower_e, tower_n = _to_utm.transform(tower_lon, tower_lat)
        dx = grid_e - tower_e
        dy = grid_n - tower_n
        dist_km = np.sqrt(dx**2 + dy**2) / 1000.0

        rx_dbm = EFFECTIVE_TX_DBM - urban_macro_path_loss_db(dist_km, FREQ_MHZ)
        rx_dbm = np.where(dist_km > MAX_RANGE_KM, -200.0, rx_dbm)
        rx_stack_dbm.append(rx_dbm)

    stack = np.stack(rx_stack_dbm, axis=0)
    signal_dbm = np.max(stack, axis=0)

    stack_mw = dbm_to_mw(stack)
    total_mw = np.sum(stack_mw, axis=0)
    serving_mw = np.max(stack_mw, axis=0)
    interference_mw = np.maximum(total_mw - serving_mw, 1e-12)

    sinr = serving_mw / (interference_mw + dbm_to_mw(NOISE_DBM))
    sinr_db = 10 * np.log10(np.maximum(sinr, 1e-12))

    return signal_dbm, mw_to_dbm(interference_mw), sinr_db


# ============================================================
# Scoring
# ============================================================


def score_layout(
    pop,
    sinr_db,
    interference_dbm,
    towers_xy,
    height,
    width,
    n_active,
    inhabited_mask,
    inhabited_count,
    zone_rows=None,
    zone_cols=None,
    max_towers=None,
):

    quality = 1.0 / (1.0 + np.exp(-(sinr_db - SINR_THRESHOLD_DB) / 3.0))
    smooth_score = float(np.sum(pop * quality)) / 1e6
    hard_served = float(pop[sinr_db >= SINR_THRESHOLD_DB].sum()) / 1e6
    total_pop_m = float(pop.sum()) / 1e6
    served_mask = sinr_db >= SINR_THRESHOLD_DB

    geographic_coverage = float((served_mask & inhabited_mask).sum()) / max(
        inhabited_count, 1.0
    )

    sinr_quality = (
        float(np.mean(np.clip(sinr_db[served_mask], 0.0, 20.0))) / 20.0
        if served_mask.any()
        else 0.0
    )

    interference_penalty = 0.0
    if served_mask.any():
        mean_interf = float(np.mean(interference_dbm[served_mask]))
        interference_penalty = max(0.0, mean_interf + 90.0) / 10.0

    zone_cov = zone_coverage_score(
        towers_xy, height, width, n_rows=zone_rows, n_cols=zone_cols
    )

    pop_coverage_ratio = hard_served / max(total_pop_m, 1e-6)
    coverage_bonus = 0.0
    if pop_coverage_ratio >= TARGET_POP_COVERAGE:
        coverage_bonus += COVERAGE_BONUS
    if geographic_coverage >= TARGET_GEO_COVERAGE:
        coverage_bonus += COVERAGE_BONUS
    if zone_cov >= TARGET_ZONE_COVERAGE:
        coverage_bonus += COVERAGE_BONUS

    # Tower-count penalty: discourage using more towers than the region needs.
    # Scales from 0 (no towers) to 1 (all MAX_TOWERS slots active).
    # Weight of 2.0 means burning the full budget costs ~2 fitness points --
    # enough to matter but not so heavy it prevents needed towers from activating.
    _cap = max_towers if max_towers else MAX_TOWERS
    tower_count_penalty = 2.0 * (n_active / max(_cap, 1))

    total = (
        1.0 * smooth_score
        + 3.0 * hard_served
        + 3.0 * geographic_coverage
        + ZONE_COV_WEIGHT * zone_cov
        + 1.0 * sinr_quality
        - 0.5 * interference_penalty
        - tower_count_penalty
        + coverage_bonus
    )

    return (
        total,
        smooth_score,
        hard_served,
        geographic_coverage,
        sinr_quality,
        interference_penalty,
        zone_cov,
    )


# ============================================================
# Chromosome representation
# ============================================================
#
# A chromosome is a numpy array of shape (MAX_TOWERS, 3):
#   chromosome[i] = [row_norm, col_norm, active_flag]
#   row_norm, col_norm in [0, 1]  (normalised to grid dimensions)
#   active_flag in {0.0, 1.0}
#
# All slots are freely evolved -- no positions are locked.
# ============================================================


def chrom_to_pixel_xy(chrom, height, width):
    """Return list of (x_col, y_row) pixel coordinates for active towers."""
    xy_list = []
    for i in range(len(chrom)):  # use chromosome's own length, not global MAX_TOWERS
        if chrom[i, 2] >= 0.5:
            x = int(np.clip(round(chrom[i, 1] * (width - 1)), 0, width - 1))
            y = int(np.clip(round(chrom[i, 0] * (height - 1)), 0, height - 1))
            xy_list.append((x, y))
    return xy_list


def chrom_to_lonlat(chrom, transform, height, width):
    """Return list of (lon, lat) for active towers."""
    lonlat = []
    for i in range(len(chrom)):  # use chromosome's own length, not global MAX_TOWERS
        if chrom[i, 2] >= 0.5:
            x = int(np.clip(round(chrom[i, 1] * (width - 1)), 0, width - 1))
            y = int(np.clip(round(chrom[i, 0] * (height - 1)), 0, height - 1))
            lon, lat = pixel_to_lonlat(transform, x, y)
            lonlat.append((lon, lat))
    return lonlat


# ============================================================
# GA operators
# ============================================================


def random_chromosome(
    top_pixels, full_height, full_width, rng, max_towers=None, initial_active=None
):
    """
    Create one random chromosome.
    Tower positions are sampled from zone-seed pixels so the initial
    population already has reasonable geographic spread.
    initial_active towers are turned on; the rest start inactive.

    max_towers / initial_active default to the global constants when called
    from the full-raster main() path; run_ga passes dynamic values for crops.
    """
    if max_towers is None:
        max_towers = MAX_TOWERS
    if initial_active is None:
        initial_active = INITIAL_ACTIVE

    chrom = np.zeros((max_towers, 3), dtype=np.float32)
    shuffled = top_pixels.copy()
    rng.shuffle(shuffled)

    for i in range(max_towers):
        if i < len(shuffled):
            row, col = np.unravel_index(shuffled[i], (full_height, full_width))
        else:
            row = rng.integers(0, full_height)
            col = rng.integers(0, full_width)
        chrom[i, 0] = row / max(full_height - 1, 1)
        chrom[i, 1] = col / max(full_width - 1, 1)
        chrom[i, 2] = 0.0

    # Activate initial_active random slots
    slots = list(range(max_towers))
    rng.shuffle(slots)
    for i in slots[:initial_active]:
        chrom[i, 2] = 1.0

    return chrom


def evaluate_chromosome(chrom, data, use_downsampled=True):
    """
    Compute fitness score for one chromosome.
    Returns (total_score, detail_dict).
    """
    if use_downsampled:
        pop = data["pop_ds"]
        transform = data["transform_ds"]
        height = data["height_ds"]
        width = data["width_ds"]
        easting = data["easting_ds"]
        northing = data["northing_ds"]
        inh_mask = data["inhabited_ds"]
        inh_count = data["inhabited_count_ds"]
    else:
        pop = data["pop"]
        transform = data["transform"]
        height = data["height"]
        width = data["width"]
        easting = data["easting_grid"]
        northing = data["northing_grid"]
        inh_mask = data["inhabited_mask"]
        inh_count = data["inhabited_count"]

    lonlat = chrom_to_lonlat(chrom, transform, height, width)
    xy_list = chrom_to_pixel_xy(chrom, height, width)

    if not lonlat:
        return -999.0, {}

    _, interf_dbm, sinr_db = compute_radio_maps(
        lonlat, transform, width, height, easting, northing
    )

    total, smooth, hard, geo_cov, sinr_q, interf_pen, zone_cov = score_layout(
        pop,
        sinr_db,
        interf_dbm,
        xy_list,
        height,
        width,
        len(xy_list),
        inh_mask,
        inh_count,
        zone_rows=data.get("dynamic_zone_rows"),
        zone_cols=data.get("dynamic_zone_cols"),
        max_towers=data.get("dynamic_max_towers"),
    )

    detail = {
        "total_score": total,
        "smooth_score": smooth,
        "hard_served_M": hard,
        "geographic_coverage": geo_cov,
        "zone_coverage": zone_cov,
        "sinr_quality": sinr_q,
        "interference_penalty": interf_pen,
        "n_active": len(xy_list),
    }
    return total, detail


def tournament_select(population, fitnesses, k, rng):
    """
    Pick one parent via k-way tournament selection.
    Randomly sample k individuals; return the one with the highest fitness.
    """
    indices = rng.choice(len(population), size=k, replace=False)
    best = indices[np.argmax([fitnesses[i] for i in indices])]
    return population[best].copy()


def uniform_crossover(parent_a, parent_b, rng):
    """
    Uniform crossover at the tower-slot level.
    For each slot, randomly pick genes from parent A or B.
    Returns two children.
    """
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    for i in range(len(parent_a)):  # use chromosome's own length
        if rng.random() < 0.5:
            child_a[i] = parent_b[i].copy()
            child_b[i] = parent_a[i].copy()
    return child_a, child_b


def mutate(chrom, full_height, full_width, rng):
    """
    Apply mutations to tower slots:
      - With probability MUTATION_RATE: perturb (row, col) by Gaussian noise
      - With probability TOGGLE_PROB:   flip the active flag
    Positions are clipped to [0, 1] after perturbation.
    """
    for i in range(len(chrom)):  # use chromosome's own length
        if rng.random() < MUTATION_RATE:
            chrom[i, 0] = float(
                np.clip(chrom[i, 0] + rng.normal(0, MUTATION_SIGMA), 0.0, 1.0)
            )
            chrom[i, 1] = float(
                np.clip(chrom[i, 1] + rng.normal(0, MUTATION_SIGMA), 0.0, 1.0)
            )
        if rng.random() < TOGGLE_PROB:
            chrom[i, 2] = 1.0 - chrom[i, 2]
    return chrom


# ============================================================
# Main GA loop
# ============================================================


def run_ga(data):
    rng = np.random.default_rng(42)
    full_height = data["full_height"]
    full_width = data["full_width"]
    top_pixels = data["top_pixels"]

    # Read region-adaptive overrides if present (set by crop_data_to_bbox),
    # otherwise fall back to the global constants (used in the full-raster main() path).
    dyn_max_towers = data.get("dynamic_max_towers", MAX_TOWERS)
    dyn_initial_active = data.get("dynamic_initial_active", INITIAL_ACTIVE)

    # --------------------------------------------------
    # 1. Initialise population
    # --------------------------------------------------
    print(
        f"\nInitialising population of {N_POP} chromosomes "
        f"(max_towers={dyn_max_towers}, initial_active={dyn_initial_active}) ..."
    )
    population = [
        random_chromosome(
            top_pixels,
            full_height,
            full_width,
            rng,
            max_towers=dyn_max_towers,
            initial_active=dyn_initial_active,
        )
        for _ in range(N_POP)
    ]

    best_chrom = None
    best_fitness = -np.inf
    best_detail = {}
    history = []  # [(gen, best_fitness, mean_fitness)]

    t0 = time.time()

    for gen in range(1, N_GENERATIONS + 1):

        # --------------------------------------------------
        # 2. Evaluate all chromosomes
        # --------------------------------------------------
        fitnesses = []
        details = []
        for chrom in population:
            score, detail = evaluate_chromosome(chrom, data, use_downsampled=True)
            fitnesses.append(score)
            details.append(detail)

        fitnesses = np.array(fitnesses, dtype=np.float64)

        # Track best across all generations
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_chrom = population[gen_best_idx].copy()
            best_detail = details[gen_best_idx]

        mean_fit = float(np.mean(fitnesses))
        history.append((gen, float(best_fitness), mean_fit))

        if gen % PRINT_EVERY == 0 or gen == 1:
            d = best_detail
            elapsed = time.time() - t0
            print(
                f"[Gen {gen:>4}/{N_GENERATIONS}] "
                f"best={best_fitness:.3f} | mean={mean_fit:.3f} | "
                f"towers={d.get('n_active', 0)} | "
                f"served={d.get('hard_served_M', 0):.3f}M | "
                f"geo={d.get('geographic_coverage', 0):.2%} | "
                f"zone={d.get('zone_coverage', 0):.2%} | "
                f"elapsed={elapsed:.0f}s"
            )

        # --------------------------------------------------
        # 3. Build next generation
        # --------------------------------------------------
        # 3a. Elitism -- copy top ELITE_SIZE unchanged
        elite_idx = np.argsort(fitnesses)[::-1][:ELITE_SIZE]
        next_gen = [population[i].copy() for i in elite_idx]

        # 3b. Fill remaining slots via tournament selection + crossover + mutation
        while len(next_gen) < N_POP:
            p_a = tournament_select(population, fitnesses, TOURNAMENT_K, rng)
            p_b = tournament_select(population, fitnesses, TOURNAMENT_K, rng)

            if rng.random() < CROSSOVER_RATE:
                child_a, child_b = uniform_crossover(p_a, p_b, rng)
            else:
                child_a, child_b = p_a.copy(), p_b.copy()

            child_a = mutate(child_a, full_height, full_width, rng)
            child_b = mutate(child_b, full_height, full_width, rng)

            next_gen.append(child_a)
            if len(next_gen) < N_POP:
                next_gen.append(child_b)

        population = next_gen

    print(f"\nGA complete. Best fitness = {best_fitness:.4f}")
    return best_chrom, best_fitness, best_detail, history


# ============================================================
# Post-processing repair
# ============================================================


def candidate_local_score(
    pop, candidate_xy, existing_xy, transform, easting_grid, northing_grid
):
    x, y = candidate_xy
    lonlat = [pixel_to_lonlat(transform, ox, oy) for ox, oy in existing_xy]
    lonlat.append(pixel_to_lonlat(transform, x, y))
    _, interf_dbm, sinr_db = compute_radio_maps(
        lonlat, transform, pop.shape[1], pop.shape[0], easting_grid, northing_grid
    )
    r0, r1 = max(0, y - 6), min(pop.shape[0], y + 7)
    c0, c1 = max(0, x - 6), min(pop.shape[1], x + 7)
    local_pop = float(pop[r0:r1, c0:c1].sum())
    local_served = float(
        pop[r0:r1, c0:c1][sinr_db[r0:r1, c0:c1] >= SINR_THRESHOLD_DB].sum()
    )
    interf_pen = max(0.0, float(np.mean(interf_dbm[r0:r1, c0:c1])) + 90.0) / 10.0
    return local_pop + 2.0 * local_served - 5000.0 * interf_pen


def repair_close_towers(data, final_xy, final_lonlat):
    if len(final_xy) <= 1:
        return final_xy, final_lonlat

    repaired_xy = list(final_xy)
    repaired_lonlat = list(final_lonlat)
    transform = data["transform"]
    pop = data["pop"]
    easting_grid = data["easting_grid"]
    northing_grid = data["northing_grid"]
    height = data["height"]
    width = data["width"]
    # Use dynamic min separation if available (set by crop_data_to_bbox)
    min_sep = data.get("dynamic_min_sep_m", MIN_SEP_METERS)

    print(
        f"\nStarting post-processing repair for close towers (min_sep={min_sep:.0f}m)..."
    )

    for _ in range(REPAIR_MAX_ITER):
        pairs = find_conflicting_pairs_xy(repaired_xy, transform, min_sep)
        if not pairs:
            print("No spacing conflicts remain.")
            break

        changed_any = False
        for i, j, dist_m in pairs:
            xi, yi = repaired_xy[i]
            xj, yj = repaired_xy[j]
            pop_i = float(
                pop[
                    max(0, yi - 3) : min(height, yi + 4),
                    max(0, xi - 3) : min(width, xi + 4),
                ].sum()
            )
            pop_j = float(
                pop[
                    max(0, yj - 3) : min(height, yj + 4),
                    max(0, xj - 3) : min(width, xj + 4),
                ].sum()
            )
            move_idx = j if pop_i >= pop_j else i
            base_x, base_y = repaired_xy[move_idx]
            others = [xy_ for k, xy_ in enumerate(repaired_xy) if k != move_idx]

            best_xy = None
            best_score = -np.inf

            for rad in REPAIR_SEARCH_RADII_PIX:
                candidates = []
                for dy in range(-rad, rad + 1):
                    for dx in range(-rad, rad + 1):
                        if dx == 0 and dy == 0:
                            continue
                        if dx * dx + dy * dy > rad * rad:
                            continue
                        cx = int(np.clip(base_x + dx, 0, width - 1))
                        cy = int(np.clip(base_y + dy, 0, height - 1))
                        candidates.append((cx, cy))

                if len(candidates) > REPAIR_MAX_CANDIDATES_PER_RADIUS:
                    idx = np.random.choice(
                        len(candidates), REPAIR_MAX_CANDIDATES_PER_RADIUS, replace=False
                    )
                    candidates = [candidates[t] for t in idx]

                for cand in candidates:
                    if not tower_spacing_ok_xy(cand, others, transform, min_sep):
                        continue
                    score = candidate_local_score(
                        pop, cand, others, transform, easting_grid, northing_grid
                    )
                    if score > best_score:
                        best_score = score
                        best_xy = cand

                if best_xy is not None:
                    break

            if best_xy is not None:
                repaired_xy[move_idx] = best_xy
                repaired_lonlat[move_idx] = pixel_to_lonlat(
                    transform, best_xy[0], best_xy[1]
                )
                changed_any = True

        if not changed_any:
            print("Repair step could not improve remaining conflicts further.")
            break

    final_pairs = find_conflicting_pairs_xy(repaired_xy, transform, min_sep)
    print(f"Remaining conflicts after repair: {len(final_pairs)}")
    return repaired_xy, repaired_lonlat


# ============================================================
# Folium map
# ---------------------------------------------------------------
# Matches the SAC map format exactly for side-by-side comparison:
#   Blue circles  = GA-placed towers
#   Red circles   = top zone-seed positions (highest-pop pixel per
#                   zone, shown for reference -- NOT fixed inputs)
# ============================================================


def make_map(ga_lonlat, center_lonlat):
    """
    All towers are GA-placed -- shown as blue circles,
    matching the SAC map's blue style for agent-placed towers.
    """
    m = folium.Map(
        location=[center_lonlat[1], center_lonlat[0]],
        zoom_start=13,
        tiles="CartoDB positron",
    )

    for i, (lon, lat) in enumerate(ga_lonlat):
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=f"GA Tower {i}  ({lat:.5f}, {lon:.5f})",
            color="blue",
            fill=True,
            fill_opacity=0.9,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ============================================================
# Save results
# ============================================================


def save_results(
    data, final_lonlat, final_xy, best_detail, sinr_after, interference_after, history
):

    served_pop = float(data["pop"][sinr_after >= SINR_THRESHOLD_DB].sum())
    total_pop = float(data["pop"].sum())
    coverage_pct = 100.0 * served_pop / max(total_pop, 1.0)
    geo_pct = (
        100.0
        * float((sinr_after >= SINR_THRESHOLD_DB).sum())
        / (data["height"] * data["width"])
    )
    budget_spent = sum(tower_cost() for _ in final_xy)

    served_mask = sinr_after >= SINR_THRESHOLD_DB
    mean_sinr_served = (
        float(np.mean(sinr_after[served_mask])) if served_mask.any() else 0.0
    )
    mean_interf = (
        float(np.mean(interference_after[served_mask])) if served_mask.any() else -200.0
    )

    # Use dynamic zone grid and max_towers if available
    zone_rows = data.get("dynamic_zone_rows", ZONE_GRID_ROWS)
    zone_cols = data.get("dynamic_zone_cols", ZONE_GRID_COLS)
    max_towers = data.get("dynamic_max_towers", MAX_TOWERS)

    zone_cov = zone_coverage_score(
        final_xy, data["height"], data["width"], n_rows=zone_rows, n_cols=zone_cols
    )

    print(f"\n========== GA FINAL RESULTS ==========")
    print(f"Towers placed        : {len(final_lonlat)} / {max_towers}  (all GA)")
    print(f"Budget spent         : ${budget_spent:,.0f} / ${CAPEX_BUDGET:,.0f}")
    print(
        f"Population served    : {served_pop:,.0f} / {total_pop:,.0f}  ({coverage_pct:.1f}%)"
    )
    print(f"Geographic coverage  : {geo_pct:.1f}% of grid area")
    print(f"Zone coverage        : {zone_cov:.1%} of {zone_rows}x{zone_cols} zones")
    print(f"Mean SINR (served)   : {mean_sinr_served:.1f} dB")
    print(f"Mean interference    : {mean_interf:.1f} dBm")
    print(
        f"SINR min/mean/max    : "
        f"{sinr_after.min():.1f} / {sinr_after.mean():.1f} / {sinr_after.max():.1f} dB"
    )
    print(f"Total GA score       : {best_detail.get('total_score', '?'):.4f}")

    # --------------------------------------------------
    # CSV of tower locations  (same schema as SAC output)
    # --------------------------------------------------
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "tower_id",
                "lon_wgs84",
                "lat_wgs84",
                "cost_usd",
                "sinr_at_tower_db",
                "type",
            ]
        )
        for i, (lon, lat) in enumerate(final_lonlat):
            x, y = final_xy[i]
            cost = tower_cost()
            col_t, row_t = lonlat_to_pixel(
                data["transform"], lon, lat, data["width"], data["height"]
            )
            sinr_t = float(sinr_after[row_t, col_t])
            w.writerow(
                [i, f"{lon:.6f}", f"{lat:.6f}", f"{cost:.0f}", f"{sinr_t:.1f}", "ga"]
            )
    print(f"Saved CSV     -> {OUT_CSV}")

    # --------------------------------------------------
    # Fitness history CSV  (useful for convergence comparison vs SAC)
    # --------------------------------------------------
    with open(OUT_HISTORY, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "best_fitness", "mean_fitness"])
        for row in history:
            w.writerow(row)
    print(f"Saved history -> {OUT_HISTORY}")

    # --------------------------------------------------
    # Folium map  (same format as SAC map for direct comparison)
    # Blue  = GA towers
    # Red   = zone-seed reference points
    # --------------------------------------------------
    if final_lonlat:
        center_lon = np.mean([p[0] for p in final_lonlat])
        center_lat = np.mean([p[1] for p in final_lonlat])
        m = make_map(final_lonlat, (center_lon, center_lat))
        m.save(OUT_HTML)
        print(f"Saved map     -> {OUT_HTML}  (blue={len(final_lonlat)} GA towers)")


# ============================================================
# Entry point
# ============================================================


def main():
    for path in [POP_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing input file: {path}")

    data = load_data()
    print("TOTAL EGYPT POPULATION:", data["pop"].sum())

    print(f"\nRaster size     : {data['height']} x {data['width']} pixels")
    print(
        f"Training grid   : {data['height_ds']} x {data['width_ds']} "
        f"(downsample={TRAIN_DOWNSAMPLE}x)"
    )
    print(f"Total population: {data['pop'].sum():,.0f}")
    print(f"CAPEX budget    : ${CAPEX_BUDGET:,.0f}")
    print(f"Max towers      : {MAX_TOWERS}  (all GA-controlled -- no locked slots)")
    print(f"\nGA settings:")
    print(f"  Population size  : {N_POP}")
    print(f"  Generations      : {N_GENERATIONS}")
    print(f"  Elitism          : top {ELITE_SIZE} survive each generation")
    print(f"  Tournament size  : {TOURNAMENT_K}")
    print(f"  Crossover rate   : {CROSSOVER_RATE}")
    print(f"  Mutation rate    : {MUTATION_RATE}  (sigma={MUTATION_SIGMA})")
    print(f"  Toggle prob      : {TOGGLE_PROB}")

    # --------------------------------------------------
    # Sanity check -- zone-seeded layout, no locks
    # --------------------------------------------------
    print("\nSanity check: testing radio model on a zone-seeded layout...")
    rng_check = np.random.default_rng(0)
    test_chrom = random_chromosome(
        data["top_pixels"], data["full_height"], data["full_width"], rng_check
    )
    test_lonlat = chrom_to_lonlat(
        test_chrom, data["transform"], data["height"], data["width"]
    )
    _, _, sinr_test = compute_radio_maps(
        test_lonlat,
        data["transform"],
        data["width"],
        data["height"],
        data["easting_grid"],
        data["northing_grid"],
    )
    served_test = float(data["pop"][sinr_test >= SINR_THRESHOLD_DB].sum())
    print(f"Zone-seeded coverage : {served_test:,.0f} population served")
    print(
        f"SINR max / mean      : {sinr_test.max():.1f} dB / {sinr_test.mean():.1f} dB"
    )
    if served_test == 0:
        print("WARNING: Zero coverage with zone-seeded layout -- check radio model!")
        return
    print("Radio model OK -- starting GA.\n")

    # --------------------------------------------------
    # Run GA
    # --------------------------------------------------
    best_chrom, best_fitness, best_detail, history = run_ga(data)

    # --------------------------------------------------
    # Final evaluation on full-resolution grid
    # --------------------------------------------------
    print("\nRe-evaluating best chromosome on full-resolution grid...")
    final_lonlat = chrom_to_lonlat(
        best_chrom, data["transform"], data["height"], data["width"]
    )
    final_xy = chrom_to_pixel_xy(best_chrom, data["height"], data["width"])

    # Repair any spacing violations in the final layout
    final_xy, final_lonlat = repair_close_towers(data, final_xy, final_lonlat)

    # Full-resolution radio maps for reporting
    _, interference_after, sinr_after = compute_radio_maps(
        final_lonlat,
        data["transform"],
        data["width"],
        data["height"],
        data["easting_grid"],
        data["northing_grid"],
    )

    # Update best_detail with full-resolution score
    score_full, best_detail = evaluate_chromosome(
        best_chrom, data, use_downsampled=False
    )
    best_detail["total_score"] = score_full

    save_results(
        data,
        final_lonlat,
        final_xy,
        best_detail,
        sinr_after,
        interference_after,
        history,
    )


def compute_region_params(south, north, west, east):
    """
    Derive GA parameters that scale with the physical size of the selected region.

    The idea is the same as how a city planner thinks: a 5 km2 neighbourhood
    needs 2-3 towers; a 500 km2 metro area needs 20-25.  Rather than forcing
    the GA to figure this out from scratch every run, we give it sensible
    starting bounds derived from the region's area.

    All thresholds below are calibrated so that:
      - a tiny  box (~25 km2)  -> 2-4 towers,  2x2 zone grid
      - a city  box (~300 km2) -> 10-15 towers, 3x4 zone grid
      - a large box (~900 km2) -> 20-25 towers, 5x6 zone grid  (original)

    Returns a dict of overrides to merge into the cropped data dict.
    """
    # --- Physical width and height in km (using simple flat-earth at Egypt's latitude) ---
    KM_PER_DEG_LAT = 111.0
    KM_PER_DEG_LON = 111.0 * np.cos(np.radians((north + south) / 2.0))

    width_km = abs(east - west) * KM_PER_DEG_LON
    height_km = abs(north - south) * KM_PER_DEG_LAT
    area_km2 = width_km * height_km

    print(f"Region size: {width_km:.1f} km x {height_km:.1f} km = {area_km2:.0f} km2")

    # --- Max towers: one tower per ~35 km2, capped between 3 and 30 ---
    # At MAX_RANGE_KM = 2.5, one tower's footprint is pi x 2.5^2 ~ 20 km2.
    # We want ~1.5x overlap factor for redundancy -> ~35 km2 per tower budget slot.
    dynamic_max_towers = int(np.clip(round(area_km2 / 35.0), 3, MAX_TOWERS))

    # --- Initial active towers: start the population at ~40% of max ---
    # This gives the GA room to both add and remove towers rather than
    # starting already saturated (which was the bug causing 23 towers on 120 km2).
    dynamic_initial_active = max(2, round(dynamic_max_towers * 0.4))

    # --- Minimum separation: scale with how densely towers can fit ---
    # Smaller region -> towers are closer -> tighten the min sep slightly.
    # Larger region -> we want towers spread out -> push min sep up.
    # Clamp between 1.5 km (dense urban) and 4 km (wide rural).
    dynamic_min_sep_m = float(
        np.clip(
            (area_km2 / dynamic_max_towers) ** 0.5 * 1000.0,
            1500.0,
            4000.0,
        )
    )

    # --- Zone grid: rows x cols should give ~ dynamic_max_towers zones ---
    # We want the grid to match the region's aspect ratio so zones are square-ish.
    # Total zones ~ max_towers (one seed per potential tower slot).
    aspect = width_km / max(height_km, 0.1)
    zone_cols = max(2, round((dynamic_max_towers * aspect) ** 0.5))
    zone_rows = max(2, round(dynamic_max_towers / max(zone_cols, 1)))
    # Clamp to reasonable grid sizes
    zone_cols = int(np.clip(zone_cols, 2, 8))
    zone_rows = int(np.clip(zone_rows, 2, 6))

    print(
        f"Dynamic params -> max_towers={dynamic_max_towers}, "
        f"initial_active={dynamic_initial_active}, "
        f"min_sep={dynamic_min_sep_m:.0f}m, "
        f"zone_grid={zone_rows}x{zone_cols}"
    )

    return {
        "dynamic_max_towers": dynamic_max_towers,
        "dynamic_initial_active": dynamic_initial_active,
        "dynamic_min_sep_m": dynamic_min_sep_m,
        "dynamic_zone_rows": zone_rows,
        "dynamic_zone_cols": zone_cols,
        "area_km2": area_km2,
    }


def crop_data_to_bbox(data, south, north, west, east):
    from rasterio.transform import Affine
    from scipy.ndimage import zoom as scipy_zoom

    transform = data["transform"]

    def lonlat_to_rowcol_raw(lon, lat):
        """Return raw (unclipped) pixel row/col. Raster is EPSG:4326 so apply inverse affine directly."""
        col, row = ~transform * (lon, lat)
        return row, col

    row_min_raw, col_min_raw = lonlat_to_rowcol_raw(west, north)
    row_max_raw, col_max_raw = lonlat_to_rowcol_raw(east, south)

    print(f"Transform: {transform}")
    print(f"Raster size: height={data['height']}, width={data['width']}")
    print(f"row_min_raw={row_min_raw:.2f}, row_max_raw={row_max_raw:.2f}")
    print(f"col_min_raw={col_min_raw:.2f}, col_max_raw={col_max_raw:.2f}")
    print(f"Input bbox: south={south}, north={north}, west={west}, east={east}")

    # Detect selections completely outside the raster before clipping destroys the info
    completely_outside = (
        row_max_raw < 0
        or row_min_raw > data["height"] - 1
        or col_max_raw < 0
        or col_min_raw > data["width"] - 1
    )
    if completely_outside:
        return None

    row_min = int(np.clip(round(row_min_raw), 0, data["height"] - 1))
    row_max = int(np.clip(round(row_max_raw), 0, data["height"] - 1))
    col_min = int(np.clip(round(col_min_raw), 0, data["width"] - 1))
    col_max = int(np.clip(round(col_max_raw), 0, data["width"] - 1))

    if row_max - row_min < 3 or col_max - col_min < 3:
        return None

    pop_crop = data["pop"][row_min:row_max, col_min:col_max]
    east_crop = data["easting_grid"][row_min:row_max, col_min:col_max]
    north_crop = data["northing_grid"][row_min:row_max, col_min:col_max]

    height, width = pop_crop.shape

    new_transform = Affine(
        transform.a,
        transform.b,
        transform.c + col_min * transform.a,
        transform.d,
        transform.e,
        transform.f + row_min * transform.e,
    )

    # --- Compute region-adaptive parameters BEFORE building zone seeds ---
    region_params = compute_region_params(south, north, west, east)
    zone_rows = region_params["dynamic_zone_rows"]
    zone_cols = region_params["dynamic_zone_cols"]

    inhabited_mask = pop_crop > 10.0
    inhabited_count = float(inhabited_mask.sum())

    # make_zone_seeds now uses the dynamic grid instead of the global 5x6
    top_pixels = make_zone_seeds(
        pop_crop, height, width, n_rows=zone_rows, n_cols=zone_cols
    )

    ds = TRAIN_DOWNSAMPLE
    pop_ds = np.asarray(scipy_zoom(pop_crop, 1.0 / ds, order=1)).astype(np.float32)
    east_ds = np.asarray(scipy_zoom(east_crop, 1.0 / ds, order=1)).astype(np.float32)
    nort_ds = np.asarray(scipy_zoom(north_crop, 1.0 / ds, order=1)).astype(np.float32)
    inh_ds = (
        np.array(scipy_zoom(inhabited_mask.astype(np.float32), 1.0 / ds, order=1)) > 0.5
    )

    height_ds, width_ds = pop_ds.shape
    transform_ds = Affine(
        new_transform.a * ds,
        new_transform.b,
        new_transform.c,
        new_transform.d,
        new_transform.e * ds,
        new_transform.f,
    )

    return {
        "pop": pop_crop,
        "transform": new_transform,
        "height": height,
        "width": width,
        "easting_grid": east_crop,
        "northing_grid": north_crop,
        "inhabited_mask": inhabited_mask,
        "inhabited_count": inhabited_count,
        "pop_ds": pop_ds,
        "easting_ds": east_ds,
        "northing_ds": nort_ds,
        "transform_ds": transform_ds,
        "height_ds": height_ds,
        "width_ds": width_ds,
        "inhabited_ds": inh_ds,
        "inhabited_count_ds": float(inh_ds.sum()),
        "top_pixels": top_pixels,
        "full_height": height,
        "full_width": width,
        # --- Region-adaptive overrides (consumed by run_ga) ---
        **region_params,
    }


if __name__ == "__main__":
    main()
