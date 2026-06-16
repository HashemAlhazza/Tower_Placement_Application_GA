import os
import json
import uuid
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ----------------------------
# We import everything from your SAC script.
# That means your script must NOT call main() or resume() at the bottom
# when imported — wrap those calls in:  if __name__ == "__main__":
# We will tell you exactly what to change in the SAC script.
# ----------------------------
from ga_script import (
    load_data,
    run_ga,
    compute_radio_maps,
    repair_close_towers,
    zone_coverage_score,
    crop_data_to_bbox,
    SINR_THRESHOLD_DB,
    CAPEX_BUDGET,
    MAX_TOWERS,
    TOWER_COST_BASE,
)

app = Flask(__name__)
CORS(app)  # allows the browser (running on the same machine) to talk to Flask

# ----------------------------
# Paths — adjust these to where your files actually are
# ----------------------------
POP_PATH = "data/egy_pop_2025_CN_1km_R2025A_UA_v1.tif"
OUTPUT_DIR = "static/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load data and model ONCE at startup — not on every request.
# This is what makes the demo feel instant.
# Loading the model takes ~2 seconds. The rasters take ~3 seconds.
# After that, each "Run" click takes only the rollout time (~5-10 seconds).
# ----------------------------
print("Loading Egypt population raster...")
data = load_data()
print("Ready. Flask server starting...")


# ----------------------------
# Route 1: serve the main page
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------
# Route 2: the core endpoint
# The browser sends a bounding box (lat/lon) + parameters as JSON.
# We crop the rasters, run inference, return tower positions + KPIs.
# ----------------------------
@app.route("/optimize", methods=["POST"])
def optimize():
    body = request.get_json()

    # --- Parse bounding box from the drawn rectangle ---
    # Leaflet's Rectangle gives us: [[south, west], [north, east]]
    south = float(body["south"])
    north = float(body["north"])
    west = float(body["west"])
    east = float(body["east"])
    print(f"DEBUG bbox: south={south}, north={north}, west={west}, east={east}")

    # --- Crop the data to the bounding box ---
    # We pass the bbox into a helper that slices the raster arrays in memory.
    # No new files are created — it's just numpy array slicing.
    cropped_data = crop_data_to_bbox(data, south, north, west, east)

    if cropped_data is None:
        return (
            jsonify({"error": "Selected region is outside the raster coverage area."}),
            400,
        )

    # --- Run GA on the cropped region ---
    best_chrom, _, _, _ = run_ga(cropped_data)
    from ga_script import chrom_to_lonlat, chrom_to_pixel_xy

    final_lonlat = chrom_to_lonlat(
        best_chrom,
        cropped_data["transform"],
        cropped_data["height"],
        cropped_data["width"],
    )
    final_xy = chrom_to_pixel_xy(
        best_chrom, cropped_data["height"], cropped_data["width"]
    )

    if not final_lonlat:
        return (
            jsonify(
                {"error": "Model placed no towers in this region. Try a larger area."}
            ),
            400,
        )

    # --- Run the repair pass to fix any towers that are too close ---
    final_xy, final_lonlat = repair_close_towers(cropped_data, final_xy, final_lonlat)

    # --- Compute final radio maps for KPI calculation ---
    _, interference_after, sinr_after = compute_radio_maps(
        final_lonlat,
        cropped_data["transform"],
        cropped_data["width"],
        cropped_data["height"],
        cropped_data["easting_grid"],
        cropped_data["northing_grid"],
    )

    # --- Calculate KPIs to show on the results page ---
    served_pop = float(cropped_data["pop"][sinr_after >= SINR_THRESHOLD_DB].sum())
    total_pop = float(cropped_data["pop"].sum())
    coverage_pct = 100.0 * served_pop / max(total_pop, 1.0)
    geo_pct = (
        100.0
        * float((sinr_after >= SINR_THRESHOLD_DB).sum())
        / (cropped_data["height"] * cropped_data["width"])
    )
    budget_spent = TOWER_COST_BASE * len(final_xy)
    zone_cov = zone_coverage_score(
        final_xy, cropped_data["height"], cropped_data["width"]
    )
    served_mask = sinr_after >= SINR_THRESHOLD_DB
    mean_sinr = float(np.mean(sinr_after[served_mask])) if served_mask.any() else 0.0
    n_towers = len(final_lonlat)

    # --- Generate the Folium map and save it to static/results/ ---
    # A unique filename per run so multiple runs don't overwrite each other.
    run_id = str(uuid.uuid4())[:8]
    map_path = os.path.join(OUTPUT_DIR, f"map_{run_id}.html")
    generate_folium_map(final_lonlat, cropped_data, map_path)

    # --- Return everything to the browser as JSON ---
    return jsonify(
        {
            "towers": [{"lat": lat, "lon": lon} for lon, lat in final_lonlat],
            "map_url": f"/static/results/map_{run_id}.html",
            "kpis": {
                "n_towers": len(final_lonlat),
                "population_served": f"{served_pop:,.0f}",
                "coverage_pct": f"{coverage_pct:.1f}%",
                "geo_coverage": f"{geo_pct:.1f}%",
                "zone_coverage": f"{zone_cov:.1%}",
                "mean_sinr": f"{mean_sinr:.1f} dB",
                "budget_spent": f"${budget_spent:,.0f}",
                "budget_remaining": f"${CAPEX_BUDGET - budget_spent:,.0f}",
            },
        }
    )


# ----------------------------
# Helper: generate the Folium result map
# Same as your make_map() function but saves to a file path we choose.
# ----------------------------
def generate_folium_map(towers_lonlat, cropped_data, save_path):
    import folium

    center_lon = float(np.mean([p[0] for p in towers_lonlat]))
    center_lat = float(np.mean([p[1] for p in towers_lonlat]))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB positron",
    )

    for i, (lon, lat) in enumerate(towers_lonlat):
        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            popup=f"Tower {i+1} ({lat:.5f}, {lon:.5f})",
            color="blue",
            fill=True,
            fill_opacity=0.9,
        ).add_to(m)

    m.save(save_path)


# ----------------------------
# Serve static files (the generated Folium maps)
# ----------------------------
@app.route("/static/results/<path:filename>")
def serve_result(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    app.run(debug=False, port=5000)
