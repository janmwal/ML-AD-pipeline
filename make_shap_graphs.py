import argparse
from nilearn import image, plotting, datasets
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import shap
from pathlib import Path


def glass_brain_plot(df, value_col, idx_col, title=None, cmap='Reds', vmax=None, vmin=None, display_mode='lyrz'):

    df = pd.DataFrame({
            'value' : df[value_col],
            'region_idx' : df[idx_col]
        })
    # Load the 3D labeled atlas NIfTI
    label_img = image.load_img('data/label_neuromorphometrics.nii') # old : data/labels_neuromorphics_extra.nii (we don't know the patient) or data/wlabel_sPR06786_AD151295-0012-00001-000176-01_MT.nii (from ferath)
    label_data = label_img.get_fdata()

    # Make an empty stat map
    stat_data = np.zeros_like(label_data, dtype=float)

    # Fill in stat map based on region values
    for region_id, value in zip(df['region_idx'], df['value']):
        stat_data[label_data == region_id] = value

    #stat_data[np.abs(stat_data) < 1e-4] = 0
    #print(np.unique(stat_data, return_counts=True))
    # Create a new NIfTI image
    # Zero-out values outside the MNI brain
    brain_mask = datasets.load_mni152_brain_mask()
    brain_mask_res = image.resample_to_img(
        brain_mask, 
        label_img, 
        interpolation='nearest', 
        copy_header=True,
        force_resample=True)
    mask_data = brain_mask_res.get_fdata().astype(bool)
    stat_data[~mask_data] = 0

    stat_img = image.new_img_like(label_img, stat_data)
    if not vmax:
        return plotting.plot_glass_brain(
        stat_img,
        display_mode=display_mode,  # Show all orthogonal projections
        colorbar=True,
        cmap=cmap,
        threshold=1e-6,
        #vmin=vmin,
        #vmax=vmax,
        title=title,
        plot_abs=False
    )
    else:
        if not vmin:
            vmin=0
        display = plotting.plot_glass_brain(
            stat_img,
            display_mode=display_mode,
            colorbar=True,
            cmap=cmap,
            cbar_tick_format='%i',
            vmin=vmin,
            vmax=vmax,
            threshold=1e-6,
            title=title,
            plot_abs=False
        )
        """
        # Shrink main plot area to leave space for colorbar
        fig = plt.gcf()
        # Manually resize each axes object to free space on the right
        for ax in fig.axes:
            pos = ax.get_position()
            ax.set_position([pos.x0, pos.y0, pos.width * 0.85, pos.height])  # shrink width

        # Create colorbar manually in reserved space
        # Add a new axis for the horizontal colorbar at the bottom
        cax = fig.add_axes([0.35, 0.08, 0.3, 0.02])  # [left, bottom, width, height]

        norm = Normalize(vmin=0, vmax=vmax)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # sm = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
        sm.set_array([])

        cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_ticks(np.arange(0, vmax + 1, 1))"""


        return display 
    
def main():
    parser = argparse.ArgumentParser(description="Predict + SHAP from per-region CSV")
    parser.add_argument(
        "--pred_folder",
        required=False,
        default="output_pred",
        type=str,
        help="Path to prediction folder from run_classification script"
    )
    parser.add_argument(
        "--output_folder",
        required=False,
        type=str,
        default="output_pred",
        help="Directory to write outputs (created as output_pred if missing)"
    )
    parser.add_argument(
        "--top_k",
        required=False,
        type=int,
        default=10,
        help="How many features to show in the waterfall plot (by |SHAP|)."
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_folder)
    out_dir  = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Load the SHAP long CSV (robust to comma/semicolon) ----
    shap_long_path = pred_dir / "shap_original_long.csv"
    if not shap_long_path.exists():
        raise FileNotFoundError(f"Could not find {shap_long_path}. Run the classifier script first.")

    df = pd.read_csv(shap_long_path, sep=None, engine="python")

    # Validate required columns
    required_cols = {"region_name", "region_index", "shap_value", "feature_value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{shap_long_path} is missing required columns: {sorted(missing)}")

    # ---- 2) Glass brain plot (save via nilearn's Display) ----
    gb_disp = glass_brain_plot(
        df=df,
        value_col="shap_value",
        idx_col="region_index",
        title=None,
        display_mode="lyrz",
        cmap="seismic"
    )
    gb_png = out_dir / "shap_glass_brain.png"
    gb_disp.savefig(str(gb_png))
    try:
        gb_disp.close()  # nilearn Display supports close()
    except Exception:
        pass

    # 1: Load the JSON config file args.pred_folder/prediction.json
    meta_path = pred_dir / "prediction.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"prediction.json not found at {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # 2) Load the shap explainer based on the model (to get a matching base value)
    expected_value = meta.get("shap_expected_value_logit")
    if expected_value is None:
        raise ValueError("prediction.json missing expected_value; rerun classification script.")


    # 2: Build a SHAP Explanation directly from the long CSV
    # Keep ALL features; SHAP will aggregate the rest into an "others" bar when max_display < n_features
    # ---- Build Explanation with ALL features so SHAP can add "others" automatically ----
    df_all = df.copy()
    df_all["abs_shap"] = df_all["shap_value"].abs()
    df_all = df_all.sort_values("abs_shap", ascending=False)

    expl = shap.Explanation(
        values=df_all["shap_value"].to_numpy(),
        base_values=expected_value,
        data=None, #df_all["feature_value"].to_numpy(),
        feature_names=df_all["region_name"].tolist()
    )

    # ---- Make the figure taller (scaled by how many features you display) ----
    # Base height 6, plus a bit per feature displayed (capped to avoid monster figures)
    features_shown = min(args.top_k, len(df_all))
    fig_height = min(12, max(6, 4 + 0.22 * features_shown))
    plt.figure(figsize=(8, fig_height))

    # ---- Draw waterfall (top_k individual features + SHAP "others") ----
    # Get current axes
    ax = plt.gca()
    ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=-1)

    shap.plots.waterfall(expl, max_display=args.top_k, show=False)

    # ---- Annotation: single sum of all SHAPs, base, total, probability ----
    sum_shap = float(df_all["shap_value"].sum())
    total = expected_value + sum_shap

    proba_from_meta = meta.get("proba", None)
    if proba_from_meta is not None:
        p_annot = float(proba_from_meta)
        p_note = "probability from model output"
    else:
        p_annot = 1.0 / (1.0 + np.exp(-total))
        p_note = "probability via sigmoid(total)"

    thr = meta.get("threshold_used", None)
    label = meta.get("predicted_label", None)

    lines = [
        f"Base = {expected_value:.3f}",
        f"Σ SHAP(all) = {sum_shap:.3f}",
        f"Total = {total:.3f}",
        f"P(AD) ≈ {p_annot:.3f}  ({p_note})"
    ]
    if thr is not None and label is not None:
        lines.append(f"Threshold = {float(thr):.3f} → Label = {label}")
    elif thr is not None:
        lines.append(f"Threshold = {float(thr):.3f}")

    # ---- Reserve a top band for the annotation & place it there ----
    # Leave ~14% of the figure height free at the top for the box
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig = plt.gcf()
    fig.text(
        0.01, 0.98, "\n".join(lines),
        ha="left", va="top", fontsize=9,
        transform=fig.transFigure,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, linewidth=0.6)
    )

    plt.savefig(out_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()


    # ================== END COMPLETION ==================

    
if __name__ == "__main__":
    main()