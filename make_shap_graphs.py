import argparse
from nilearn import image, plotting, datasets
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import shap
from pathlib import Path
import joblib
from typing import Dict, Optional


def _load_space_to_index_map() -> Dict[str, int]:
    rename_path = Path("data/roi_rename_map.csv")
    neuro_path = Path("data/neuromorphometrics.csv")
    if not rename_path.exists() or not neuro_path.exists():
        return {}

    rename_df = pd.read_csv(rename_path)
    space_to_nospace = {row["space_name"]: row["nospace_name"] for _, row in rename_df.iterrows()}
    nospace_to_space = {v: k for k, v in space_to_nospace.items()}

    neuro_df = pd.read_csv(neuro_path)
    mapping: Dict[str, int] = {}

    def _record(nospace: Optional[str], idx_val: Optional[float]) -> None:
        if not nospace or nospace not in nospace_to_space or pd.isna(idx_val):
            return
        try:
            mapping[nospace_to_space[nospace]] = int(idx_val)
        except (ValueError, TypeError):
            pass

    for _, row in neuro_df.iterrows():
        left_full = row.get("left_full_name")
        right_full = row.get("right_full_name")
        mid_full = row.get("midline_full_name")

        if isinstance(left_full, str):
            nospace = left_full.split("_")[0]
            _record(nospace, row.get("left_id"))
        if isinstance(right_full, str):
            nospace = right_full.split("_")[0]
            _record(nospace, row.get("right_id"))
        if isinstance(mid_full, str):
            nospace = mid_full.split("_")[0]
            _record(nospace, row.get("midline_id"))

    return mapping

SPACE_TO_INDEX_MAP = _load_space_to_index_map()


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
    parser = argparse.ArgumentParser(description="Generate SHAP graphs from prediction outputs")
    parser.add_argument(
        "--pred_folder",
        required=False,
        default="output_pred",
        type=str,
        help=(
            "Path to either a single prediction folder (containing shap_original_long.csv) "
            "or a directory of multiple prediction folders."
        ),
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

    if (pred_dir / "shap_original_long.csv").exists():
        target_dirs = [pred_dir]
    else:
        target_dirs = sorted(
            p for p in pred_dir.glob("*/")
            if (p / "shap_original_long.csv").exists()
        )
        if not target_dirs:
            raise FileNotFoundError(
                f"No prediction folders with shap_original_long.csv found under {pred_dir}"
            )

    for folder in target_dirs:
        shap_long_path = folder / "shap_original_long.csv"
        print(f"Processing {folder}")

        out_dir = folder
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(shap_long_path, sep=None, engine="python")

        if "region_index" in df.columns and SPACE_TO_INDEX_MAP:
            mapped = df["region_name"].map(SPACE_TO_INDEX_MAP)
            df["region_index"] = df["region_index"].fillna(mapped)
            if df["region_index"].isna().all():
                df["region_index"] = mapped
        elif SPACE_TO_INDEX_MAP:
            df["region_index"] = df["region_name"].map(SPACE_TO_INDEX_MAP)

        missing_regions = []
        if df["region_index"].isna().any():
            missing_regions = df.loc[df["region_index"].isna(), "region_name"].tolist()
            if missing_regions:
                print(
                    f"[WARN] Skipping regions without atlas indices in {folder}: {missing_regions[:5]}"
                )

        df_glass = df.dropna(subset=["region_index"]).copy()
        if not df_glass.empty:
            df_glass["region_index"] = df_glass["region_index"].astype(int)
        else:
            print(f"[WARN] No regions with valid indices for glass brain in {folder}.")

        # Validate required columns
        required_cols = {"region_name", "region_index", "shap_value", "feature_value"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{shap_long_path} is missing required columns: {sorted(missing)}")

        # ---- 2) Glass brain plot (save via nilearn's Display) ----
        if not df_glass.empty:
            gb_disp = glass_brain_plot(
                df=df_glass,
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

        meta_path = folder / "prediction.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"prediction.json not found at {meta_path}")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        expected_value = meta.get("shap_expected_value")
        if expected_value is None:
            raise ValueError("prediction.json missing expected_value; rerun classification script.")

        df_all = df.copy()
        df_all["abs_shap"] = df_all["shap_value"].abs()
        df_all = df_all.sort_values("abs_shap", ascending=False)

        expl = shap.Explanation(
            values=df_all["shap_value"].to_numpy(),
            base_values=expected_value,
            data=None,
            feature_names=df_all["region_name"].tolist()
        )

        features_shown = min(args.top_k, len(df_all))
        fig_height = min(12, max(6, 4 + 0.22 * features_shown))
        plt.figure(figsize=(8, fig_height))

        ax = plt.gca()

        shap.plots.waterfall(expl, max_display=args.top_k, show=False)

        shap_mode = meta.get("shap_model_output", "raw")
        is_logit = shap_mode == "log_loss"

        threshold_prob = meta.get("threshold_used", None)
        threshold_position: Optional[float] = None
        if threshold_prob is not None:
            try:
                threshold_prob = float(threshold_prob)
                if is_logit and 0 < threshold_prob < 1:
                    threshold_position = float(np.log(threshold_prob / (1.0 - threshold_prob)))
                elif not is_logit:
                    threshold_position = threshold_prob
            except (TypeError, ValueError):
                threshold_prob = None

        xmin, xmax = ax.get_xlim()
        if threshold_position is not None:
            left = min(xmin, threshold_position)
            right = max(xmax, threshold_position)
            ax.axvspan(left, threshold_position, color="#1f77b4", alpha=0.08, zorder=-2)
            ax.axvspan(threshold_position, right, color="#d62728", alpha=0.08, zorder=-2)
            ax.axvline(threshold_position, color="black", linestyle="--", linewidth=1.2, zorder=-1)
            ax.set_xlim(left, right)
        else:
            ax.axvline(0, color="black", linestyle="--", linewidth=1, zorder=-1)
            ax.set_xlim(xmin, xmax)

        sum_shap = float(df_all["shap_value"].sum())
        total = expected_value + sum_shap

        proba_from_meta = meta.get("proba", None)
        try:
            proba_from_meta = None if proba_from_meta is None else float(proba_from_meta)
        except (TypeError, ValueError):
            proba_from_meta = None

        if is_logit:
            total_prob = 1.0 / (1.0 + np.exp(-total))
            lines = [
                f"Base (logit) = {expected_value:.3f}",
                f"Σ SHAP (logit) = {sum_shap:.3f}",
                f"Total logit = {total:.3f}",
                f"P(AD) ≈ {total_prob:.3f} (sigmoid(total))"
            ]
            if proba_from_meta is not None:
                lines.append(f"Model probability = {proba_from_meta:.3f}")
        else:
            total_prob = total
            lines = [
                f"Base (prob) = {expected_value:.3f}",
                f"Σ SHAP (prob) = {sum_shap:.3f}",
                f"Total prob = {total_prob:.3f}"
            ]
            if proba_from_meta is not None:
                lines.append(f"Model probability = {proba_from_meta:.3f}")
            else:
                lines.append(f"P(AD) ≈ {total_prob:.3f}")

        if threshold_prob is not None:
            label = meta.get("predicted_label", None)
            if label is not None:
                lines.append(f"Threshold = {threshold_prob:.3f} → Label = {label}")
            else:
                lines.append(f"Threshold = {threshold_prob:.3f}")

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
