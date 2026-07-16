import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Set publication-quality plotting styles globally
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 600,         # Upgraded to 600 DPI for ultra-sharp print quality
    "savefig.bbox": "tight"
})

# Professional, highly-readable academic color palette
COLORS = {
    "peptide": "#E41A1C",   # Strong Red
    "protein": "#377EB8",   # Deep Blue
    "receptor": "#377EB8",  # Deep Blue
    "default": "#4DAF4A",   # Emerald Green
    "pca": "#984EA3"        # Purple
}

# Output folder for plots
OUTPUT_DIR = "analysis_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_xvg(file_path):
    """Parses a GROMACS .xvg file, extracting metadata and numerical data."""
    title = ""
    xlabel = ""
    ylabel = ""
    legends = []
    data_lines = []

    title_re = re.compile(r'@\s+title\s+"(.*)"')
    xlabel_re = re.compile(r'@\s+xaxis\s+label\s+"(.*)"')
    ylabel_re = re.compile(r'@\s+yaxis\s+label\s+"(.*)"')
    legend_re = re.compile(r'@\s+s\d+\s+legend\s+"(.*)"')

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(('@', '#')):
                t_match = title_re.match(line)
                x_match = xlabel_re.match(line)
                y_match = ylabel_re.match(line)
                l_match = legend_re.match(line)
                
                if t_match: title = t_match.group(1)
                elif x_match: xlabel = x_match.group(1)
                elif y_match: ylabel = y_match.group(1)
                elif l_match: legends.append(l_match.group(1))
            else:
                parts = line.split()
                try:
                    data_lines.append([float(x) for x in parts])
                except ValueError:
                    continue

    data = np.array(data_lines)
    return data, title, xlabel, ylabel, legends

def clean_label(label):
    """Converts GROMACS formatting tags into LaTeX-style formatting for Matplotlib."""
    if not label:
        return ""
    label = label.replace(r"\s", "^")
    label = label.replace(r"\S", "")
    label = label.replace(r"\N", "")
    label = label.replace(r"\m", "")
    label = label.replace("nm", "nm")
    label = re.sub(r'\^(\d+)', r'$^{\1}$', label)
    return label

def generate_custom_title(file_name, original_title):
    """Generates a polished title forcing 'Peptide' or 'Protein' context."""
    lower_name = file_name.lower()
    
    # Identify target entity
    if "pep" in lower_name:
        entity = "Peptide"
    elif "prot" in lower_name or "receptor" in lower_name:
        entity = "Protein (Receptor)"
    else:
        entity = "System"

    # Match metric types to build a readable, structured title
    if "rmsd" in lower_name:
        return f"Root Mean Square Deviation (RMSD) — {entity}"
    elif "rmsf" in lower_name:
        return f"Root Mean Square Fluctuation (RMSF) — {entity}"
    elif "gyrate" in lower_name:
        return f"Radius of Gyration ($R_g$) — {entity}"
    elif "sasa" in lower_name:
        return f"Solvent Accessible Surface Area (SASA) — {entity}"
    elif "hb_num" in lower_name or "hbond" in lower_name:
        return "Intermolecular Hydrogen Bonds — Protein-Peptide"
    elif "mindist" in lower_name:
        return "Minimum Contact Distance — Protein-Peptide"
    elif "2dproj" in lower_name:
        return f"PCA 2D Projection — {entity}"
    elif "eigenval" in lower_name:
        return f"PCA Eigenvalues — {entity}"
    
    # Fallback if name is unexpected
    cleaned_fallback = original_title if original_title else file_name.replace(".xvg", "").replace("_", " ").title()
    return f"{cleaned_fallback} ({entity})"

def plot_xvg(file_name):
    """Generates and saves a clean, standalone plot for a given .xvg file."""
    if not os.path.exists(file_name):
        print(f"Skipping {file_name}: File not found.")
        return

    data, title, xlabel, ylabel, legends = parse_xvg(file_name)

    if data.size == 0:
        print(f"Warning: No data found in {file_name}")
        return

    print(f"Processing: {file_name} -> Saving to {OUTPUT_DIR}/")

    # Determine plot color
    lower_name = file_name.lower()
    if "pep" in lower_name:
        color = COLORS["peptide"]
    elif "prot" in lower_name or "receptor" in lower_name:
        color = COLORS["protein"]
    elif "2dproj" in lower_name:
        color = COLORS["pca"]
    else:
        color = COLORS["default"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    x = data[:, 0]
    
    # Handle PCA scatter plot separately
    if "2dproj" in lower_name and data.shape[1] >= 2:
        y = data[:, 1]
        ax.scatter(x, y, c=color, alpha=0.4, s=3, edgecolors='none')
        ax.plot(x, y, color=color, alpha=0.15, linewidth=0.5)
    else:
        # Standard time-series plot
        for i in range(1, data.shape[1]):
            col_label = legends[i-1] if i-1 < len(legends) else f"Set {i}"
            ax.plot(x, data[:, i], color=color, linewidth=1.2, label=col_label, alpha=0.85)
        
        if data.shape[1] > 2:
            ax.legend(frameon=True, facecolor='white', edgecolor='none', fancybox=False)

    # Set polished labels and generated clean titles
    ax.set_xlabel(clean_label(xlabel) if xlabel else "Time (ps)")
    ax.set_ylabel(clean_label(ylabel) if ylabel else "Value")
    
    custom_title = generate_custom_title(file_name, title)
    ax.set_title(custom_title, pad=12)

    # Subtle grid lines
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    
    # Save files to output directory
    output_base = os.path.join(OUTPUT_DIR, file_name.replace(".xvg", ""))
    
    # PNG saved at ultra-crisp 600 DPI
    plt.savefig(f"{output_base}_pub.png", dpi=600)
    # Vector PDF (infinite resolution)
    plt.savefig(f"{output_base}_pub.pdf")
    plt.close()

if __name__ == "__main__":
    xvg_files = [
        "rmsd_peptide.xvg", "rmsd_protein.xvg",
        "rmsf_peptide.xvg", "rmsf_protein.xvg",
        "gyrate_peptide.xvg", "gyrate_protein.xvg",
        "hb_num.xvg", "mindist.xvg",
        "sasa_peptide.xvg", "sasa_protein.xvg",
        "2dproj_pep.xvg", "2dproj_prot.xvg"
    ]

    for f in xvg_files:
        plot_xvg(f)
        
    print(f"\nSuccess! All 600 DPI plots are safely saved inside the '{OUTPUT_DIR}' folder.")