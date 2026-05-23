"""
Shared evaluation framework for 3-class causal relation extraction.

PRIMARY METRIC: Macro F1 over the 2 Cause-Effect classes ONLY (excluding Other)

SECONDARY METRICS (also reported and saved):
    - Per-class Precision, Recall, F1 for all 3 classes
    - Macro F1 over all 3 classes (including Other)
    - Micro F1 (= accuracy for balanced-class assumptions)
    - Accuracy
    - Confusion matrix
"""

import os
import json
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_ORDER = ['Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)', 'Other']
CAUSAL_LABELS = ['Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)']

# Short names for display / plot axes
SHORT_NAMES = {
    'Cause-Effect(e1,e2)': 'CE(e1,e2)',
    'Cause-Effect(e2,e1)': 'CE(e2,e1)',
    'Other':               'Other',
}


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true,
    y_pred,
    model_name: str,
    dataset_name: str,
    output_dir: str = None,
):
    y_true = list(y_true)
    y_pred = list(y_pred)

    # --- Core metrics ---
    report_dict = classification_report(
        y_true, y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )

    primary_f1 = f1_score(
        y_true, y_pred,
        labels=CAUSAL_LABELS,
        average='macro',
        zero_division=0,
    )

    macro_f1_all3 = f1_score(
        y_true, y_pred,
        labels=LABEL_ORDER,
        average='macro',
        zero_division=0,
    )

    micro_f1 = f1_score(
        y_true, y_pred,
        labels=LABEL_ORDER,
        average='micro',
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)

    metrics = {
        'model':    model_name,
        'dataset':  dataset_name,
        'primary': {
            'macro_f1_causal_only': round(primary_f1, 4),
        },
        'per_class': {
            label: {
                'precision': round(report_dict[label]['precision'], 4),
                'recall':    round(report_dict[label]['recall'],    4),
                'f1':        round(report_dict[label]['f1-score'],  4),
                'support':   int(report_dict[label]['support']),
            }
            for label in LABEL_ORDER
        },
        'overall': {
            'macro_f1_all3': round(macro_f1_all3, 4),
            'micro_f1':      round(micro_f1,      4),
            'accuracy':      round(acc,            4),
        },
        'confusion_matrix': cm,
    }

    _print_report(metrics)

    if output_dir:
        _save_results(metrics, cm, output_dir)

    return metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_report(metrics):
    m = metrics
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {m['model']}  |  dataset: {m['dataset']}")
    print(f"{'='*60}")

    print(f"\n  *** PRIMARY METRIC ***")
    print(f"  Macro F1 (Cause-Effect classes only): "
          f"{m['primary']['macro_f1_causal_only']:.4f}  "
          f"({m['primary']['macro_f1_causal_only']*100:.2f}%)")

    print(f"\n  Per-class results:")
    print(f"  {'Label':<26} {'P':>7} {'R':>7} {'F1':>7} {'Support':>9}")
    print(f"  {'-'*56}")
    for label in LABEL_ORDER:
        pc = m['per_class'][label]
        print(f"  {label:<26} {pc['precision']:>7.4f} {pc['recall']:>7.4f} "
              f"{pc['f1']:>7.4f} {pc['support']:>9d}")

    print(f"\n  Overall:")
    print(f"    Macro F1 (all 3 classes): {m['overall']['macro_f1_all3']:.4f}")
    print(f"    Micro F1:                 {m['overall']['micro_f1']:.4f}")
    print(f"    Accuracy:                 {m['overall']['accuracy']:.4f}")
    print(f"{'='*60}\n")


def _save_results(metrics, cm, output_dir):
    """Saves text report, JSON metrics, and confusion matrix PNG."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Human-readable text report
    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        m = metrics
        f.write(f"EVALUATION REPORT\n")
        f.write(f"Model:   {m['model']}\n")
        f.write(f"Dataset: {m['dataset']}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"PRIMARY METRIC\n")
        f.write(f"  Macro F1 (Cause-Effect only): "
                f"{m['primary']['macro_f1_causal_only']:.4f}\n\n")
        f.write(f"PER-CLASS RESULTS\n")
        f.write(f"{'Label':<26} {'P':>7} {'R':>7} {'F1':>7} {'Support':>9}\n")
        f.write(f"{'-'*56}\n")
        for label in LABEL_ORDER:
            pc = m['per_class'][label]
            f.write(f"{label:<26} {pc['precision']:>7.4f} {pc['recall']:>7.4f} "
                    f"{pc['f1']:>7.4f} {pc['support']:>9d}\n")
        f.write(f"\nOVERALL\n")
        f.write(f"  Macro F1 (all 3): {m['overall']['macro_f1_all3']:.4f}\n")
        f.write(f"  Micro F1:         {m['overall']['micro_f1']:.4f}\n")
        f.write(f"  Accuracy:         {m['overall']['accuracy']:.4f}\n\n")
        f.write(f"CONFUSION MATRIX (rows=true, cols=pred)\n")
        f.write(f"Labels: {LABEL_ORDER}\n")
        f.write(str(cm) + "\n")

    # 2. Metrics in JSON format (useful when comparing different models)
    json_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    json_metrics['confusion_matrix'] = cm.tolist()
    json_path = os.path.join(output_dir, 'metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_metrics, f, indent=2)

    # 3. Confusion matrix PNG (saved but not shown inline)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')   # non-interactive backend for saving

    short = [SHORT_NAMES[l] for l in LABEL_ORDER]
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Greens',
                   vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100
            text = f"{cm[i, j]}\n({pct:.1f}%)"
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, text, ha='center', va='center',
                    color=color, fontsize=9)

    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('True', fontsize=10)
    ax.set_title(
        f'Confusion Matrix — {metrics["model"]} on {metrics["dataset"]}',
        fontsize=11,
    )
    plt.tight_layout()
    png_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\nResults saved to: {output_dir}")
    print(f"  {os.path.basename(report_path)}")
    print(f"  {os.path.basename(json_path)}")
    print(f"  confusion_matrix.png")
