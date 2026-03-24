import os
import subprocess
from sklearn.metrics import classification_report, f1_score, accuracy_score


def save_predictions_official_format(ids, predictions, output_filepath):
    """
    Saves model predictions in the format required by the official scorer.

    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    with open(output_filepath, 'w', encoding='utf-8', newline='\n') as f:
        for sent_id, pred in zip(ids, predictions):
            # MUST be tab-separated — the format checker enforces this
            f.write(f'{sent_id}\t{pred}\n')

    print(f"Predictions saved to: {output_filepath}")


def run_format_checker(predictions_filepath, checker_script_path):
    """
    Runs the official Perl format checker on your predictions file.
    This validates your file before scoring — always run this first!

    """
    print("Running official format checker...")

    result = subprocess.run(
        ['perl', checker_script_path, predictions_filepath],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    if 'format is OK' in result.stdout:
        print("✓ Format is valid — ready to score!")
        return True
    else:
        print("✗ Format has errors — fix before scoring!")
        return False


def run_official_scorer(predictions_filepath, key_filepath, scorer_script_path,
                        results_output_path=None):
    """
    Runs the official Perl scorer and captures all output.

    This is the ONLY way to get the official macro F1 score that
    matches what the paper reports.

    Returns
    -------
    str
        The full text output from the scorer
    """
    print("Running official semeval scorer")

    result = subprocess.run(
        ['perl', scorer_script_path, predictions_filepath, key_filepath],
        capture_output=True,
        text=True
    )

    scorer_output = result.stdout
    if result.stderr:
        print("Scorer warnings:", result.stderr)

    print(scorer_output)

    if results_output_path:
        os.makedirs(os.path.dirname(results_output_path), exist_ok=True)
        with open(results_output_path, 'w') as f:
            f.write(scorer_output)
        print(f"\nScorer output saved to: {results_output_path}")

    return scorer_output