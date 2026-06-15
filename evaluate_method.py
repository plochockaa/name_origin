"""
Evaluation Script for Name Country Inference

This script evaluates the accuracy of the name country inference system by:
1. Splitting lookup tables into train/test sets
2. Using the training set for lookups
3. Testing predictions against ground truth labels
4. Generating comprehensive evaluation metrics
"""

import pandas as pd
import os
import json
import random as _random
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
import argparse
from datetime import datetime
from utils.cleaning import normalize_name, romanize_if_needed
from main import llm_call

load_dotenv()


def get_ground_truth_countries(test_row):
    """Extract ground truth countries from test row."""
    countries_value = test_row['countries']
    if isinstance(countries_value, str):
        return json.loads(countries_value.replace("'", '"'))
    return countries_value if isinstance(countries_value, list) else []


def _build_country_pool(tables):
    """Collect all unique country names across both lookup tables."""
    pool = set()
    for df in tables:
        for _, row in df.iterrows():
            pool.update(get_ground_truth_countries(row))
    return sorted(pool)


def evaluate_single_name(name, name_type, test_row, country_pool, n_distractors=5, rng=None):
    """
    Evaluate a single name from the test set.

    For each test name, distractor countries (not associated with the name)
    are sampled from the global country pool and mixed with the true
    candidates. This forces the LLM to discriminate rather than simply
    picking from a list that already contains only correct answers.

    Args:
        name: The normalized name
        name_type: 'forename' or 'surname'
        test_row: Row from test set containing ground truth countries
        country_pool: Sorted list of all countries in the dataset
        n_distractors: Number of distractor countries to add per evaluation
        rng: random.Random instance for reproducibility

    Returns:
        dict: Evaluation result for this name
    """
    if rng is None:
        rng = _random.Random(42)

    true_countries = get_ground_truth_countries(test_row)
    originally_single = len(true_countries) == 1

    if not true_countries:
        return {
            'name': name,
            'type': name_type,
            'true_countries': [],
            'candidate_countries': [],
            'predicted_country': None,
            'probability': None,
            'all_probabilities': {},
            'correct': False,
            'status': 'no_candidates'
        }

    # Add distractor countries so the LLM must discriminate, not just pick
    distractor_pool = [c for c in country_pool if c not in true_countries]
    n = min(n_distractors, len(distractor_pool))
    distractors = rng.sample(distractor_pool, n)
    candidate_countries = list(true_countries) + distractors
    rng.shuffle(candidate_countries)

    if name_type == 'forename':
        probabilities = llm_call(name, "", candidate_countries)
    else:
        probabilities = llm_call("", name, candidate_countries)

    if not probabilities:
        return {
            'name': name,
            'type': name_type,
            'true_countries': true_countries,
            'candidate_countries': candidate_countries,
            'predicted_country': None,
            'probability': None,
            'all_probabilities': {},
            'correct': False,
            'status': 'llm_prediction_failed'
        }

    predicted_country = max(probabilities.items(), key=lambda x: x[1])[0]
    is_correct = predicted_country in true_countries

    return {
        'name': name,
        'type': name_type,
        'true_countries': true_countries,
        'candidate_countries': candidate_countries,
        'predicted_country': predicted_country,
        'probability': probabilities.get(predicted_country, 0),
        'all_probabilities': probabilities,
        'correct': is_correct,
        'status': 'single_true_country' if originally_single else 'success'
    }


def run_evaluation(forename_table, surname_table, test_size=0.2,
                   max_test_samples=None, random_state=42, n_distractors=5):
    """
    Run full evaluation on both forename and surname tables.
    
    Args:
        forename_table: DataFrame with forename data
        surname_table: DataFrame with surname data
        test_size: Proportion for test set (default 0.2)
        max_test_samples: Maximum number of test samples to evaluate (None for all)
        random_state: Random seed for reproducibility
    
    Returns:
        dict: Complete evaluation results
    """
    print("\n" + "="*70)
    print("NAME COUNTRY INFERENCE EVALUATION")
    print("="*70)

    # Deduplicate on name so the split is on unique names, not rows
    forename_table = forename_table.dropna(subset=["name"]).drop_duplicates(subset="name")
    surname_table = surname_table.dropna(subset=["name"]).drop_duplicates(subset="name")

    # Build country pool before split so distractors can come from the full set
    country_pool = _build_country_pool([forename_table, surname_table])

    # Split data
    forename_train, forename_test = train_test_split(
        forename_table, test_size=test_size, random_state=random_state
    )
    surname_train, surname_test = train_test_split(
        surname_table, test_size=test_size, random_state=random_state
    )

    rng = _random.Random(random_state)

    print(f"\nDataset Split (test_size={test_size}):")
    print(f"  Forenames: {len(forename_train)} train, {len(forename_test)} test")
    print(f"  Surnames:  {len(surname_train)} train, {len(surname_test)} test")
    print(f"  Country pool: {len(country_pool)} unique countries")
    print(f"  Distractors per name: {n_distractors}")
    
    # Limit test samples if specified
    if max_test_samples:
        forename_test = forename_test.head(max_test_samples)
        surname_test = surname_test.head(max_test_samples)
        print(f"\nLimiting evaluation to {max_test_samples} samples per name type")
    
    results = {
        'forename': {'results': [], 'stats': {}},
        'surname': {'results': [], 'stats': {}}
    }
    
    # Evaluate forenames
    print("\n" + "-"*70)
    print("EVALUATING FORENAMES")
    print("-"*70)
    
    for idx, row in forename_test.iterrows():
        result = evaluate_single_name(row['name'], 'forename', row, country_pool, n_distractors, rng)
        results['forename']['results'].append(result)
        
        if (len(results['forename']['results']) % 10) == 0:
            correct = sum(1 for r in results['forename']['results'] if r['correct'])
            total = sum(1 for r in results['forename']['results']
                       if r['status'] in ['success', 'single_true_country'])
            if total > 0:
                print(f"  Processed {len(results['forename']['results'])} forenames... (Accuracy: {correct/total*100:.1f}%)")
    
    # Evaluate surnames
    print("\n" + "-"*70)
    print("EVALUATING SURNAMES")
    print("-"*70)
    
    for idx, row in surname_test.iterrows():
        result = evaluate_single_name(row['name'], 'surname', row, country_pool, n_distractors, rng)
        results['surname']['results'].append(result)
        
        if (len(results['surname']['results']) % 10) == 0:
            correct = sum(1 for r in results['surname']['results'] if r['correct'])
            total = sum(1 for r in results['surname']['results']
                       if r['status'] in ['success', 'single_true_country'])
            if total > 0:
                print(f"  Processed {len(results['surname']['results'])} surnames... (Accuracy: {correct/total*100:.1f}%)")
    
    # Calculate statistics
    for name_type in ['forename', 'surname']:
        res = results[name_type]['results']
        
        total = len(res)
        successful = sum(1 for r in res if r['status'] in ['success', 'single_true_country'])
        correct = sum(1 for r in res if r['correct'])
        no_candidates = sum(1 for r in res if r['status'] == 'no_candidates')
        single_true = sum(1 for r in res if r['status'] == 'single_true_country')
        llm_failed = sum(1 for r in res if r['status'] == 'llm_prediction_failed')

        multi_true_cases = successful - single_true
        multi_true_correct = sum(1 for r in res if r['correct'] and r['status'] == 'success')

        results[name_type]['stats'] = {
            'total': total,
            'successful_predictions': successful,
            'correct': correct,
            'incorrect': successful - correct,
            'no_candidates': no_candidates,
            'single_true_country': single_true,
            'llm_prediction_failed': llm_failed,
            'multi_true_cases': multi_true_cases,
            'multi_true_correct': multi_true_correct,
            'accuracy_all': (correct / successful * 100) if successful > 0 else 0,
            'accuracy_multi_true': (multi_true_correct / multi_true_cases * 100) if multi_true_cases > 0 else 0,
            'coverage': (successful / total * 100) if total > 0 else 0
        }
    
    return results


def print_evaluation_summary(results):
    """Print a well-formatted evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print("\nNote: Each test name is evaluated with its true countries plus random")
    print("distractor countries. The LLM must pick a true country from that mix.")

    for name_type in ['forename', 'surname']:
        stats = results[name_type]['stats']

        print(f"\n{name_type.upper()}S:")
        print("-"*70)
        print(f"  Total Test Set:              {stats['total']}")
        print(f"  Successful Predictions:      {stats['successful_predictions']} ({stats['coverage']:.1f}% coverage)")
        print(f"  - Single true country:        {stats['single_true_country']}")
        print(f"  - Multiple true countries:    {stats['multi_true_cases']}")
        print(f"\n  Accuracy (all):              {stats['correct']}/{stats['successful_predictions']} ({stats['accuracy_all']:.2f}%)")
        print(f"  Accuracy (multi-true):       {stats['multi_true_correct']}/{stats['multi_true_cases']} ({stats['accuracy_multi_true']:.2f}%)")
        print(f"\n  No Candidates:               {stats['no_candidates']}")
        print(f"  LLM Prediction Failed:       {stats['llm_prediction_failed']}")
    
    # Overall statistics
    total_all = sum(results[t]['stats']['total'] for t in ['forename', 'surname'])
    successful_all = sum(results[t]['stats']['successful_predictions'] for t in ['forename', 'surname'])
    correct_all = sum(results[t]['stats']['correct'] for t in ['forename', 'surname'])
    multi_all = sum(results[t]['stats']['multi_true_cases'] for t in ['forename', 'surname'])
    multi_correct_all = sum(results[t]['stats']['multi_true_correct'] for t in ['forename', 'surname'])
    
    overall_accuracy = (correct_all / successful_all * 100) if successful_all > 0 else 0
    overall_accuracy_multi = (multi_correct_all / multi_all * 100) if multi_all > 0 else 0
    overall_coverage = (successful_all / total_all * 100) if total_all > 0 else 0

    print("\n" + "="*70)
    print(f"OVERALL STATISTICS:")
    print(f"  Total Test Set:              {total_all}")
    print(f"  Coverage:                    {successful_all}/{total_all} ({overall_coverage:.1f}%)")
    print(f"  Accuracy (all):              {correct_all}/{successful_all} ({overall_accuracy:.2f}%)")
    print(f"  Accuracy (multi-true):       {multi_correct_all}/{multi_all} ({overall_accuracy_multi:.2f}%)")
    print("="*70 + "\n")


def save_detailed_results(results, output_file='evaluation_results.csv'):
    """Save detailed results to CSV."""
    all_results = []
    
    for name_type in ['forename', 'surname']:
        for r in results[name_type]['results']:
            row = {
                'name': r['name'],
                'type': r['type'],
                'true_countries': ', '.join(r['true_countries']) if r['true_countries'] else '',
                'candidate_countries': ', '.join(r['candidate_countries']) if r['candidate_countries'] else '',
                'predicted_country': r['predicted_country'] or '',
                'probability': r['probability'] if r['probability'] is not None else 0,
                'correct': r['correct'],
                'status': r['status']
            }
            all_results.append(row)
    
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")


def save_summary_report(results, output_file='evaluation_summary.txt'):
    """Save summary report to text file."""
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NAME COUNTRY INFERENCE EVALUATION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write("Note: Each test name is evaluated with its true countries plus random\n")
        f.write("distractor countries. The LLM must pick a true country from that mix.\n\n")

        for name_type in ['forename', 'surname']:
            stats = results[name_type]['stats']

            f.write(f"{name_type.upper()}S:\n")
            f.write("-"*70 + "\n")
            f.write(f"  Total Test Set:              {stats['total']}\n")
            f.write(f"  Successful Predictions:      {stats['successful_predictions']} ({stats['coverage']:.1f}% coverage)\n")
            f.write(f"  - Single true country:        {stats['single_true_country']}\n")
            f.write(f"  - Multiple true countries:    {stats['multi_true_cases']}\n")
            f.write(f"\n  Accuracy (all):              {stats['correct']}/{stats['successful_predictions']} ({stats['accuracy_all']:.2f}%)\n")
            f.write(f"  Accuracy (multi-true):       {stats['multi_true_correct']}/{stats['multi_true_cases']} ({stats['accuracy_multi_true']:.2f}%)\n")
            f.write(f"\n  No Candidates:               {stats['no_candidates']}\n")
            f.write(f"  LLM Prediction Failed:       {stats['llm_prediction_failed']}\n\n")

        total_all = sum(results[t]['stats']['total'] for t in ['forename', 'surname'])
        successful_all = sum(results[t]['stats']['successful_predictions'] for t in ['forename', 'surname'])
        correct_all = sum(results[t]['stats']['correct'] for t in ['forename', 'surname'])
        multi_all = sum(results[t]['stats']['multi_true_cases'] for t in ['forename', 'surname'])
        multi_correct_all = sum(results[t]['stats']['multi_true_correct'] for t in ['forename', 'surname'])

        overall_accuracy = (correct_all / successful_all * 100) if successful_all > 0 else 0
        overall_accuracy_multi = (multi_correct_all / multi_all * 100) if multi_all > 0 else 0
        overall_coverage = (successful_all / total_all * 100) if total_all > 0 else 0

        f.write("="*70 + "\n")
        f.write(f"OVERALL STATISTICS:\n")
        f.write(f"  Total Test Set:              {total_all}\n")
        f.write(f"  Coverage:                    {successful_all}/{total_all} ({overall_coverage:.1f}%)\n")
        f.write(f"  Accuracy (all):              {correct_all}/{successful_all} ({overall_accuracy:.2f}%)\n")
        f.write(f"  Accuracy (multi-true):       {multi_correct_all}/{multi_all} ({overall_accuracy_multi:.2f}%)\n")
        f.write("="*70 + "\n")
    
    print(f"Summary report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate name country inference accuracy'
    )
    parser.add_argument(
        '--forename-table',
        default='data/final_forenames.csv',
        help='Path to forename lookup table CSV'
    )
    parser.add_argument(
        '--surname-table',
        default='data/final_surnames.csv',
        help='Path to surname lookup table CSV'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data to use for testing (default: 0.2)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of test samples per name type (default: all)'
    )
    parser.add_argument(
        '--output-csv',
        default='evaluation_results.csv',
        help='Output file for detailed results'
    )
    parser.add_argument(
        '--output-summary',
        default='evaluation_summary.txt',
        help='Output file for summary report'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--n-distractors',
        type=int,
        default=5,
        help='Number of distractor countries to add per evaluation (default: 5)'
    )

    args = parser.parse_args()

    # Load data
    print("Loading lookup tables...")
    forename_table = pd.read_csv(args.forename_table)
    surname_table = pd.read_csv(args.surname_table)

    # Run evaluation
    results = run_evaluation(
        forename_table=forename_table,
        surname_table=surname_table,
        test_size=args.test_size,
        max_test_samples=args.max_samples,
        random_state=args.random_seed,
        n_distractors=args.n_distractors,
    )
    
    # Print summary
    print_evaluation_summary(results)
    
    # Save results
    save_detailed_results(results, args.output_csv)
    save_summary_report(results, args.output_summary)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()