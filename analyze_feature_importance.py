#!/usr/bin/env python3
"""
analyze_feature_importance.py

This script analyzes the feature importance data for linear models and
generates a comprehensive report.

Usage:
    python3 analyze_feature_importance.py [feature_importance_dir] [output_report]
"""

import os
import sys
import json
import glob

def extract_base_feature(feature_name):
    """
    Extract the base feature name from the time-dependent feature name format.
    Example: "jitter_t-3" -> "jitter"
    """
    if '_t-' in feature_name:
        return feature_name.split('_t-')[0]
    # For statistics features (mean, std, min, max)
    for suffix in ['_mean', '_std', '_min', '_max']:
        if suffix in feature_name:
            return feature_name.replace(suffix, '')
    # For raw f0, f1, etc. features, we can't determine the base name
    return feature_name

def analyze_feature_importance(feature_importance_dir):
    """
    Analyzes feature importance data from JSON files for linear models.
    
    Returns:
        dict: Dictionary containing aggregated feature importance information
    """
    if not os.path.exists(feature_importance_dir):
        print(f"Feature importance directory not found: {feature_importance_dir}")
        return None
        
    # Initialize results dictionary
    results = {
        'linear_models': [],
        'feature_importance_by_base_feature': {},
        'feature_importance_by_time': {},
        'feature_importance_by_stat': {}
    }
    
    # Find all importance files
    importance_files = glob.glob(os.path.join(feature_importance_dir, "*.importance.json"))
    
    if not importance_files:
        print(f"No feature importance files found in {feature_importance_dir}")
        return None
    
    print(f"Found {len(importance_files)} feature importance files")
    
    # Process each file
    for filepath in importance_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Store basic model information
            model_info = {
                'model_name': data['model_name'],
                'top_features': data['feature_importance'][:10],  # Top 10 features
                'bias': data['bias']
            }
            results['linear_models'].append(model_info)
            
            # Analyze features by base feature type (jitter, throughput, etc.)
            for feature_data in data['feature_importance']:
                feature_name = feature_data['feature']
                weight = abs(feature_data['weight'])  # Use absolute value for importance
                
                # Extract base feature name
                base_feature = extract_base_feature(feature_name)
                
                # Aggregate by base feature
                if base_feature not in results['feature_importance_by_base_feature']:
                    results['feature_importance_by_base_feature'][base_feature] = {
                        'total_importance': 0,
                        'count': 0
                    }
                
                results['feature_importance_by_base_feature'][base_feature]['total_importance'] += weight
                results['feature_importance_by_base_feature'][base_feature]['count'] += 1
                
                # Analyze by time step if applicable
                if '_t-' in feature_name:
                    time_step = int(feature_name.split('_t-')[1])
                    if time_step not in results['feature_importance_by_time']:
                        results['feature_importance_by_time'][time_step] = {
                            'total_importance': 0,
                            'count': 0
                        }
                    results['feature_importance_by_time'][time_step]['total_importance'] += weight
                    results['feature_importance_by_time'][time_step]['count'] += 1
                
                # Analyze by statistical feature type if applicable
                for stat_suffix in ['_mean', '_std', '_min', '_max']:
                    if stat_suffix in feature_name:
                        if stat_suffix not in results['feature_importance_by_stat']:
                            results['feature_importance_by_stat'][stat_suffix] = {
                                'total_importance': 0,
                                'count': 0
                            }
                        results['feature_importance_by_stat'][stat_suffix]['total_importance'] += weight
                        results['feature_importance_by_stat'][stat_suffix]['count'] += 1
                        break
                        
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
    
    # Calculate average importance
    for category in ['feature_importance_by_base_feature', 'feature_importance_by_time', 'feature_importance_by_stat']:
        for key in results[category]:
            if results[category][key]['count'] > 0:
                results[category][key]['avg_importance'] = (
                    results[category][key]['total_importance'] / results[category][key]['count']
                )
    
    return results

def generate_report(analysis_results, output_file):
    """
    Generates a report from the analysis results.
    """
    if not analysis_results or not analysis_results['linear_models']:
        print("No analysis results to report")
        return False
    
    with open(output_file, 'w') as f:
        f.write("FEATURE IMPORTANCE ANALYSIS REPORT\n")
        f.write("===============================\n\n")
        
        # Individual model analysis
        f.write("Linear Model Feature Importance:\n")
        f.write("-------------------------------\n\n")
        
        for model_info in analysis_results['linear_models']:
            f.write(f"Model: {model_info['model_name']}\n")
            f.write("Top 10 Features by Importance (absolute weight):\n")
            
            for i, feature_data in enumerate(model_info['top_features'], 1):
                feature = feature_data['feature']
                weight = feature_data['weight']
                f.write(f"  {i}. {feature}: {weight:.6f}\n")
            
            f.write(f"Bias term: {model_info['bias']:.6f}\n\n")
        
        # Analysis by base feature
        f.write("\nAggregate Feature Importance by Base Feature:\n")
        f.write("-------------------------------------------\n")
        f.write("(Higher values indicate more influential features across models)\n\n")
        
        # Sort by average importance
        sorted_base_features = sorted(
            analysis_results['feature_importance_by_base_feature'].items(),
            key=lambda x: x[1]['avg_importance'],
            reverse=True
        )
        
        for base_feature, stats in sorted_base_features:
            f.write(f"{base_feature}: {stats['avg_importance']:.6f} (across {stats['count']} occurrences)\n")
        
        # Analysis by time step if available
        if analysis_results['feature_importance_by_time']:
            f.write("\nFeature Importance by Time Step:\n")
            f.write("------------------------------\n")
            f.write("(Shows whether recent or older measurements are more important)\n\n")
            
            # Sort by time step (desc order)
            sorted_time_steps = sorted(
                analysis_results['feature_importance_by_time'].items(),
                key=lambda x: x[0],
                reverse=True
            )
            
            for time_step, stats in sorted_time_steps:
                f.write(f"t-{time_step}: {stats['avg_importance']:.6f} (across {stats['count']} features)\n")
        
        # Analysis by statistical feature type if available
        if analysis_results['feature_importance_by_stat']:
            f.write("\nFeature Importance by Statistical Feature Type:\n")
            f.write("-------------------------------------------\n")
            f.write("(Shows which statistical measures are most predictive)\n\n")
            
            # Sort by average importance
            sorted_stats = sorted(
                analysis_results['feature_importance_by_stat'].items(),
                key=lambda x: x[1]['avg_importance'],
                reverse=True
            )
            
            for stat_type, stats in sorted_stats:
                stat_name = stat_type.replace('_', ' ').strip()
                f.write(f"{stat_name}: {stats['avg_importance']:.6f} (across {stats['count']} features)\n")
    
    print(f"Report saved to {output_file}")
    return True

def main():
    # Get directory from command line or use default
    if len(sys.argv) > 1:
        feature_importance_dir = sys.argv[1]
    else:
        feature_importance_dir = os.path.expanduser("~/Impact-xG_prediction_model/forecasting_models_v5/feature_importance")
    
    # Get output file from command line or use default
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = os.path.expanduser("~/Impact-xG_prediction_model/forecasting_models_v5/feature_importance_report.txt")
    
    print(f"Analyzing feature importance data in {feature_importance_dir}")
    
    # Analyze the feature importance data
    analysis_results = analyze_feature_importance(feature_importance_dir)
    
    if analysis_results:
        # Generate the report
        generate_report(analysis_results, output_file)
    else:
        print("No analysis results to report")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())