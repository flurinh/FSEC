#!/usr/bin/env python3
"""
FSEC Analysis Example Script
This script demonstrates the use of the fsec_utils module for analyzing
fluorescence size exclusion chromatography (FSEC) data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from fsec_utils import (
    load_fsec_data_combined,
    find_peaks_with_prominence,
    Chromatogram,
    analyze_chromatogram
)

def main():
    """Main function to run FSEC analysis."""
    # Step 1: Load the combined FSEC data
    print("Loading FSEC data...")
    combined_df = load_fsec_data_combined(include_filename=True, filename_column='source_file')
    
    if combined_df.empty:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded data with {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")
    print(f"Available columns: {combined_df.columns.tolist()}")
    
    # Step 2: Basic data exploration
    print("\n--- Basic Data Statistics ---")
    print(combined_df.describe())
    
    # Get the signal columns (all except Timepoint and source_file)
    signal_columns = [col for col in combined_df.columns 
                     if col not in ['Timepoint', 'source_file']]
    
    # Step 3: Create an overview plot
    print("\nCreating overview plot...")
    plt.figure(figsize=(12, 6))
    
    for col in signal_columns[:3]:  # Plot first 3 columns for brevity
        plt.plot(combined_df['Timepoint'], combined_df[col], label=col)
    
    plt.xlabel('Timepoint')
    plt.ylabel('Signal')
    plt.title('FSEC Data Overview (First 3 Columns)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fsec_overview.png')
    print("Overview plot saved as 'fsec_overview.png'")
    
    # Step 4: Analyze the first signal column in detail
    first_signal_col = signal_columns[0]
    print(f"\nAnalyzing column: {first_signal_col}")
    
    # Method 1: Using find_peaks_with_prominence
    print("\n--- Method 1: Using find_peaks_with_prominence ---")
    peak_indices, corrected_signal, baseline, normalized_signal = find_peaks_with_prominence(
        combined_df, 
        column_name=first_signal_col,
        prominence=0.02, 
        baseline_window=100,
        plot=True
    )
    plt.savefig('peaks_method1.png')
    print("Peak detection plot saved as 'peaks_method1.png'")
    
    # Method 2: Using Chromatogram class
    print("\n--- Method 2: Using Chromatogram class ---")
    chrom = Chromatogram(combined_df, cols={'time': 'Timepoint', 'signal': first_signal_col})
    
    # Crop to interesting region if needed
    # Determine data range
    time_min, time_max = combined_df['Timepoint'].min(), combined_df['Timepoint'].max()
    # Crop to middle 80% for analysis
    time_range = [time_min + 0.1 * (time_max - time_min), 
                 time_max - 0.1 * (time_max - time_min)]
    
    chrom.crop(time_range)
    chrom.show()
    plt.savefig('chromatogram_raw.png')
    print("Raw chromatogram plot saved as 'chromatogram_raw.png'")
    
    # Fit peaks
    peaks = chrom.fit_peaks(prominence=0.01, width=5, buffer=0.2)
    chrom.show()
    plt.savefig('chromatogram_peaks.png')
    print("Fitted peaks plot saved as 'chromatogram_peaks.png'")
    
    # Assess fit quality
    fit_scores = chrom.assess_fit()
    print("\nPeak fitting quality assessment:")
    print(fit_scores.head())
    
    # Method 3: Using analyze_chromatogram
    print("\n--- Method 3: Using analyze_chromatogram ---")
    results = analyze_chromatogram(combined_df,
                                   column_name=first_signal_col,  # Use the same column as before
                                   prominence=0.05,  # Start here
                                   width=10,
                                   shoulder_curvature=0.01,
                                   plot=True,
                                   debug_shoulders=True)  # Enable debugging

    plt.savefig('peaks_method3.png')
    print("Advanced analysis plot saved as 'peaks_method3.png'")
    
    if results:
        print(f"Detected {len(results['peak_indices'])} main peaks")
        print(f"Detected {len(results['shoulder_indices'])} shoulder peaks")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()