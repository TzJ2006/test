"""
Command-line interface for the cross-platform benchmarking tool.

Usage:
    python -m benchmark.cli                    # Run all benchmarks
    python -m benchmark.cli --cpu-only         # CPU benchmarks only
    python -m benchmark.cli --gpu-only         # GPU benchmarks only
    python -m benchmark.cli --report           # Generate HTML report
    python -m benchmark.cli --report-only      # Only generate report, don't run benchmarks
"""
import argparse
import sys
from pathlib import Path

from . import detect, cpu, gpu, core, report


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Cross-platform CPU/GPU benchmarking tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark.cli                    # Run all benchmarks (CSV auto-appends)
  python -m benchmark.cli --cpu-only         # CPU benchmarks only
  python -m benchmark.cli --gpu-only         # GPU benchmarks only
  python -m benchmark.cli --report           # Run benchmarks + generate HTML report
  python -m benchmark.cli --report-only      # Only generate report from existing CSV

Workflow for accumulating results:
  1. Run benchmark on different hardware (results append to CSV):
     python -m benchmark.cli --output my_results.csv

  2. Generate/update HTML report (includes ALL historical data):
     python -m benchmark.cli --report-only --input-csv my_results.csv

The HTML report automatically includes all hardware ever benchmarked!
        """
    )

    # Benchmark selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--cpu-only',
        action='store_true',
        help='Run only CPU benchmarks'
    )
    group.add_argument(
        '--gpu-only',
        action='store_true',
        help='Run only GPU benchmarks'
    )
    group.add_argument(
        '--report-only',
        action='store_true',
        help='Only generate HTML report, don\'t run benchmarks'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='benchmark_results.csv',
        help='Output CSV file path (default: benchmark_results.csv)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Run benchmarks without saving to CSV'
    )

    # Benchmark parameters
    parser.add_argument(
        '--matrix-size',
        type=int,
        default=None,
        help='Matrix size for GEMM benchmarks (default: 4096 GPU, 8192/4096 CPU)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=None,
        help='Number of iterations (default: auto)'
    )

    # Display options
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (minimal output)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose mode (detailed output)'
    )

    # Info options
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show system information and exit'
    )

    # Report options
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML report after benchmarks'
    )
    parser.add_argument(
        '--report-output',
        type=str,
        default='benchmark_report.html',
        help='HTML report output path (default: benchmark_report.html)'
    )
    parser.add_argument(
        '--input-csv',
        type=str,
        default='benchmark_results.csv',
        help='Input CSV file for report generation (default: benchmark_results.csv)'
    )

    return parser.parse_args()


def print_header():
    """Print benchmark header."""
    print("\n" + "=" * 60)
    print("Cross-Platform CPU/GPU Benchmarking Tool")
    print("=" * 60)


def print_summary(all_results: list, system_info: dict):
    """Print summary of all benchmark results."""
    print("\n" + "=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)

    for result in all_results:
        if 'error' in result:
            continue

        name = result.get('name', 'Unknown')
        dtype = result.get('dtype', 'N/A')
        flops = result.get('flops_formatted', 'N/A')

        print(f"\n{name}")
        if dtype != 'N/A':
            print(f"  Precision: {dtype}")
        print(f"  Performance: {flops}")

    print("\n" + "=" * 60 + "\n")


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Report-only mode: just generate report from existing CSV
    if args.report_only:
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            print(f"Please run benchmarks first to generate data, or specify a different file with --input-csv")
            return 1

        print(f"Generating HTML report from: {csv_path}")
        try:
            report.generate_report(str(csv_path), args.report_output)
            print("\n✓ Report generation complete!")
            return 0
        except Exception as e:
            print(f"\n✗ Error generating report: {e}")
            import traceback
            if args.verbose:
                traceback.print_exc()
            return 1

    # Get system information
    system_info = detect.get_system_info()

    # Info only mode
    if args.info:
        print_header()
        detect.print_system_info(system_info)
        return 0

    # Print header
    if not args.quiet:
        print_header()
        detect.print_system_info(system_info)

    # Initialize results manager
    if not args.no_save:
        results_manager = core.BenchmarkResults(args.output)

    # Run benchmarks
    all_results = []

    try:
        # CPU benchmarks
        if not args.gpu_only:
            cpu_results = cpu.run_all_cpu_benchmarks()
            all_results.extend(cpu_results)

        # GPU benchmarks
        if not args.cpu_only:
            gpu_results = gpu.run_all_gpu_benchmarks(
                matrix_size=args.matrix_size,
                iterations=args.iterations
            )
            all_results.extend(gpu_results)

        # Print summary
        if not args.quiet:
            print_summary(all_results, system_info)

        # Save results
        if not args.no_save:
            for result in all_results:
                results_manager.add(result, system_info)
            results_manager.save()

        # Generate HTML report if requested
        if args.report and not args.no_save:
            print("\nGenerating HTML report...")
            try:
                report.generate_report(args.output, args.report_output)
            except FileNotFoundError:
                print(f"Warning: CSV file not found, skipping report generation")
            except Exception as e:
                print(f"Warning: Error generating report: {e}")

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\nError during benchmark: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
