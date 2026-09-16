"""Measure complete-game throughput without diagnostic profiling."""

from profile_gumbel_pipeline import main


if __name__ == "__main__":
    main(diagnostics=False)
