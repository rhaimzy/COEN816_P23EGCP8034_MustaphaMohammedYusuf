from dataset_generation import run_dataset
from analysis import run_analysis
from filters import run_filters
from restoration import run_restoration
from segmentation import run_segmentation
from integration import run_integration

if __name__ == "__main__":

    print("Running full COEN816 pipeline...")

    seed, S1, S2, S3 = run_dataset()

    run_analysis()
    run_filters(seed, S1)
    run_restoration(S1)
    run_segmentation()
    run_integration(S2, S3)
