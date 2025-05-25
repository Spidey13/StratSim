from preprocessor import F1DataPreprocessor
from pathlib import Path
import argparse
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process F1 race data")
    parser.add_argument(
        "--input_file",
        default="data/raw/seasons_2023_2024_data.csv",
        help="Path to the combined CSV data file",
    )
    parser.add_argument(
        "--output_file",
        default="data/processed/seasons_2023_2024_processed.csv",
        help="Path to save the processed data",
    )
    parser.add_argument(
        "--categories_dir",
        default="data/categories",
        help="Directory to save category JSON files",
    )

    args = parser.parse_args()

    # Convert relative paths to absolute if needed
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    categories_dir = Path(args.categories_dir)

    # Ensure input file exists
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    # Display configuration
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Categories directory: {categories_dir}")

    # Initialize preprocessor with custom categories directory
    preprocessor = F1DataPreprocessor(categories_dir=str(categories_dir))

    try:
        # Process the data
        preprocessor.process_and_save(str(input_file), str(output_file))

        # Check if files were successfully created
        drivers_json = preprocessor.drivers_json_path
        teams_json = preprocessor.teams_json_path
        events_json = preprocessor.events_json_path

        logger.info("Preprocessing completed successfully!")
        logger.info(f"Processed data saved to: {output_file}")

        if Path(drivers_json).exists():
            logger.info(f"Driver categories saved to: {drivers_json}")
        else:
            logger.warning(f"Driver categories file not created at: {drivers_json}")

        if Path(teams_json).exists():
            logger.info(f"Team categories saved to: {teams_json}")
        else:
            logger.warning(f"Team categories file not created at: {teams_json}")

        if Path(events_json).exists():
            logger.info(f"Event categories saved to: {events_json}")
        else:
            logger.warning(f"Event categories file not created at: {events_json}")

    except Exception as e:
        logger.error(f"Error processing data: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
