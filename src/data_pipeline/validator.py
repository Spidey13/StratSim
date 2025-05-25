import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class F1DataValidator:
    def __init__(self):
        """Initialize the F1 data validator with expected schema and constraints."""
        # Define expected columns and their data types
        self.expected_schema = {
            "Driver": str,
            "LapTime": float,
            "Position": int,
            "LapNumber": int,
            "Compound": str,
            "TyreLife": float,
            "FreshTyre": bool,
            "Team": str,
            "Event": str,
            "Session": str,
            "Year": int,
        }

        # Define valid value ranges
        self.value_ranges = {
            "LapTime": (60.0, 300.0),  # 1-5 minutes in seconds
            "Position": (1, 20),
            "LapNumber": (1, 100),
            "TyreLife": (0, 60),
            "Year": (2022, 2024),
        }

        # Define valid categorical values
        self.valid_categories = {
            "Compound": ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
            "Session": ["FP1", "FP2", "FP3", "Q", "R"],
        }

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate the DataFrame schema against expected schema."""
        issues = {"missing_columns": [], "extra_columns": [], "wrong_type": []}

        # Check for missing columns
        for col, dtype in self.expected_schema.items():
            if col not in df.columns:
                issues["missing_columns"].append(col)
            elif not isinstance(df[col].dtype, type(pd.Series([], dtype=dtype).dtype)):
                issues["wrong_type"].append(
                    f"{col} (expected: {dtype}, got: {df[col].dtype})"
                )

        # Check for extra columns
        for col in df.columns:
            if col not in self.expected_schema:
                issues["extra_columns"].append(col)

        return issues

    def validate_values(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Validate value ranges and categorical values."""
        issues = {
            "out_of_range": {},
            "invalid_categories": {},
            "missing_values": {},
            "duplicates": None,
        }

        # Check numerical ranges
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in df.columns:
                # Attempt to convert column to numeric, coercing errors to NaN
                numeric_col = pd.to_numeric(df[col], errors="coerce")

                # Perform comparison only on valid numeric values
                out_of_range_mask = numeric_col.notna() & (
                    (numeric_col < min_val) | (numeric_col > max_val)
                )
                out_of_range_count = out_of_range_mask.sum()

                if out_of_range_count > 0:
                    # Report original column min/max if possible, otherwise report numeric
                    try:
                        min_found = float(df[col][numeric_col.notna()].min())
                        max_found = float(df[col][numeric_col.notna()].max())
                    except:
                        min_found = float(
                            numeric_col.min()
                        )  # Use coerced numeric if original fails
                        max_found = float(numeric_col.max())

                    issues["out_of_range"][col] = {
                        "count": int(out_of_range_count),
                        "min_found": min_found,
                        "max_found": max_found,
                        "expected_range": [min_val, max_val],
                    }

        # Check categorical values
        for col, valid_values in self.valid_categories.items():
            if col in df.columns:
                invalid_values = df[~df[col].isin(valid_values)][col].unique()
                if len(invalid_values) > 0:
                    issues["invalid_categories"][col] = {
                        "invalid_values": list(invalid_values),
                        "valid_values": valid_values,
                    }

        # Check missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues["missing_values"][col] = {
                    "count": int(missing_count),
                    "percentage": float((missing_count / len(df)) * 100),
                }

        # Check duplicates
        duplicates = df.duplicated().sum()
        issues["duplicates"] = int(duplicates)

        return issues

    def validate_relationships(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate logical relationships between columns."""
        issues = []

        # Lap time should be consistent with sector times if available (use _s columns)
        lap_time_col_s = "LapTime_s"
        sector_cols_s = ["Sector1Time_s", "Sector2Time_s", "Sector3Time_s"]
        if lap_time_col_s in df.columns and all(
            col in df.columns for col in sector_cols_s
        ):
            try:
                # Ensure columns are numeric, coercing errors
                lap_time_s = pd.to_numeric(df[lap_time_col_s], errors="coerce")
                sector_sum_s = (
                    pd.to_numeric(df[sector_cols_s[0]], errors="coerce")
                    + pd.to_numeric(df[sector_cols_s[1]], errors="coerce")
                    + pd.to_numeric(df[sector_cols_s[2]], errors="coerce")
                )

                # Perform comparison only on valid numeric values
                valid_comparison = lap_time_s.notna() & sector_sum_s.notna()
                difference = (lap_time_s - sector_sum_s).abs()
                inconsistent_mask = valid_comparison & (difference > 0.1)
                inconsistent_count = inconsistent_mask.sum()

                if inconsistent_count > 0:
                    issues.append(
                        f"Found {inconsistent_count} laps with inconsistent LapTime_s vs Sector*_s sum (difference > 0.1s)"
                    )
            except Exception as e:
                issues.append(
                    f"Error during lap/sector time consistency check: {str(e)}"
                )

        # Position should be unique within each lap
        if all(
            col in df.columns for col in ["Position", "LapNumber", "Event", "Session"]
        ):
            position_duplicates = df.groupby(["Event", "Session", "LapNumber"])[
                "Position"
            ].nunique()
            invalid_positions = position_duplicates[
                position_duplicates > df["Position"].max()
            ]
            if len(invalid_positions) > 0:
                issues.append(
                    f"Found {len(invalid_positions)} laps with duplicate positions"
                )

        return issues

    def generate_validation_report(
        self, df: pd.DataFrame, output_path: Optional[str] = None
    ) -> str:
        """Generate a comprehensive validation report."""
        # Perform all validations
        schema_issues = self.validate_schema(df)
        value_issues = self.validate_values(df)
        relationship_issues = self.validate_relationships(df)

        # Generate report
        report = []
        report.append("F1 Data Validation Report")
        report.append("=" * 50)

        # Basic information
        report.append(f"\nDataset Overview:")
        report.append(f"Total rows: {len(df)}")
        report.append(f"Total columns: {len(df.columns)}")

        # Schema issues
        report.append("\nSchema Validation:")
        if any(schema_issues.values()):
            for issue_type, issues in schema_issues.items():
                if issues:
                    report.append(f"\n{issue_type.replace('_', ' ').title()}:")
                    for issue in issues:
                        report.append(f"  - {issue}")
        else:
            report.append("  No schema issues found")

        # Value issues
        report.append("\nValue Validation:")
        if any(value_issues.values()):
            for issue_type, issues in value_issues.items():
                if issues:
                    report.append(f"\n{issue_type.replace('_', ' ').title()}:")
                    if isinstance(issues, dict):
                        for col, details in issues.items():
                            report.append(f"  {col}:")
                            for k, v in details.items():
                                report.append(f"    - {k}: {v}")
                    else:
                        report.append(f"  {issues}")
        else:
            report.append("  No value issues found")

        # Relationship issues
        report.append("\nRelationship Validation:")
        if relationship_issues:
            for issue in relationship_issues:
                report.append(f"  - {issue}")
        else:
            report.append("  No relationship issues found")

        report_text = "\n".join(report)

        # Save report if output path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)

        return report_text

    def validate_dataset(
        self, data_path: str, output_path: Optional[str] = None
    ) -> str:
        """Validate a dataset and generate a report."""
        try:
            # Read the dataset
            df = pd.read_csv(data_path)

            # Generate validation report
            return self.generate_validation_report(df, output_path)

        except Exception as e:
            error_msg = f"Error validating dataset: {str(e)}"
            logger.error(error_msg)
            return error_msg
