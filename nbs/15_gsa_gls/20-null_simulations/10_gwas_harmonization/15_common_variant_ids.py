import pickle
import csv
from pathlib import Path
import argparse

import pandas as pd


def run():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--input-gwas-file",
		required=True,
		type=str,
	)
	parser.add_argument(
		"--common-variant-ids-file",
		required=True,
		type=str,
	)
	parser.add_argument(
		"--output-dir",
		required=True,
		type=str,
	)
	
	args = parser.parse_args()

	print(f"Common variants file: {args.common_variant_ids_file}")
	with open(args.common_variant_ids_file, "rb") as handle:
		common_variants = pickle.load(handle)

	# read gwas and filter variants
	input_file = Path(args.input_gwas_file).resolve()

	print(f"Reading GWAS file {input_file}")
	gwas_data = pd.read_table(input_file, dtype=str)

	print(f"Filtering variants: {len(common_variants)}")
	gwas_data = gwas_data[gwas_data["panel_variant_id"].isin(common_variants)]
	assert gwas_data.shape[0] == len(common_variants)

	# save
	output_dir = Path(args.output_dir).resolve()
	output_dir.mkdir(exist_ok=True, parents=True)
	print(f"Output dir: {output_dir}")

	output_file = output_dir / input_file.name
	print(f"Saving output file: {output_file}")
	gwas_data.to_csv(output_file, sep="\t", na_rep="NA", quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
	run()

