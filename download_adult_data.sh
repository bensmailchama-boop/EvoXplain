#!/bin/bash
# Download Adult Income dataset from UCI ML Repository
# Run this from your HPC_EvoXplain directory

set -euo pipefail

mkdir -p data
cd data

echo "Downloading Adult dataset from UCI ML Repository..."

# Download training data
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O adult_train.csv

# Download test data  
wget -q https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -O adult_test_raw.csv

# Remove the first line from test file (it has a spurious header line with a period)
tail -n +2 adult_test_raw.csv > adult_test.csv
rm adult_test_raw.csv

# Add column headers and combine into single file
echo "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income" > adult.csv

# Append training data (strip leading/trailing whitespace issues)
cat adult_train.csv >> adult.csv

# Append test data
cat adult_test.csv >> adult.csv

# Clean up intermediate files
rm adult_train.csv adult_test.csv

# Show stats
echo ""
echo "=== Download complete ==="
echo "File: data/adult.csv"
echo "Lines: $(wc -l < adult.csv)"
echo "Preview:"
head -3 adult.csv

echo ""
echo "Done! Dataset ready at data/adult.csv"
