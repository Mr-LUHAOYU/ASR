mfccs=(13 26 39)
deltas=(1 2 3)
datasetName="Ravdess"
for mfcc in "${mfccs[@]}"; do
  for delta in "${deltas[@]}"; do
    python extract_features.py --datasetName $datasetName \
                              --mfcc $mfcc --delta $delta
  done
done
