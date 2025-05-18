epoch=30
batchSize=64
dataset='EmoDB'

python train.py --epoch $epoch --batchSize $batchSize --dataset $dataset --model combine
python translate.py --dataset $dataset --model combine
python plot.py --dataset $dataset --model combine

python train.py --epoch $epoch --batchSize $batchSize --dataset $dataset --model lstm
python translate.py --dataset $dataset --model lstm
python plot.py --dataset $dataset --model lstm

python train.py --epoch $epoch --batchSize $batchSize --dataset $dataset --model mlp
python translate.py --dataset $dataset --model mlp
python plot.py --dataset $dataset --model mlp