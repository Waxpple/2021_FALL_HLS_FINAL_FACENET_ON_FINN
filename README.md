# 2021_FALL_HLS_FINAL_FACENET_ON_FINN
2021_FALL_HLS_FINAL_FACENET_ON_FINN

# Fix
1. Fix last layer return_quant_tensor=false
2. Fix requirements "==" on line 6

# New
1. Add the distance matrix

# Command on training

```
python train_triplet_loss.py -d "/home/jovyan/vggface2_224" --lfw "/home/jovyan/lfw_224" --num_workers 0 --batch_size 64 --model_architecture sqvgg_2bits --epochs 25 --training_dataset_csv_path datasets/vggface2_full.csv
```
