# MNIST Without collective communication, Adam(lr=0.01)

## ResNet18-2080ti-128batch_size
- Step200: 100.71 data/sec
- Step400: 110 data/sec
- Epoch 0 in 543.88 sec, test time 13.77 sec, test accuracy:0.86263
- Step600: 113.32 data/sec
- Step800: 116.14 data/sec
- Epoch 1 in 478.52 sec, test time 8.71 sec, test accuracy:0.9186
- Step1000: 117.81 data/sec
- Step1200: 118.99 data/sec
- Step1400: 119.95 data/sec
- Epoch 2 in 477.96 sec, test time 8.71 sec, test accuracy:0.9275
- Step1600: 120.59 data/sec
- Step1800: 120.70 data/sec
- Epoch 3 in 495.96 sec, test time 11.25 sec, test accuracy:0.9352

## ResNet50-2080ti-128batch_size
- Step200: 57.27 data/sec
- Step400: 58.51 data/sec
- Epoch 0 in 1033.6 sec, test time 23.66 sec, test accuracy:0.9743
- Step600: 59.01 data/sec
- Step800: 58.95 data/sec
- Epoch 1 in 984.57 sec, test time 17.91 sec, test accuracy:0.9747
- Step1000: 59.65 data/sec
- Step1200: 60.17 data/sec
- Epoch 2 in 957.91 sec, test time 18.63 sec. test accuracy 0.9786
- Step 1600 60.44 data/sec
- Step 1800 59.89 data/sec
- Epoch 3 in 1025.45 sec, test time 18.65 sec. test accuracy 0.9815

## ResNet101-2080ti-128batch_size
- Step200: 29.33 data/sec
- Step400: 30.31 data/sec
- Epoch 0 in 1980.37 sec, test time 43.67 sec, test accuracy:0.9580
- Step600: 29.92 data/sec
- Step800: 30.38 data/sec
- Epoch 1 in 1930.71 sec, test time 34.77 sec, test accuracy:0.9650
- Step1000: 30.79 data/sec
- Step1200: 31.04 data/sec
- Epoch 2 in 1848.75 sec, test time 34.44 sec. test accuracy 0.9760
- Step 1600 31.39 data/sec
- Step 1800 31.51 data/sec
- Epoch 3 in 1852.36 sec, test time 14.47 sec. test accuracy 0.9792

# MNIST With collective communication, Adam(lr=0.01)

## ResNet18-2x2080ti-128batch_size
- Step200: 194.62 data/sec
- Step400: 204.16 data/sec
- Epoch 0 in 592.16 sec, test time 15.5 sec, test accuracy:0.9221
- Step600: 203.8 data/sec
- Step800: 205.67 data/sec
- Epoch 1 in 563.78 sec, test time 10.13 sec, test accuracy: 0.9484
- Step1000: 207.73 data/sec
- Step1200: 209.25 data/sec
- Step1400: 210.28 data/sec
- Epoch 2 in 553.28 sec, test time 10.00 sec, test accuracy: 0.9534
- Step1600: 211.12 data/sec
- Step1800: 210.97 data/sec
- Epoch 3 in 562.83 sec, test time 9.85 sec, test accuracy: 0.9643

## ResNet50-2x2080ti-128batch_size
- Step200: 100.38 data/sec
- Step400: 103.13 data/sec
- Epoch 0 in 1167.73.16 sec, test time 26.04 sec. Test accuracy 0.9790
- Step600: 103.78 data/sec
- Step800: 104.40 data/sec
- Epoch 1 in 1121.86 sec, test time 21.07 sec. Test accuracy 0.9866
- Step1000: 104.97 data/sec
- Step1200: 104.96 data/sec
- Step1400: 105.18 data/sec
- Epoch 2 in 1133.25 sec, test time 20.41 sec. Test accuracy 0.9859
- Step1600: 105.51 data/sec
- Step1800: 105.37 data/sec
- Epoch 3 in 1129.23 sec, test time 21.01 sec. est accuracy 0.9853

## ResNet101-2x2080ti-128batch_size
- Step200: 54.26 data/sec
- Step400: 54.96 data/sec
- Epoch 0 in 2190.16 sec, test time 47.03 sec. Test accuracy 0.9610
- Step600: 54.48 data/sec
- Step800: 54.99 data/sec
- Epoch 1 in 2166.43 sec, test time 40.14 sec. Test accuracy 0.9775
- Step1000: 55.16 data/sec
- Step1200: 54.86 data/sec
- Step1400: 54.99 data/sec
- Epoch 2 in 2191.31 sec, test time 40.23 sec. Test accuracy 0.9817
- Step1600: 55.12 data/sec
- Step1800: 55.22 data/sec
- Epoch 3 in 2143.79 sec, test time 42.08 sec. est accuracy 0.9829