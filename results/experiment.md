# ParkSense experiment summary

Training data: UFPR04 Sunny + Cloudy training days only. All numbers are on held-out test frames.
*Arm F also trains on UFPR05 and PUCPR (other days), so for F those columns are held-out DAYS, not unseen car parks; CNRPark-EXT stays unseen for every arm.
CNN arms: mean ± std over seeds. Accuracy and macro-F1 per slot; count MAE = mean absolute error in the number of occupied bays per image.

## Slot accuracy

| Model | seeds | UFPR04 seen weather | UFPR04 real rain | Unseen PKLot parks | Unseen camera set (CNRPark) | All unseen rain |
|---|---|---|---|---|---|---|
| A0 Classical baseline (edges + colour) | 1 | 0.932 | 0.947 | 0.902 | 0.879 | 0.896 |
| A  CNN, geometric augmentation only | 3 | 0.995 ± 0.001 | 0.998 ± 0.001 | 0.962 ± 0.002 | 0.794 ± 0.002 | 0.933 ± 0.003 |
| B  CNN + classical photometric weather simulation | 3 | 0.988 ± 0.001 | 0.980 ± 0.008 | 0.958 ± 0.002 | 0.757 ± 0.000 | 0.947 ± 0.001 |
| C  CNN + CycleGAN synthetic rain | 3 | 0.995 ± 0.001 | 0.998 ± 0.001 | 0.964 ± 0.004 | 0.792 ± 0.008 | 0.934 ± 0.006 |
| F  CNN, trained on all 3 PKLot parks* | 1 | 0.997 | 0.999 | 0.988 | 0.788 | 0.942 |
| G  F + photometric simulation (deployed)* | 1 | 0.997 | 0.994 | 0.984 | 0.810 | 0.963 |

## Macro-F1

| Model | seeds | UFPR04 seen weather | UFPR04 real rain | Unseen PKLot parks | Unseen camera set (CNRPark) | All unseen rain |
|---|---|---|---|---|---|---|
| A0 Classical baseline (edges + colour) | 1 | 0.932 | 0.914 | 0.902 | 0.879 | 0.891 |
| A  CNN, geometric augmentation only | 3 | 0.995 ± 0.001 | 0.997 ± 0.001 | 0.962 ± 0.002 | 0.791 ± 0.003 | 0.930 ± 0.003 |
| B  CNN + classical photometric weather simulation | 3 | 0.988 ± 0.001 | 0.971 ± 0.012 | 0.958 ± 0.002 | 0.752 ± 0.000 | 0.945 ± 0.001 |
| C  CNN + CycleGAN synthetic rain | 3 | 0.995 ± 0.001 | 0.997 ± 0.001 | 0.964 ± 0.004 | 0.788 ± 0.009 | 0.931 ± 0.006 |
| F  CNN, trained on all 3 PKLot parks* | 1 | 0.997 | 0.999 | 0.988 | 0.784 | 0.940 |
| G  F + photometric simulation (deployed)* | 1 | 0.997 | 0.991 | 0.984 | 0.808 | 0.962 |

## Occupied-count MAE (bays per image)

| Model | seeds | UFPR04 seen weather | UFPR04 real rain | Unseen PKLot parks | Unseen camera set (CNRPark) | All unseen rain |
|---|---|---|---|---|---|---|
| A0 Classical baseline (edges + colour) | 1 | 2.37 | 1.65 | 7.46 | 4.55 | 5.28 |
| A  CNN, geometric augmentation only | 3 | 0.19 ± 0.03 | 0.07 ± 0.03 | 3.59 ± 0.14 | 7.95 ± 0.16 | 4.59 ± 0.18 |
| B  CNN + classical photometric weather simulation | 3 | 0.45 ± 0.02 | 0.68 ± 0.28 | 3.62 ± 0.22 | 9.60 ± 0.03 | 3.27 ± 0.13 |
| C  CNN + CycleGAN synthetic rain | 3 | 0.19 ± 0.04 | 0.07 ± 0.03 | 3.39 ± 0.43 | 8.10 ± 0.33 | 4.51 ± 0.41 |
| F  CNN, trained on all 3 PKLot parks* | 1 | 0.11 | 0.03 | 0.92 | 8.33 | 4.20 |
| G  F + photometric simulation (deployed)* | 1 | 0.10 | 0.20 | 1.26 | 7.35 | 2.33 |

## Per car park and weather (accuracy / macro-F1 / count MAE)

| Car park | Weather | Slots | A0 | A | B | C | F | G |
|---|---|---|---|---|---|---|---|---|
| UFPR04 | Sunny | 1830 | 0.919 / 0.909 / 2.8 | 0.991 / 0.990 / 0.3 | 0.980 / 0.978 / 0.7 | 0.991 / 0.990 / 0.3 | 0.995 / 0.994 / 0.2 | 0.995 / 0.994 / 0.2 |
| UFPR04 | Cloudy | 1460 | 0.949 / 0.937 / 1.9 | 1.000 / 1.000 / 0.0 | 0.997 / 0.996 / 0.1 | 1.000 / 1.000 / 0.0 | 0.999 / 0.999 / 0.0 | 1.000 / 1.000 / 0.0 |
| UFPR04 | Rainy | 2666 | 0.947 / 0.914 / 1.6 | 0.998 / 0.997 / 0.1 | 0.980 / 0.971 / 0.7 | 0.998 / 0.997 / 0.1 | 0.999 / 0.999 / 0.0 | 0.994 / 0.991 / 0.2 |
| UFPR05 | Sunny | 1840 | 0.873 / 0.867 / 4.4 | 0.961 / 0.959 / 1.8 | 0.951 / 0.949 / 1.9 | 0.960 / 0.959 / 1.8 | 0.989 / 0.989 / 0.3 | 0.979 / 0.978 / 0.8 |
| UFPR05 | Cloudy | 1840 | 0.912 / 0.909 / 3.8 | 0.980 / 0.979 / 0.9 | 0.976 / 0.975 / 1.1 | 0.984 / 0.983 / 0.8 | 0.995 / 0.994 / 0.2 | 0.988 / 0.987 / 0.6 |
| UFPR05 | Rainy | 1840 | 0.929 / 0.922 / 2.9 | 0.978 / 0.976 / 1.0 | 0.976 / 0.975 / 1.0 | 0.980 / 0.978 / 0.9 | 0.984 / 0.983 / 0.6 | 0.986 / 0.985 / 0.6 |
| PUCPR | Sunny | 5850 | 0.910 / 0.907 / 13.4 | 0.941 / 0.938 / 10.8 | 0.926 / 0.923 / 11.8 | 0.942 / 0.940 / 10.5 | 0.984 / 0.983 / 2.1 | 0.979 / 0.978 / 2.8 |
| PUCPR | Cloudy | 5850 | 0.896 / 0.887 / 15.5 | 0.976 / 0.973 / 3.9 | 0.969 / 0.965 / 5.2 | 0.977 / 0.974 / 3.8 | 0.992 / 0.991 / 1.4 | 0.989 / 0.987 / 2.0 |
| PUCPR | Rainy | 5850 | 0.897 / 0.885 / 8.6 | 0.959 / 0.954 / 5.4 | 0.969 / 0.965 / 3.0 | 0.962 / 0.958 / 4.9 | 0.986 / 0.985 / 1.4 | 0.985 / 0.983 / 1.5 |
| CNRPark-EXT | Sunny | 2196 | 0.863 / 0.860 / 4.9 | 0.778 / 0.778 / 8.4 | 0.697 / 0.697 / 12.0 | 0.777 / 0.777 / 8.6 | 0.778 / 0.778 / 8.5 | 0.776 / 0.776 / 8.6 |
| CNRPark-EXT | Cloudy | 2196 | 0.909 / 0.909 / 3.5 | 0.780 / 0.774 / 8.7 | 0.709 / 0.692 / 11.7 | 0.778 / 0.771 / 8.8 | 0.797 / 0.791 / 8.0 | 0.767 / 0.759 / 9.3 |
| CNRPark-EXT | Rainy | 2196 | 0.864 / 0.858 / 5.2 | 0.823 / 0.815 / 6.8 | 0.866 / 0.861 / 5.1 | 0.820 / 0.811 / 7.0 | 0.788 / 0.772 / 8.4 | 0.888 / 0.884 / 4.1 |
