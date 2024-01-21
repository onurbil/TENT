## TENT
Official implementation code of the paper: "TENT: Tensorized Encoder Transformer for temperature forecasting" [(ArXiv link)](https://arxiv.org/abs/2106.14742).

## Model architecture
![TENT](images/all_together.jpg)

(a) Model architectureTensorial. (b) Tensorial self-attention. (c) Tensorial multi-head attention.

## Results

![TENT](images/result_USA.png)
The obtained test MAE of the models for USA-Canada dataset averaged over cities (a) and prediction time steps (b).

![TENT](images/result_vancouver.png)
The comparison between the predictions of TENT model and the real measurements for hourly temperature of the test set of Vancouver.

## Attention visualization

![TENT](images/vancouver_maps.png)
Attention visualization for Dallas in USA-Canada dataset.

## Data

In order to download the data, please email to the following address:

s.mehrkanoon@uu.nl


## Usage
Execute the notebook on colab (Use TPU for TENT): [TT_All_models_experiments.ipynb](notebooks/TT_All_models_experiments.ipynb)

## Citation
If you use our data and code, please cite the paper using the following bibtex reference:
```
@article{bilgin2021tent,
  title={TENT: Tensorized Encoder Transformer for Temperature Forecasting},
  author={Bilgin, Onur and M{\k{a}}ka, Pawe{\l} and Vergutz, Thomas and Mehrkanoon, Siamak},
  journal={arXiv preprint arXiv:2106.14742v2},
  year={2021}
}
```

