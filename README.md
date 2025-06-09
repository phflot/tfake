![Image](img/fake-thermal.jpg)

# T-FAKE: Synthesizing Thermal Images for Facial Landmarking


## Downloading the dataset

To download the color images, sparse annotations, and segmentation masks for the dataset, please use the links in the [FaceSynthetics repository](https://github.com/microsoft/FaceSynthetics).

Our dataset has been generated for a warm and for a cold condition. Each dataset can be downloaded separately as

- A small sample with 100 images from [here warm](https://drive.google.com/file/d/1-Y40_wqVV5WM1swEpjFTJiB8sGdZtQwR/view?usp=sharing) and [here cold](https://drive.google.com/file/d/1-_-RHg7ZDzFFtoyeXJsyrdtgkcnJ3FMR/view?usp=sharing)
- A medium sample with 1,000 images from [here warm](https://drive.google.com/file/d/1-NcsaNa6dbfmQ0l6UjmwZSJWDsUFM4vW/view?usp=sharing) and [here cold](https://drive.google.com/file/d/1-PqPR86GDj5LB_6PZKlek6o6FNbkf7Fo/view?usp=sharing)
- The full dataset with 100,000 images from [here warm](https://drive.google.com/file/d/1-3-OC-VYL14uyLA4Vi9DpwDlkauuNh7K/view?usp=sharing) and [here cold](https://drive.google.com/file/d/1wh25Yi9sT-0j6qXz0JlHUtIIbLAYUnrZ/view?usp=sharing)
- The dense annotations are available from [here](https://drive.google.com/file/d/1-lMYaok0xbfQyBTxj6dcuxT1iryU7TOs/view?usp=sharing)

## Using the landmarker

Coming soon.

![landmarks](img/landmarks.jpg)

## Pre-trained models

The models for the thermalization as well as the landmarkers can be downloaded from [here](https://drive.google.com/drive/folders/1-ppKS4xuBY-EbmGCkvKTLYMXHA3lK8R8?usp=sharing).

### Thermalization

Our baseline U-Net translation model is imported from [segmentation_models_pytorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/) library. Specifically, we define the translator as follows:  

```python
import segmentation_models_pytorch as smp

translator = smp.Unet(
    encoder_name="resnet34",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=1,                      
    activation="sigmoid"
)
```

This model is based on a U-Net architecture with a ResNet-34 encoder pre-trained on ImageNet. It takes three-channel RGB input images and outputs a single-channel thermal image with a sigmoid activation function. For training progress of the thermalization model see [ThermalizationCode/ThermalizerOutput.ipynb](https://github.com/phflot/tfake/blob/main/ThermalizationCode/ThermalizerOutput.ipynb).

### Landmarking

Will be added soon.

## Running the benchmark

To run the benchmark, you have to download the [CHARLOTTE ThermalFace dataset](https://github.com/TeCSAR-UNCC/UNCC-ThermalFace). 


## License

This dataset and the landmarking methods are licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International](LICENSE.txt) license as it is derived from the [FaceSynthetics dataset](https://github.com/microsoft/FaceSynthetics).

## Citation

If you use this code for your own work, please cite our paper:
  
> P. Flotho, M. Piening, A. Kukleva and G. Steidl, “T-FAKE: Synthesizing Thermal Images for Facial Landmarking,” Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), 2025. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Flotho_T-FAKE_Synthesizing_Thermal_Images_for_Facial_Landmarking_CVPR_2025_paper.html)

BibTeX entry
```
@InProceedings{tfake2025_CVPR,
    author    = {Flotho, Philipp and Piening, Moritz and Kukleva, Anna and Steidl, Gabriele},
    title     = {T-FAKE: Synthesizing Thermal Images for Facial Landmarking},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {26356-26366}
}
```