# Augment3D
: Augmentation of a 3D image of a focal lesion with 3D rotation, flip, and translation

## Introduction
When detecting a lesion from a medical image, it is often a good strategy to first identify parts of the image (“patches”) that likely contain the lesion and classify the patches as containing the lesion or not. It can be especially effective when the lesion is relatively small compared to the whole image, as is the case for detecting lymph node metastases from thoraco-abdominal MRI scans or for detecting lung nodules from lung CT scans. 

The challenge is that there are usually not enough positive examples. Augment3D addresses this challenge by augmenting the positive examples with 3D rotation, flip, and translation, in-memory. Because the image is 3D, rotation in angles multiple of 90 degrees alone gives 3! = 6 times the data. Combined with flip (2^3=8), it gives 48 times the original data.

In translation, Augment3D shifts the data with a vector sampled uniformly within the sphere of the annotated radius of the lesion. When the radius of the lesion is 5 voxels, for example, the method augments the data more than 200 times. 

Augment3D achieves all augmentation including translation in-memory, by first saving the patches with enough margin, and then slicing the patches on retrieval. When there is not enough memory to contain all candidate patches (especially negative patches, since there can be vast number of them), Augment3D can load them incrementally from disk after set number of reuse.

Augment3D includes both a general-purpose module that augments 3D patches, and a special-purpose module that imports images and annotations of the LUNA (LUng Nodule Analysis) 2016 challenge. The core algorithm of Augment3D can be useful for classification of many types of 3D images when there is a small region of interest with defined size and when the classification criterion is approximately invariant to rotation, flip, and translation. 

## Requirements
* pandas
* PIL
* gzip

## Installation
1. Download this code
2. Put it next to the 'Data' folder. An example folder structure would be

   ```bash
   augment3D
   +-- code # this repository
   +-- Data 
   ```
        
3. Download data
    1. Go to https://grand-challenge.org/site/luna16/download/
    2. Download `annotations.csv` and `candidates.csv` to `Data/LUNA/`
    3. Download `subset*.zip` to a suitable folder, such as `ExternalHDD/LUNA/`
    4. Extract the zip files, so that `.mhd` and `.raw` files are in `ExternalHDD/LUNA/subset*/`

## Usage
1. Rename `img_dir_root`, `img_dir`, and `patch_dir` in `paths.py`.
2. Move all images into one folder and create a `.csv` containing the UID and the subset info

    ```python
    gather_subsets # separate from import_mhd (next step) because import_mhd takes a long time
    ```
    
3. Import images

    ```python
    import_mhd # Takes a long time, potentially hours to days
    ```
    
4. Use the data. Positive samples are automatically augmented.

    ```python
    import datasets as ds
    batch_size = 100
    imgs_train, labels_train, imgs_valid, labels_valid = \
            ds.get_train_valid(batch_size)
    ```

