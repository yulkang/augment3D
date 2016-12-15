# Augment3D


## Installation
1. Download this code
2. Put it next to 'Data' folder. An example folder structure would be
  * augment3D
    * code # this repository
    * Data 
3. Download data
  1. Go to https://grand-challenge.org/site/luna16/download/
  2. Download `annotations.csv` and `candidates.csv` to `Data/LUNA/`
  3. Download `subset*.zip` to a suitable folder, such as `ExternalHDD/LUNA/`
  4. Extract the zip files, so that `.mhd` and `.raw` files are in `ExternalHDD/LUNA/subset*/`

## Usage
1. Rename `img_dir_root`, `img_dir`, and `patch_dir` in `paths.py`.
2. Move all images into one folder and create a `.csv` containing the UID and the subset info
  ```
  gather_subsets
  ```
3. Import images
  ```
  import_mhd # Takes a long time, potentially hours to days
  ```
2. 
