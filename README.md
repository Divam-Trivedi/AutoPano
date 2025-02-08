# Project 1 AutoPano

## RBE/CS 549 COmputer Vision

The Phase 1 folder consists of a single file named Wrapper.py

It takes in 2 parameters -

```
--inputfolder
--outputfolder
```

The default input and output paths are given

### USAGE
To run the file, run the command -

```
python3 Wrapper.py --inputfolder="/path/to/images" --outputfolder="/path/to/save/images"
```


The Phase 2 folder has two files for Training and Testing of Unsupervised Netork. It also has a folder named 'Supervised' for training and testing files of Supervised Network. There is a Network folder which contains the architectures of both the methods. The wrapper.py file is used for getting the homographies from the network and stitch together two iamges.

This script generates a panorama from input images using either a supervised or unsupervised homography estimation model. It takes the following command-line arguments:  

- `--ModelPath`: Path to the trained model checkpoint.  
- `--TestJsonPath`: Path to the JSON file containing test image pairs and ground truth homographies.  
- `--InputImagesPath`: Directory containing input image patches.  
- `--OutputDir`: Directory to save the output panorama.  
- `--ModelType`: Specifies the model type (`Sup` for supervised, `Unsup` for unsupervised).  

### Usage  
Run the script with the following command:  

```bash
python3 wrapper.py --ModelPath /path/to/model.ckpt --TestJsonPath /path/to/test.json --InputImagesPath /path/to/images --OutputDir /path/to/output --ModelType Unsup
