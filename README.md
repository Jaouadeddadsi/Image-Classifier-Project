# Data Scientist Nanodegree
# Deep learning
## Project: Create an Image Classifier

### Description
In this project, I built a python application that can train an image classifier on a dataset, then predict new images using the trained model. The project was devided to two parts.

  ##### Part 1 - Developing an Image Classifier with Deep Learning
In this first part of the project, I implemened an image classifier with PyTorch.

  ##### Part 2 - Building the command line application
 Build a pair of Python scripts that run from the command line to train an image classifier and to predict new images using the trained model.

###  Installations

 This project requires **Python 3.x** and the following Python libraries installed:

 - [Argparse](https://docs.python.org/3/library/argparse.html)
 - [Numpy](https://www.numpy.org/)
 - [Pytorch](https://pytorch.org/)
 - [Json](https://docs.python.org/2/library/json.html)
 - [Pillow](https://pillow.readthedocs.io/en/stable/)
 - [Seaborn](https://seaborn.pydata.org/)
 - [Matplotlib](https://matplotlib.org/)


### Files Descriptions

 - Image Classifier Project: A Jupyter notebook contains code to implement an image classifier with PyTorch.
 - train.py: Code to train a new deep learning network on a dataset and save the model as a checkpoint.
 - predict.py: Code uses a trained network to predict the class for an input image.

### Data
   You can download the data used in this project from  [here](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

### Running codes
 - To train a new network on a data set with `train.py`:
     - Basic usage: `python train.py data_directory`
     - Options:
        - Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
        - Choose architecture: `python train.py data_dir --arch "vgg13"`
        - Set hyperparameters:  `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
        - Use GPU for training: `python train.py data_dir --gpu`

- To predict flower name from an image with `predict.py`:
     - Basic usage: `python predict.py /path/to/image checkpoint`
     - Options:
        - Return top KKK most likely classes: `python predict.py input checkpoint --top_k 3`
        - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
        - Use GPU for inference: `python predict.py input checkpoint --gpu`

### Author
 -   **Jaouad Eddadsi**  [linkedin](https://www.linkedin.com/in/jaouad-eddadsi-01bb34163/)

### License

 This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
