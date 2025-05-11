
# Step 1. Model selection:
Generic LLMs can be downloaded from the open-source platform at [Hugging Face Models](https://huggingface.co/models). Taking the QwQ-32B LLM as an example, you can directly access the following link：https://huggingface.co/Qwen/QwQ-32B/tree/main.
If you're in Mainland China and find Hugging Face downloads slow, please refer to the ["downloaded model by Python" file](https://github.com/baikediguo/3D-Chiplet-model/blob/main/downloaded%20model%20by%20Python). Note that you need to first install the required module via `pip install huggingface_hub`.
![Image](https://github.com/user-attachments/assets/337095fd-9b12-4806-abfe-fc3b29f5657e)

# Step 2. Data download:
 The data can be downloaded from the official website at GitHub - [ansys/pyaedt: AEDT Python Client Packag](https://github.com/ansys/pyaedt/tree/main). Note that this data is updated regularly, so ensure your Python environment settings stay consistent with it.
 ![Image](https://github.com/user-attachments/assets/fa267869-9a09-432c-99b4-c8d6588b808e)

# Step 3. Data pre-processing:
To analyze the data files of each new release, we developed a data statistical code. This code is designed to load and preprocess the code data, remove unnecessary elements such as comments and extra spaces, and collect information including the number of records, file paths, creation and modification times, code line counts, and encoding formats. The statistical code helps us systematically evaluate and understand the changes and features of each new version of the software. 
For detailed code, refer to [Data statistics](https://github.com/baikediguo/3D-Chiplet-model/blob/main/Data%20%20statistics). 
![Image](https://github.com/user-attachments/assets/2294b1ef-dd30-44ae-a020-28c9a3f880f4)

# Step 4. Model training:
We've trained data using 6 generic models. You can find the source code in the repository by looking for files starting with "model training for [Name]". Due to the long training time, we only show the training results in pictures instead of videos. For detailed code, refer to
![1741006152378](https://github.com/user-attachments/assets/79280bc8-c624-41fb-b336-6ceb93b8b08d)![1741006196827](https://github.com/user-attachments/assets/7ff905ef-c6db-41d0-8a4a-0b0cf0834711)


# Step 5. Model Verification:
We employed two methods for model validation. The first method involves directly writing text in Python to generate python code for invoking PyAEDT. For detailed code,refer to [verification model method-1](https://github.com/baikediguo/3D-Chiplet-model/blob/main/verification%20model%20method-1) 
![Image](https://github.com/user-attachments/assets/eca3ff1d-2153-4f3b-a28b-35101cbdc8da)

The second method utilizes a graphical user interface (GUI) to input files, thereby generating the python code by [verification model method-2](https://github.com/baikediguo/3D-Chiplet-model/blob/main/verification%20model%20method-2)
![Image](https://github.com/user-attachments/assets/a78bfcba-e210-487c-90d8-c684b28bb85c)

# Step 6. 3D Chiplet Modeling and Thermal Analysis:
 Users can iterate through numerous parameters (geometric, materials, meshing, boundary conditions, etc.) to explore hotspots, reducing design time consumption and enhancing efficiency.
![Image](https://github.com/user-attachments/assets/ad4e8623-9792-4d3c-8612-7138c31ac626)

All the code utilized in this project has been visualized using Code2flow（a Python tool）, making it easier for others to grasp the execution flow and dependencies within the codebase. Code2flow automatically generates flowcharts and call graphs to provide a clear visual representation of the code structure.


pip install pyaedt==0.14.0



>📋  A template README.md for code accompanying a Machine Learning paper

# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
