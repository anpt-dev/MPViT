This Repo for MPViT models.
As illustrated in the figure, the proposed architecture adopts a hierarchical CNN–Transformer framework in which convolutional operations and attention mechanisms are progressively integrated across different stages to effectively capture both local structural information and long-range contextual dependencies while maintaining computational efficiency. Given an input image of size (224x224x3), the Stem module first extracts low-level visual features and reduces the spatial resolution to (56x56). It consists of convolutional layers followed by GELU activation, batch normalization, and a Squeeze-and-Excitation (SE) module, providing an efficient initial representation before hierarchical feature learning.

<img width="831" height="499" alt="image" src="https://github.com/user-attachments/assets/a3c7fd3c-2160-439b-b729-2ba8c862c484" />
Fig. 1. Overvew Architecture

Patch Mixing mechanism

<img width="527" height="213" alt="Fig 4" src="https://github.com/user-attachments/assets/19581fd5-4055-4a34-b642-43b8a729d306" />
Fig. 2. Patch Mixing

Checkpoint on Imagenet_1K (100 epoches) https://drive.google.com/file/d/1_g9n6Jwz5GM5PCjm_ZnB9UfELwi7y_nP/view?usp=sharing

5 seeds data split on IQ-OTH/NCCD https://drive.google.com/drive/folders/1NEaKfAIi3PpYerN_Yfd5uf1hdekmtrhd?usp=sharing
