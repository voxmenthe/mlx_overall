#!/bin/bash

# Install the project in editable mode
pip install -e .

# Create and install the IPython kernel for the project
#python -m ipykernel install --user --name=mlx3132 --display-name "MLX 3.13.2"
#python -m ipykernel install --sys-prefix --name=mlx3132 --display-name "MLX 3.13.2"
python -m ipykernel install --sys-prefix --name=mlx3129 --display-name "MLX 3.12.9"

#echo "Jupyter kernel 'MLX 3.13.2' has been installed."
echo "Jupyter kernel 'MLX 3.12.9' has been installed."