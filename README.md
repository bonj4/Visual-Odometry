# Visual-Odometry

This repository contains code for Visual Odometry. The project is structured as follows:

- **app:** Source code for the Visual Odometry application.
- **Dockerfile:** Docker configuration file for containerization.
- **pytorch_cuda_kurulum_komutu.txt:** Commands for installing PyTorch with CUDA support.
- **requirements.txt:** List of Python dependencies for the project.

## Getting Started

To get started with Visual Odometry, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/bonj4/Visual-Odometry.git
    cd Visual-Odometry
    python VisualOdometry.py
    ```

2. Review the contents of `pytorch_cuda_kurulum_komutu.txt` for PyTorch installation commands, especially if you are using a GPU.

3. Build the Docker image (if applicable):

    ```bash
    docker build -t visual-odometry .
    ```

4. Run the Visual Odometry application:

    ```bash
    python app/main.py
    ```

## Docker Support

If you prefer to run the application in a Docker container, follow the steps in the Dockerfile and build the Docker image as mentioned in the Getting Started section.

## Dependencies

Ensure you have the required dependencies installed. You can install them using:

```bash
pip install -r requirements.txt
```
