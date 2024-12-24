# Visual Disambiguation Project

This project implements a Visual Disambiguation Model that integrates depth estimation and feature extraction using the DINOv2 backbone and ResNet. The model is designed to process image pairs and predict their relationships based on visual features and depth information.

## Project Structure

```
visual-disambiguation-project
├── src
│   ├── app.py                  # Main entry point of the application
│   ├── models
│   │   ├── __init__.py         # Models package initialization
│   │   ├── visual_disambiguation_model.py  # Definition of the VisualDisambiguationModel class
│   │   └── depth_estimation.py  # Depth estimation functionality
│   ├── utils
│   │   ├── __init__.py         # Utils package initialization
│   │   └── image_utils.py      # Image processing utility functions
│   └── data
│       ├── __init__.py         # Data package initialization
│       └── dataset.py          # Dataset class for loading image pairs
├── requirements.txt             # Project dependencies
├── setup.py                     # Packaging information
└── README.md                    # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Run the application**: Execute the main script to start the training and evaluation process.

```bash
python src/app.py
```

2. **Model Training**: The model will load the dataset, initialize the DINOv2 backbone, and perform training using the specified parameters.

3. **Depth Estimation**: The model will also estimate depth for each image, which will be used as a mask during feature extraction.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.