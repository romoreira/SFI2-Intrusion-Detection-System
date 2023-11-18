# Intrusion Detection System (IDS) based on Federated Learning for Network Slicing Reference Architectures

Developed by Rodrigo M.  <br>
[![](https://img.shields.io/badge/GitHub%20Pages-222222?style=for-the-badge&logo=GitHub%20Pages&logoColor=white)](https://romoreira.github.io)

## Overview :bulb:

This project aims to develop an Intrusion Detection System (IDS) leveraging Artificial Intelligence techniques within the context of a Network Slicing Reference Architecture. The system is designed to monitor, analyze, and detect potential intrusions or anomalous activities within distinct network slices.

![DDoS](https://github.com/romoreira/SFI2-Intrusion-Detection-System/blob/20c55f890befe9342e414661df80bb16be3104fe/DDoS.gif)


## Objectives :art:

- Implement machine learning algorithms for anomaly detection within specific network slices.
- Develop a scalable and adaptable IDS framework capable of integrating with diverse network architectures.
- Provide real-time monitoring and alerts for potential security threats within each network slice.

## Features :rotating_light:

- **Machine Learning Models:** Utilize various AI models (e.g., neural networks, decision trees) for anomaly detection.
- **Network Slice Integration:** Ensure compatibility with different network slicing architectures.
- **Real-time Monitoring:** Continuous monitoring of network behavior for immediate threat detection.
- **Alerting System:** Notify administrators or relevant entities upon detection of potential intrusions.

## Architecture :office:

### Components

1. **Data Collection Module:** Gathers network traffic data within each network slice.
2. **Preprocessing Unit:** Cleans and preprocesses incoming data for analysis.
3. **AI Engine:** Utilizes machine learning models to detect anomalies.
5. **Alerting Mechanism:** Notifies stakeholders in case of detected intrusions.

### Workflow

1. **Data Collection:** Traffic data collected from each network slice.
2. **Preprocessing:** Data undergoes cleaning and transformation.
3. **Anomaly Detection:** AI Engine analyzes preprocessed data for anomalies.
4. **Alert Generation:** Alerts triggered upon detection of potential intrusions.

## Usage

### Requirements

- Python 3.x
- Machine learning libraries (e.g., Pytorch, Scikit-learn, Flower)
- Network slicing framework dependencies

### Installation :computer:

1. Clone the repository:

    ```bash
    git clone https://github.com/romoreira/SFI2-Intrusion-Detection-System.git
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Configure settings and parameters according to your network architecture.

### Running the System

1. Navigate to the project directory:

    ```bash
    cd IDS-Network-Slicing
    ```

2. Execute the main script (run_experiments.sh):

    ```bash
    ./run_experiments.sh
    ```

### Branches

This repository contains the following branches:

- `local_training`
- `round_2`
- `round_4`
- `round_8`
- `round_16`

Each `round` branch refers to a specific number of interactions between clients and servers in federated learning.

### Hyperparameter Optimization

The hyperparameters have been optimized using [Optuna](https://optuna.org/) to enhance the system's performance.

## Contribution

Contributions are welcome! Feel free to fork this repository, make changes, and submit a pull request with your improvements or additional features.

## Financial Support

* [FAPESP](https://fapesp.br/)

## License

This project is licensed under the [MIT License](LICENSE).
