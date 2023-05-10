# Moment Gauge Installation Guide

This guide will help you install and set up the Moment Gauge project on your local machine.

## Prerequisites

You will need to have Anaconda or Miniconda installed on your system. If you don't have it already, please follow the instructions to download and install Anaconda from [here](https://www.anaconda.com/distribution/) or Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

## Step 1: Clone the Moment Gauge Repository

Clone the Moment Gauge repository from GitHub to your local machine using the following command:

```
git clone https://github.com/scraed/TheMomentGauge.git
```

Replace `username` with the appropriate GitHub username.

## Step 2: Create a Conda Environment

Navigate to the Moment Gauge directory:

```
cd moment-gauge
```

Create a new conda environment using the `environment.yml` file provided:

```
conda env create -f environment.yml
```

This will create a new environment named `moment-gauge` with all the necessary dependencies.

## Step 3: Activate the Environment

Activate the newly created environment:

```
conda activate moment-gauge
```

## Step 4: Add Moment Gauge to Python Path

Add the Moment Gauge package to your Python path by running the following command:

```
export PYTHONPATH=$PYTHONPATH:[directory to moment gauge package]
```

Replace `[directory to moment gauge package]` with the full path to the `moment-gauge` directory you cloned in Step 1.

## Step 5: Verify Installation

You should now be able to use the Moment Gauge package in your Python projects. To verify that the installation was successful, open a Python interpreter and try to import the package:

```
python
```

```
import moment_gauge
```

If there are no errors, the installation was successful.

That's it! You have successfully installed and set up the Moment Gauge project. Happy coding!