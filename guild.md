# Using Guild AI

You can use Guild AI with this project to reproduce and compare
results.

- Automate model operations
- Automatically capture run results
- Compare loss and test accuray
- View run logs with TensorBoard

You can also use Guild AI to run model operations on EC2 GPUs to
accelerate training.

Contents:

- [Setup](#setup)

## Setup

### Install Guild AI

Install Guild AI using pip:

``` bash
pip install guildai
```

### Clone this repository

If you haven't already cloned this repository, run:

``` bash
git clone https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow.git
```

### Initialize an environment

We recommend running Guild commands in a virtual environment.

Initialize an environment from the project directory:

``` bash
cd Matrix-Capsules-EM-Tensorflow
guild init -p python3
```

This creates an environment `env` in the current directory that uses
Python 3. It also installs TensorFlow and the packages listed in
`requirements.txt`.

### Activate the environment

Before running Guild commands in an environment, you must actiate it
using the `source` command:

``` bash
source guild-env
```

### Check Guild AI

Finally, check the environment by running:

``` bash
guild check
```
