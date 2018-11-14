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
- [Review project](#review-project)
- [Results on smallNORB](#results-on-smallnorb)
- [CapsNet vs baseline CNN](#capsnet-vs-baseline)
- [Experiment with parameters](#experiment-with-parameters)

## Setup

### Install Guild AI

Install Guild AI using pip:

``` bash
pip install guildai
```

### Clone project

If you haven't already cloned the project repository, run:

``` bash
git clone https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow.git
```

### Initialize environment

We recommend running Guild commands in a virtual environment.

Initialize an environment from the project directory:

``` bash
cd Matrix-Capsules-EM-Tensorflow
guild init -p python3
```

This creates an environment `env` in the current directory that uses
Python 3. It also installs TensorFlow and the packages listed in
`requirements.txt`.

### Activate environment

Before running Guild commands in an environment, you must actiate it
using the `source` command:

``` bash
source guild-env
```

Note that the command prompt changes to include the environment name
in the format `(Matrix-Capsules-EM-Tensorflow) <old prompt>`. Use this
to verify that your environment is activated.

### Check Guild

Finally, check Guild by running:

``` bash
guild check
```

If you get errors and cannot resolve them, try getting more
information by using the `--verbose` option:

``` bash
guild check --verbose
```

and open an issue on the [Guild AI issue
tracker](https://github.com/guildai/guildai/issues).

## Review project

Before running operations, review the project by listing available
models, operations, and viewing project help.

All of the project information displayed below is defined in
[guild.yml](guild.yml).

The steps below must be run from the project directory.

List project models:

``` bash
guild models
```

Guild shows three models:

```
./baseline    Baseline CNN
./capsnet     CapsNet model
./small-norb  Support for small NORB dataset
```

`baseline` and `capsnet` represent the networks we train and
evaluate. `small-norb` represents the smallNORB dataset, which we
prepare for use in training and test.

Each model supports one or more operations.

List available operations:

``` bash
guild ops
```

Guild shows the available operations:

```
./baseline:evaluate   Evaluate trained baseline CNN
./baseline:train      Train baseline CNN
./capsnet:evaluate    Evaluate trained CapsNet
./capsnet:train       Train CapsNet
./small-norb:prepare  Prepare small NORB for training
```

Each operation is run using the command form `guild run
MODEL:OPERATION`.

You can run operations to reproduce results and conduct your own
experiments. Refer to the sections below for help with various
scenarios.

- [Reproduce CapsNet results on smallNORB](#results-on-smallnorb)
- [Compare CapsNet to baseline CNN](#capsnet-vs-baseline-cnn)
- [Experiment with parameters](#experiment-with-parameters)

## Results on smallNORB

The steps below reproduce the results reported in [README.md](README.md).

Ensure that you have followed the steps for [Setup](#setup) above
before continuing.

The steps below must be run from the project directory.

### Train CapsNet on smallNORB

``` bash
guild run capsnet:train dataset=smallNORB
```

Review the flags for the operation and press `Enter` to confirm.

If you want to change any flag values, press `n` and then `Enter` and
re-run the operation using the destired flags.

For help with available flags, run:

``` bash
guild run capsnet:train --help-op
```

To train over 5 epochs instead of the 50 (the default), run:

``` command
guild run capsnet:train epochs=5 dataset=smallNOBR
```

**NOTE** The project's default model is `capset` and the default value
for `dataset` is `smallNORB`. This means you can train CapsNet on
smallNORB by simply running `guild run train`.

**NOTE** The default number of epochs is 50, which reproduces the
results in README.md. However, training over 50 epochs even on a fast
GPU can take over several hours. If you want to train quickly, use a
smaller number of epochs. Your results in this case, however, will not
match those in README.md.

### View training in TensorBoard

You can use TensorBoard to monitor your training operations.

While CapsNet is training, open a second command console and activate
the environment:

``` bash
cd Matrix-Capsules-EM-Tensorflow
source guild-init
```

**NOTE** The above command must be run in a second command console if
the train operation is running.

Verify that you can see the active operation by listing runs:

``` bash
guild runs
```

You should see the active run (run ID and date will differ):

```
[1:53411e98]  ./capsnet:train       2018-11-14 10:14:12  running    smallNORB
```

Use Guild to open TensorBoard:

```
guild tensorboard
```

This start TensorBoard on an available port and opens it in your
browser. Guild monitors project runs and updates TensorBoard
automatically. You can leave TensorBoard running while CapsNet trains.

**NOTE** If you are training on a remote server, you may want to
specify the port that TensorBoard listens to. In this case, run:

``` bash
guild tensorboard --port PORT
```

Guild does not open TensorBoard in your browser when run on a remote
server. You must open the link that Guild displays in the command
prompt manually.

### Evaluate CapsNet

You can test CapsNet either while it is learning or after the train
operation finishes.

In either case, run:

``` bash
guild run capsnet:evaluate dataset=smallNORB
```

The `evaluate` operation calculates model accuracy using the test
examples associated with the specified dataset.

If you are viewing project runs in TensorFlow, the test accuracy is
displayed as `average_accuracy` for the applicable training epoch.

## CapsNet vs baseline CNN

You can use Guild to compare performance between CapsNet and the
baseline CNN.

If you haven't already trained and evaluated a CapsNet model, follow
the steps in [Results on smallNORB](#results-on-smallnorb) before
completing the steps below.

### Train baseline CNN

Train the baseline model on smallNORB by running:

``` bash
guild run baseline:train dataset=smallNORB
```

Guild shows the default flags, which are the same default values used
for `capsnet:train`. To accept the defaults, press `Enter`.

**NOTE** If you trained CapsNet in the previous section, use the same
flag values when training the baseline CNN.

### Evaluate baseline CNN

Evaluate the baseline CNN by running:

``` bash
guild run baseline:evaluate dataset=smallNORB
```

### Compare CapsNet and baseline CNN

With CapsNet and baseline CNN trained and tested, use Guild Compare to
compare accuracy and loss.

``` bash
guild compare
```

This command launches a program that lets you browse runs and their
results.

When you're finished comparing, press `q` to exit the program.

Next, use TensorBoard to compare runs:

``` bash
guild tensorboard
```

Expand the **accuracy** or **average_accuracy** scalars to compare
model accuracy.

**NOTE** TensorBoard may take a minute or two to load all of the data
available. If you don't see all of the data you expect, wait a few
minutes. TensorBoard automatically refreshes when new data is loaded.

![](img/caps_vs_baseline_accuracy.png)

## Experiment with parameters

Experiment with different parameters for `train` operations.

To get project help, including available models, operations and flags,
run:

``` bash
guild help
```

Flags are set for an operation in the form `NAME=VALUE`. For example,
the following trains capsnet with a batch size of 32:

``` command
guild run train batch_size=32
```

### Datasets

In addition to smallNORB, which is the default dataset used in train
operations, the following additional datasets are supported:

- `mnist`
- `fashion_mnist`
- `cifar10` (TODO)
- `cifar100` (TODO)

### Structure parameters

The following structure parameters are supported:

- `A` - Number of channels in output from ReLU Conv1
- `B` - Number of capsules in output from PrimaryCaps
- `C` - Number of channels in output from ConvCaps1
- `D` - Number of channels in output from ConvCaps2
