# Contributing

We'd love to accept your contributions. Please follow the guidelines below.

## Reporting bugs and requesting features

Users are encouraged to use GitHub issues for reporting issues and requesting features.

## Code review

All submissions require review. To make a submission, open a GitHub pull request.

## Submitting a new research experiment

You can submit an experiment if it satisfies the following requirements:

- It is programmed using Qiskit.
- It is executed on quantum hardware.
- There is an associated publication or publicly available preprint describing the results from running the experiment.

To submit a new research experiment:

1. Create a new module for your experiment under the [`qiskit_research`](qiskit_research/) base module.
2. Create a new directory under [docs/](docs/) and add Jupyter notebooks to document your experiment.

## Running tests locally

The tests can be run locally using [tox](https://tox.wiki/en/latest/).
To run the full test suite, execute `tox -e ALL`.
Individual checks can also be run separately. For example:

Run unit tests for Python 3.9

    tox -epy39

Run lint check

    tox -elint

Run type check

    tox -emypy

Run format check

    tox -eblack

## Building and checking the docs locally

To build the docs locally, run

    tox -edocs

This will generate the documentation files and place them in the directory
`docs/_build/html`. You can then view the files in your web browser, for example,
by navigating to `file:///QISKIT_RESEARCH_DIRECTORY/docs/_build/html/index.html`.
Please take this step when submitting a pull request to ensure that the changes
you make to the documentation look correct.
