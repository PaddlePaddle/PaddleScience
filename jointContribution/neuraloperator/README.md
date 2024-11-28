# neuraloperator

## usage

### install

1. paddle

    use nightly version.

    ```sh
    python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
    ```

2. paddle_harmonics

    ```sh
    # install triton
    pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly

    https://github.com/co63oc/PaddleScience.git
    git checkout fix2
    cd PaddleScience/jointContribution/paddle_harmonics
    pip install .
    export PYTHONPATH=/path/PaddleScience/jointContribution/paddle_harmonics:$PYTHONPATH
    ```

3. neuraloperator

    ```sh
    # this pr

    # install ppsci
    pip install .

    # install neuralop
    cd PaddleScience/jointContribution/neuraloperator
    pip install .
    ```

### test

    ```sh
    cd PaddleScience/jointContribution/neuraloperator
    pytest
    ```
