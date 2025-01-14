## File description

In each folder, please run all required files in the following order:

```_fit.py```: the codes are to train the ICNN model as the surrogate loss function.

```_independ_train.py```: the codes are to train the LSTM and thermal dynamics model following the traditional MPC and solve the optimization results.

```_joint_train.py```: the codes are to reproduce the proposed SMC strategy by taking the pre-trained models as the warm start.

```DF_Model_Func.py```: all required functions to be cited in other files.
