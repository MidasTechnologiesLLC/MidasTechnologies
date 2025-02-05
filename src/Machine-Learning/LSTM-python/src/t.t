(venv) kleinpanic@kleinpanic:~/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src$ py LSTMDQN.py BAT.csv
2025-01-31 22:41:37.524313: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1738363297.545380 3148462 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1738363297.551750 3148462 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-01-31 22:41:37.573675: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2025-01-31 22:41:43,005 - INFO - ===== Resource Statistics =====
2025-01-31 22:41:43,005 - INFO - Physical CPU Cores: 28
2025-01-31 22:41:43,005 - INFO - Logical CPU Cores: 56
2025-01-31 22:41:43,005 - INFO - CPU Usage per Core: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]%
2025-01-31 22:41:43,005 - INFO - No GPUs detected.
2025-01-31 22:41:43,005 - INFO - =================================
2025-01-31 22:41:43,006 - INFO - Configured TensorFlow to use CPU with optimized thread settings.
2025-01-31 22:41:43,006 - INFO - Loading data from: BAT.csv
2025-01-31 22:41:44,326 - INFO - Data columns after renaming: ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
2025-01-31 22:41:44,339 - INFO - Data loaded and sorted successfully.
2025-01-31 22:41:44,339 - INFO - Calculating technical indicators...
2025-01-31 22:41:44,370 - INFO - Technical indicators calculated successfully.
2025-01-31 22:41:44,379 - INFO - Starting parallel feature engineering with 54 workers...
2025-01-31 22:41:53,902 - INFO - Parallel feature engineering completed.
2025-01-31 22:41:54,028 - INFO - Scaled training features shape: (14134, 15, 17)
2025-01-31 22:41:54,028 - INFO - Scaled validation features shape: (3028, 15, 17)
2025-01-31 22:41:54,028 - INFO - Scaled testing features shape: (3030, 15, 17)
2025-01-31 22:41:54,028 - INFO - Scaled training target shape: (14134,)
2025-01-31 22:41:54,028 - INFO - Scaled validation target shape: (3028,)
2025-01-31 22:41:54,029 - INFO - Scaled testing target shape: (3030,)
2025-01-31 22:41:54,029 - INFO - Starting LSTM hyperparameter optimization with Optuna using 54 parallel trials...
[I 2025-01-31 22:41:54,029] A new study created in memory with name: no-name-58aeb7f7-b8be-4643-9d01-0d7bcf35db2e
/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py:370: FutureWarning: suggest_loguniform has been deprecated in v3.0.0. This feature will be removed in v6.0.0. See https://github.com/optuna/optuna/releases/tag/v3.0.0. Use suggest_float(..., log=True) instead.
  learning_rate   = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
[W 2025-01-31 22:41:54,037] Trial 0 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 128, 'dropout_rate': 0.3458004047482393, 'learning_rate': 0.00032571516657639116, 'optimizer': 'Adam', 'decay': 5.1271378208025266e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,040] Trial 1 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 64, 'dropout_rate': 0.41366725075244426, 'learning_rate': 1.4215518116455374e-05, 'optimizer': 'Adam', 'decay': 2.4425472693131955e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,041] Trial 0 failed with value None.
[W 2025-01-31 22:41:54,044] Trial 2 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 64, 'dropout_rate': 0.4338960746358078, 'learning_rate': 0.0008904040106011442, 'optimizer': 'Nadam', 'decay': 5.346913345250019e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,045] Trial 1 failed with value None.
[W 2025-01-31 22:41:54,048] Trial 3 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 64, 'dropout_rate': 0.12636442800548273, 'learning_rate': 0.00021216094172774624, 'optimizer': 'Adam', 'decay': 6.289573710217091e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,051] Trial 4 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 96, 'dropout_rate': 0.4118163224442708, 'learning_rate': 0.0001753425558060621, 'optimizer': 'Nadam', 'decay': 1.0106893106530013e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,053] Trial 5 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 96, 'dropout_rate': 0.22600776619683294, 'learning_rate': 4.6020052773101484e-05, 'optimizer': 'Nadam', 'decay': 1.401502701741485e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,054] Trial 2 failed with value None.
[W 2025-01-31 22:41:54,059] Trial 6 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 96, 'dropout_rate': 0.49745444543788064, 'learning_rate': 0.004560559624417403, 'optimizer': 'Adam', 'decay': 9.80562105055051e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,060] Trial 3 failed with value None.
[W 2025-01-31 22:41:54,064] Trial 7 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 32, 'dropout_rate': 0.11175568582439271, 'learning_rate': 0.000970072556392495, 'optimizer': 'Adam', 'decay': 5.792236253956584e-06} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,065] Trial 4 failed with value None.
[W 2025-01-31 22:41:54,069] Trial 8 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 128, 'dropout_rate': 0.4128314285072633, 'learning_rate': 0.000545928656752339, 'optimizer': 'Adam', 'decay': 8.349182110406793e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,095] Trial 8 failed with value None.
[W 2025-01-31 22:41:54,073] Trial 5 failed with value None.
[W 2025-01-31 22:41:54,076] Trial 10 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 96, 'dropout_rate': 0.312090359026424, 'learning_rate': 0.004334434878981849, 'optimizer': 'Nadam', 'decay': 8.946685227991797e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,078] Trial 11 failed with parameters: {'num_lstm_layers': 3, 'lstm_units': 32, 'dropout_rate': 0.3176109191721788, 'learning_rate': 0.0010138486071155559, 'optimizer': 'Nadam', 'decay': 2.864596673239629e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,084] Trial 6 failed with value None.
[W 2025-01-31 22:41:54,084] Trial 12 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 96, 'dropout_rate': 0.23624224169024638, 'learning_rate': 0.0007065434808473306, 'optimizer': 'Adam', 'decay': 1.6045047417478787e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,088] Trial 7 failed with value None.
[W 2025-01-31 22:41:54,072] Trial 9 failed with parameters: {'num_lstm_layers': 3, 'lstm_units': 64, 'dropout_rate': 0.32982534569008337, 'learning_rate': 0.00044815992336546054, 'optimizer': 'Nadam', 'decay': 1.2045464023339681e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,097] Trial 10 failed with value None.
[W 2025-01-31 22:41:54,101] Trial 11 failed with value None.
[W 2025-01-31 22:41:54,104] Trial 12 failed with value None.
[W 2025-01-31 22:41:54,108] Trial 9 failed with value None.
[W 2025-01-31 22:41:54,126] Trial 13 failed with parameters: {'num_lstm_layers': 3, 'lstm_units': 32, 'dropout_rate': 0.4314674109696518, 'learning_rate': 0.00020500811974021594, 'optimizer': 'Nadam', 'decay': 9.329438318207097e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,126] Trial 13 failed with value None.
[W 2025-01-31 22:41:54,137] Trial 14 failed with parameters: {'num_lstm_layers': 3, 'lstm_units': 64, 'dropout_rate': 0.45933740233556053, 'learning_rate': 0.0016981825407295947, 'optimizer': 'Nadam', 'decay': 3.7526439477629106e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,138] Trial 14 failed with value None.
[W 2025-01-31 22:41:54,139] Trial 15 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 128, 'dropout_rate': 0.13179726561423677, 'learning_rate': 0.009702870830616994, 'optimizer': 'Nadam', 'decay': 1.5717160470745384e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,140] Trial 15 failed with value None.
[W 2025-01-31 22:41:54,142] Trial 16 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 64, 'dropout_rate': 0.1184952725205303, 'learning_rate': 0.0002901212127436873, 'optimizer': 'Adam', 'decay': 1.2671796687995818e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,143] Trial 16 failed with value None.
[W 2025-01-31 22:41:54,145] Trial 17 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 128, 'dropout_rate': 0.3911357548507932, 'learning_rate': 2.1174519659994443e-05, 'optimizer': 'Adam', 'decay': 7.113124525281298e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,146] Trial 18 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 128, 'dropout_rate': 0.194308829860494, 'learning_rate': 2.3684641389781485e-05, 'optimizer': 'Nadam', 'decay': 2.1823222065039084e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,146] Trial 17 failed with value None.
[W 2025-01-31 22:41:54,147] Trial 19 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 64, 'dropout_rate': 0.34952903992289974, 'learning_rate': 0.0001649975428188158, 'optimizer': 'Nadam', 'decay': 8.961070238582916e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,148] Trial 18 failed with value None.
[W 2025-01-31 22:41:54,150] Trial 19 failed with value None.
[W 2025-01-31 22:41:54,151] Trial 20 failed with parameters: {'num_lstm_layers': 3, 'lstm_units': 32, 'dropout_rate': 0.24862299600787863, 'learning_rate': 3.160302043940613e-05, 'optimizer': 'Nadam', 'decay': 4.432627646713297e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,152] Trial 22 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 128, 'dropout_rate': 0.24247452680935244, 'learning_rate': 0.009143026717679506, 'optimizer': 'Nadam', 'decay': 3.8695560131185495e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,154] Trial 23 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 96, 'dropout_rate': 0.27974565379013505, 'learning_rate': 0.0005552121580002416, 'optimizer': 'Adam', 'decay': 6.460942114176827e-06} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,155] Trial 20 failed with value None.
[W 2025-01-31 22:41:54,155] Trial 21 failed with parameters: {'num_lstm_layers': 3, 'lstm_units': 64, 'dropout_rate': 0.31566223075768207, 'learning_rate': 0.00013277190404539305, 'optimizer': 'Nadam', 'decay': 5.448184988496794e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,156] Trial 22 failed with value None.
[W 2025-01-31 22:41:54,157] Trial 24 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 64, 'dropout_rate': 0.20684570701871122, 'learning_rate': 2.02919005955524e-05, 'optimizer': 'Nadam', 'decay': 6.367297091468678e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,158] Trial 23 failed with value None.
[W 2025-01-31 22:41:54,158] Trial 25 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 64, 'dropout_rate': 0.14749229469818195, 'learning_rate': 1.6074589705354466e-05, 'optimizer': 'Nadam', 'decay': 2.9293835054420393e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,161] Trial 26 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 128, 'dropout_rate': 0.38879633341946584, 'learning_rate': 2.5036537142341482e-05, 'optimizer': 'Nadam', 'decay': 4.8346386929100394e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,161] Trial 21 failed with value None.
[W 2025-01-31 22:41:54,161] Trial 27 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 32, 'dropout_rate': 0.4311830196294676, 'learning_rate': 6.15743775325322e-05, 'optimizer': 'Adam', 'decay': 2.5290071255921133e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,162] Trial 28 failed with parameters: {'num_lstm_layers': 1, 'lstm_units': 32, 'dropout_rate': 0.14813081091496075, 'learning_rate': 0.0017948222377220397, 'optimizer': 'Adam', 'decay': 9.679895886200194e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,163] Trial 24 failed with value None.
[W 2025-01-31 22:41:54,164] Trial 29 failed with parameters: {'num_lstm_layers': 2, 'lstm_units': 64, 'dropout_rate': 0.4827525644514289, 'learning_rate': 0.000583829520138558, 'optimizer': 'Adam', 'decay': 3.9540551700479366e-05} because of the following error: NameError("name 'X_train' is not defined").
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 383, in lstm_objective
    model_ = build_lstm((X_train.shape[1], X_train.shape[2]), hyperparams)
                         ^^^^^^^
NameError: name 'X_train' is not defined
[W 2025-01-31 22:41:54,165] Trial 25 failed with value None.
[W 2025-01-31 22:41:54,166] Trial 26 failed with value None.
[W 2025-01-31 22:41:54,167] Trial 27 failed with value None.
[W 2025-01-31 22:41:54,168] Trial 28 failed with value None.
[W 2025-01-31 22:41:54,169] Trial 29 failed with value None.
Traceback (most recent call last):
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 897, in <module>
    main()
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/LSTMDQN.py", line 685, in main
    best_lstm_params = study_lstm.best_params
                       ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/study.py", line 119, in best_params
    return self.best_trial.params
           ^^^^^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/study/study.py", line 162, in best_trial
    best_trial = self._storage.get_best_trial(self._study_id)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kleinpanic/git-clones/MidasTechnologies/src/Machine-Learning/LSTM-python/src/venv/lib/python3.11/site-packages/optuna/storages/_in_memory.py", line 249, in get_best_trial
    raise ValueError("No trials are completed yet.")
ValueError: No trials are completed yet.

