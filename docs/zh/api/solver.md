# Solver(求解器) 模块

::: ppsci.solver
    handler: python
    options:
      members:
        - Solver
      show_root_heading: true
      heading_level: 3

::: ppsci.solver.train
    handler: python
    options:
      members:
        - train_epoch_func
        - train_LBFGS_epoch_func
      show_root_heading: true
      heading_level: 3

::: ppsci.solver.eval
    handler: python
    options:
      members:
        - eval_func
      show_root_heading: true
      heading_level: 3

::: ppsci.solver.visu
    handler: python
    options:
      members:
        - visualize_func
      show_root_heading: true
      heading_level: 3

::: ppsci.solver.printer
    handler: python
    options:
      members:
        - update_train_loss
        - update_eval_loss
        - log_train_info
        - log_eval_info
      show_root_heading: true
      heading_level: 3
