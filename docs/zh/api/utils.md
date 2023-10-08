# Utils(工具) 模块

::: ppsci.utils
    handler: python
    options:
      members:
        - AttrDict
        - AverageMeter
        - ExpressionSolver
        - initializer
        - logger
        - misc
        - reader
        - profiler
        - load_csv_file
        - load_mat_file
        - load_npz_file
        - load_vtk_file
        - load_vtk_with_time_file
        - dynamic_import_to_globals
        - run_check
        - run_check_mesh
        - set_random_seed
        - load_checkpoint
        - load_pretrain
        - save_checkpoint
        - lambdify
      show_root_heading: True
      heading_level: 2

::: ppsci.utils.checker
    handler: python
    options:
      members:
        - run_check
        - run_check_mesh
        - dynamic_import_to_globals
      show_root_heading: True
      heading_level: 2

::: ppsci.utils.initializer
    handler: python
    options:
      members:
        - uniform_
        - normal_
        - trunc_normal_
        - constant_
        - ones_
        - zeros_
        - xavier_uniform_
        - xavier_normal_
        - kaiming_uniform_
        - kaiming_normal_
        - linear_init_
        - conv_init_
      show_root_heading: True
      heading_level: 2

::: ppsci.utils.logger
    handler: python
    options:
      members:
        - init_logger
        - set_log_level
        - info
        - message
        - debug
        - warning
        - error
        - scaler
        - advertise
      show_root_heading: True
      heading_level: 2

::: ppsci.utils.reader
    handler: python
    options:
      members:
        - load_csv_file
        - load_mat_file
        - load_npz_file
        - load_vtk_file
        - load_vtk_with_time_file
      show_root_heading: True
      heading_level: 2

::: ppsci.utils.misc
    handler: python
    options:
      members:
        - Timer
        - convert_to_dict
        - convert_to_array
        - concat_dict_list
        - stack_dict_list
        - all_gather
        - typename
        - combine_array_with_time
        - cartesian_product
        - set_random_seed
      show_root_heading: True
      heading_level: 2
