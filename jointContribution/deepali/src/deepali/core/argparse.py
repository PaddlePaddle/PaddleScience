r"""Utility functions for creating argparse based command-line interfaces.

In order to implement a CLI module, first define an argument parser by implementing
a ``parser()`` function within the module which instantiates an ``ArgumentParser``
with the respective command-line arguments of the CLI module.

.. code::

    def parser(*args, **kwargs) -> ArgumentParser:
        ...

Next, implement the main function of the CLI module. To indicate that this function
receives already parsed command-line arguments, and in accordance with the respective
attribute of an argparse subparsers, this function must be named ``func()``. In addition,
an ``init()`` function can optionally be defined. It can be used, for example, to
configure the loggers and to set environment variables before ``func()`` is invoked.
These initialization steps can also be done in ``func()``, but by splitting these out
into ``init()`` it is possible to implement a CLI module which calls the ``func()`` of
other CLI modules without repeating these initialization steps. The return value of
``init()`` and ``func()`` must be the exit code of the program.

.. code::

    logger = logging.getLogger()

    def init(args: ParsedArguments) -> int:
        configure_logging(args, logger)
        init_omp_num_threads(args.threads)

    def func(args: ParsedArguments) -> int:
        ...
        return 0

The above functions are sufficient to implement a CLI module. Finally, the description
of the CLI module which is printed as part of the default help output should be given
as docstring at the top of the module.

Lastly, an actual ``main()`` function which invokes the CLI module has to be defined
and either be called explicitly when ``__name__ == "__main__"``, or in a special module
file named ``__main__.py`` which is executed by the Python interpreter when the package
containing the CLI module is being executed using ``python -m``.

The ``main_func()`` closure can be used to implement a script for a single CLI, i.e.,

.. code::

    main = main_func(parser, func, init=init)

    if __name__ == "__main__":
        sys.exit(main())

The ``entry_point()`` closure can be used to implement a script which can invoke multiple CLIs.
Which CLI is being executed depends on the subcommand name. Subcommands can further be nested.
An example of such nested subcommands is given by ``conda``, e.g., ``conda env config vars set``.
This can be implemented using the ``entry_point()`` defined by this module as follows.

Implement the CLI module ``conda/env/config/vars/set.py`` which must define the ``parser()`` and
``func()`` function as detailed above. Then copy the following code into ``conda/__main__.py``

.. code::

    import sys

    from hfresearch.cli.argparse import entry_point

    main = entry_point(
        package=__name__,
        path=__path__,
        description="...",
        subcommands=["env.config.vars.set"],
    )

    sys.exit(main())

The ``subcommands`` keyword argument is optional. If not specified, ``entry_point()`` will
auto-discover the available CLI modules when ``--help`` is requested. By specifying the
available subcommands explicitly, the help output is generated faster.

"""

import argparse
import inspect
import os
import sys
from collections import namedtuple
from importlib import import_module
from pkgutil import walk_packages
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution
from typing_extensions import Protocol

from .psutil import memory_limit
from .typing import BoolStr

ParsedArguments = argparse.Namespace  # for type annotations
UnknownArguments = List[str]

Args = ParsedArguments


STANDARD_ARGUMENTS_GROUP_TITLE = "Standard arguments"

# Valid values for arguments of type BoolStr
TRUE = (True, "1", "yes", "on", "true", "True", "TRUE")
FALSE = (False, "0", "no", "off", "false", "False", "FALSE")


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    # See: https://stackoverflow.com/a/35848313

    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = "Usage: "
        return super().add_usage(usage, actions, groups, prefix)


class ArgumentParser(argparse.ArgumentParser):
    """Customized ArgumentParser subclass to be propagated to subparsers."""

    def __init__(self, add_help=False, formatter_class=None, version=None, **kwargs):
        if formatter_class is None:
            formatter_class = HelpFormatter
        super().__init__(add_help=False, formatter_class=formatter_class, **kwargs)
        # See: https://stackoverflow.com/a/35848313
        self._positionals.title = "Positional arguments"
        self._optionals.title = "Optional arguments"
        group = self.add_argument_group(STANDARD_ARGUMENTS_GROUP_TITLE)
        if version:
            group.add_argument(
                "--version",
                action="version",
                version="%(prog)s " + version,
                help="Show program version number and exit.",
            )
        if add_help:
            group.add_argument(
                "-h",
                "--help",
                action="help",
                default=argparse.SUPPRESS,
                help="Show this help message and exit.",
            )


class ParserCallable(Protocol):
    r"""Type annotation for argument parser constructor."""

    def __call__(self, **kwargs) -> ArgumentParser:
        ...


class MainCallable(Protocol):
    r"""Type annotation for main function."""

    def __call__(self, args: Optional[List[Any]] = None) -> int:
        ...


def entry_point(
    package: str,
    path: Iterable[str],
    description: Optional[str] = None,
    subcommands: Optional[Union[Dict[str, str], Sequence[str]]] = None,
) -> Callable[[Optional[List[Any]]], int]:
    r"""Create entry point for running named subcommand with given arguments.

    Args:
        package: Name of package containing CLI modules.
        path: Package paths containing CLI modules.
        description: Description of entry point command.
        subcommands: Description of subcommands. Dictionary keys are subcommand module
            names excluding the ``package`` name of the main CLI module. The included
            Python modules must define functions ``parser()`` and ``func()`` within
            their global scope. If ``None``, all Python subpackages and modules within
            the ``package`` are added automatically.

    Returns:
        main: Main entry point of package CLI which takes a list
            of command arguments as ``argv`` argument and returns an exit code.

    """
    if subcommands and not isinstance(subcommands, dict):
        subcommands = {name: "" for name in sorted(subcommands)}

    def main(argv: Optional[List[Any]] = None) -> int:
        r"""Run named subcommand with given arguments.

        Args:
            argv (list): Command-line arguments. If None, sys.argv[1:] is used.

        Returns:
            exit_code (int): Exit code of subcommand.

        """
        if argv is None:
            argv = sys.argv[1:]
        argv = [str(arg) for arg in argv]

        # Special 'help' subcommand or no argument
        if not argv:
            argv = ["-h"]
        elif argv[0] == "help":
            argv = argv[1:] + ["-h"]

        # CLI information
        dist = _pypi_package_name(package)
        if "__main__" in sys.argv[0]:
            prog = f"{os.path.basename(sys.executable)} -m {package}"
        else:
            prog = os.path.basename(sys.argv[0])
        try:
            version = get_distribution(dist).version
        except DistributionNotFound:
            version = None

        # Main parser
        mainparser = ArgumentParser(
            prog=prog, description=description, add_help=True, version=version
        )
        subparsers = mainparser.add_subparsers()

        default_options_parser = ArgumentParser(add_help=False)
        group = default_options_parser.add_argument_group(STANDARD_ARGUMENTS_GROUP_TITLE)
        group.add_argument(
            "-h",
            "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Show this help message and exit.",
        )
        if version:
            group.add_argument(
                "--version",
                action="version",
                version="%(prog)s " + version,
                help="Show program version number and exit.",
            )

        # Discover CLIs
        CommandInfo = namedtuple("CommandInfo", ["module", "commands"])

        def find_commands(name: str, path: List[str]) -> Dict[str, CommandInfo]:
            commands = {}
            module_infos = [
                (basename, ispkg)
                for _, basename, ispkg in walk_packages(path)
                if not basename.startswith("_")
            ]
            for basename, ispkg in module_infos:
                module_name = ".".join([name, basename])
                command_name = basename.replace("_", "-")
                commands[command_name] = CommandInfo(
                    module=module_name,
                    commands=find_commands(
                        name=module_name, path=[os.path.join(prefix, basename) for prefix in path]
                    )
                    if ispkg
                    else None,
                )
            return commands

        if subcommands:
            commands = {}
            for module_name in subcommands.keys():
                name_parts = module_name.split(".")
                command_name = name_parts[-1].replace("_", "-")
                parent_module = package
                parent_commands = commands
                for parent_name in name_parts[:-1]:
                    parent_module = ".".join([parent_module, parent_name])
                    parent_command = parent_name.replace("_", "-")
                    if parent_command not in parent_commands:
                        parent_commands[parent_command] = CommandInfo(
                            module=parent_module, commands={}
                        )
                    parent_commands = parent_commands[parent_command].commands
                parent_commands[command_name] = CommandInfo(
                    module=".".join([package, module_name]), commands=None
                )
        else:
            commands = find_commands(name=package, path=path)

        # Get command description for help output
        def get_description(module):
            description = None
            if subcommands and module.startswith(package + "."):
                description = subcommands.get(module[len(package) + 1 :])
            if description is None:
                module = import_module(module)
                description = module.__doc__
                parser_fn = getattr(module, "parser", None)
                if parser_fn is not None:
                    description = parser_fn(add_help=False).description
            return description or ""

        # Add subparsers, while avoiding import of unused CLIs
        def add_commands(subparsers, commands, argv):
            if not argv:
                return
            if argv[0] in commands:
                command_name = argv[0]
                command_info = commands[command_name]
                module = import_module(command_info.module)
                if command_info.commands is None:
                    parser_fn = getattr(module, "parser")
                    parser = parser_fn(add_help=False)
                    func = getattr(module, "func")
                    func = _func_wrapper(func=func, init=getattr(module, "init", None))
                    subparsers.add_parser(
                        command_name,
                        parents=[parser, default_options_parser],
                        help=parser.description,
                    ).set_defaults(func=func)
                else:
                    add_commands(
                        subparsers.add_parser(
                            command_name, parents=[default_options_parser], help=module.__doc__
                        ).add_subparsers(),
                        command_info.commands,
                        argv[1:],
                    )
            else:
                for command_name, command_info in commands.items():
                    subparsers.add_parser(
                        command_name,
                        parents=[default_options_parser],
                        help=get_description(command_info.module),
                    )

        add_commands(subparsers, commands, argv)

        # Parse command arguments and call requested subcommand function
        args, unknown = mainparser.parse_known_args(argv)

        if hasattr(args, "func"):
            try:
                exit_code = args.func(args, unknown)
            except KeyboardInterrupt:
                sys.stderr.write("Interrupted by user\n")
                exit_code = 1
        else:
            mainparser.print_usage()
            exit_code = 1

        return exit_code

    # Return closure
    return main


def main_func(
    parser: ParserCallable,
    func: Callable[[ParsedArguments], int],
    init: Callable[[ParsedArguments], int] = None,
) -> MainCallable:
    r"""Create main function of subcommand.

    Args:
        parser (callable): Builder function of ``ArgumentParser`` object.
        func (callable): Function accepting ``ParsedArguments`` and returning exit code.
        init (callable): Function accepting ``ParsedArguments`` and returning exit code.
            This initialization function is called before ``func`` if provided.

    Returns:
        main (callable): Main entry point of subcommand which takes a list
            of command arguments as ``argv`` argument and returns an exit code.

    """

    func = _func_wrapper(func, init=init)

    def main(argv: Optional[List[Any]] = None) -> int:
        r"""Call main function with parsed CLI arguments.

        Args:
            argv (list): Command-line arguments. If None, sys.argv[1:] is used.

        Returns:
            Exit code, zero on success. Note that not every CLI must return an exit code,
            but throw an exception in case of an error instead. If the exception is not caught,
            it will result in a non-zero exit code of the Python interpreter command.
            Use the exit code return value for different success (or error) codes if needed.

        """
        argv = None if argv is None else [str(arg) for arg in argv]
        try:
            p = parser(add_help=True)
            args = p.parse_args(argv)
            exit_code = func(args)
        except KeyboardInterrupt:
            sys.stderr.write("Interrupted by user\n")
            exit_code = 1
        return exit_code

    # Return closure
    return main


def _func_wrapper(
    func: Callable[[ParsedArguments], int], init: Callable[[ParsedArguments], int] = None
) -> Callable[[ParsedArguments], int]:
    r"""Wrap command 'init' and 'func' callables."""

    nparams = len(inspect.signature(func).parameters)
    assert nparams == 1 or nparams == 2, "func() must have one or two parameters"

    def call_init_then_func(args: ParsedArguments, unknown: UnknownArguments = []) -> int:
        exit_code = 0
        if nparams == 1 and unknown:
            raise RuntimeError(f"Unparsed arguments: {unknown}")
        if init is not None:
            exit_code = init(args)
        if exit_code == 0:
            if nparams == 2:
                exit_code = func(args, unknown)
            else:
                exit_code = func(args)
        return exit_code

    return call_init_then_func


def _pypi_package_name(module: str) -> str:
    r"""Get Python package name given CLI module __name__."""
    parts = []
    for part in module.split("."):
        if part in ("app", "apps", "cli"):
            break
        parts.append(part)
    return ".".join(parts)


def bool_from_arg(arg: BoolStr) -> bool:
    r"""Convert string argument to boolean value."""
    if arg in TRUE:
        return True
    if arg in FALSE:
        return False
    raise ValueError("bool_from_arg() 'arg' must be valid boolean string argument")


def memory_from_arg(arg: Union[int, str, None]) -> int:
    r"""Parse --memory option argument.

    Args:
        arg: Amount of memory. If None, returns the total available memory.

    Returns:
        Amount of memory in number of bytes.

    """
    if arg is None:
        return memory_limit()
    if isinstance(arg, int):
        return arg
    if not isinstance(arg, str):
        raise TypeError("memory_from_arg() 'arg' must be int or str")
    if arg.endswith("B"):
        return int(arg[:-1])
    if arg.endswith("K"):
        return int(float(arg[:-1]) * 1024)
    if arg.endswith("M"):
        return int(float(arg[:-1]) * 1024**2)
    if arg.endswith("G"):
        return int(float(arg[:-1]) * 1024**3)
    return int(arg)
