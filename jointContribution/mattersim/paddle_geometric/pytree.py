import paddle


# Custom implementation of pytree
class pytree:
    @staticmethod
    def tree_map(fn, data):
        """
        Recursively traverses a nested data structure (e.g., list, tuple, dict),
        applying a given function `fn` to each element.

        Args:
            fn (Callable): The function to apply to each element in the structure.
            data (Any): The nested data structure to traverse (can be a list, tuple, dict, or leaf node).

        Returns:
            Any: A new data structure of the same type as `data`, with `fn` applied to each element.
        """
        # If the current item is a list, apply the function to each element recursively.
        if isinstance(data, list):
            return [pytree.tree_map(fn, item) for item in data]

        # If the current item is a tuple, apply the function to each element recursively.
        elif isinstance(data, tuple):
            return tuple(pytree.tree_map(fn, item) for item in data)

        # If the current item is a dictionary, apply the function to each value recursively.
        elif isinstance(data, dict):
            return {key: pytree.tree_map(fn, value) for key, value in data.items()}

        # If the current item is a leaf (not a list, tuple, or dict), apply the function directly.
        else:
            return fn(data)

    @staticmethod
    def tree_flatten(data):
        """
        Flattens a nested data structure into a 1D list of leaf nodes.

        Args:
            data (Any): The nested data structure to flatten.

        Returns:
            List: A list of all the leaf elements in the input structure.
        """
        if isinstance(data, list):
            flattened = []
            for item in data:
                flattened.extend(pytree.tree_flatten(item))
            return flattened

        elif isinstance(data, tuple):
            flattened = []
            for item in data:
                flattened.extend(pytree.tree_flatten(item))
            return flattened

        elif isinstance(data, dict):
            flattened = []
            for value in data.values():
                flattened.extend(pytree.tree_flatten(value))
            return flattened

        # If the current item is a leaf, return it as a list.
        else:
            return [data]  # Leaf node (e.g., a paddle.Tensor)

    @staticmethod
    def tree_unflatten(flattened_data, structure):
        """
        Reconstructs the original nested data structure from the flattened list.

        Args:
            flattened_data (List): A 1D list of leaf nodes to unflatten.
            structure (Any): The original structure of the data (e.g., list, tuple, dict) to guide the unflattening.

        Returns:
            Any: A reconstructed nested data structure matching the original structure, with elements filled in.
        """
        # If the structure is a list, reconstruct by splitting the flattened data accordingly.
        if isinstance(structure, list):
            size = len(structure)
            return [pytree.tree_unflatten(flattened_data[i:i + size], item) for i, item in enumerate(structure)]

        # If the structure is a tuple, reconstruct by splitting the flattened data accordingly.
        elif isinstance(structure, tuple):
            size = len(structure)
            return tuple(pytree.tree_unflatten(flattened_data[i:i + size], item) for i, item in enumerate(structure))

        # If the structure is a dictionary, reconstruct by splitting the flattened data accordingly.
        elif isinstance(structure, dict):
            keys = list(structure.keys())
            size = len(structure)
            return {key: pytree.tree_unflatten(flattened_data[i:i + size], value) for i, (key, value) in
                    enumerate(structure.items())}

        # If the structure is a leaf, return the single value from the flattened list.
        else:
            return flattened_data[0]