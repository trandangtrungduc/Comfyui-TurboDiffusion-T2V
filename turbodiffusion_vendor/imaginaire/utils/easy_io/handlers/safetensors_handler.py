# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SafeTensors file handler for easy_io."""

from pathlib import Path
from typing import Any

from imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class SafeTensorsHandler(BaseFileHandler):
    """Handler for safetensors files."""

    str_like = False

    def load_from_fileobj(self, file, **kwargs):
        """Load from file object.

        Args:
            file: File object or file path
            **kwargs: Additional arguments

        Returns:
            Dictionary of tensors loaded from the file
        """
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors package is required to load .safetensors files.\n"
                "Install with: pip install safetensors"
            )

        # Filter out kwargs that safetensors doesn't support
        # safetensors.load_file only accepts: device (optional)
        # Remove torch-specific kwargs like map_location, weights_only, etc.
        safetensors_kwargs = {}
        if 'device' in kwargs:
            safetensors_kwargs['device'] = kwargs['device']
        # Map torch's map_location to safetensors' device parameter
        elif 'map_location' in kwargs:
            safetensors_kwargs['device'] = kwargs['map_location']

        # If file is a file object with a name attribute, use the path
        if hasattr(file, 'name'):
            return load_file(file.name, **safetensors_kwargs)
        # If file is already a path string
        elif isinstance(file, (str, Path)):
            return load_file(str(file), **safetensors_kwargs)
        else:
            # Read the entire file into memory and load from bytes
            # This is less efficient but works for file-like objects
            file_bytes = file.read()
            from safetensors import safe_open
            import torch

            # Create a temporary file to load from
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                result = load_file(tmp_path, **safetensors_kwargs)
            finally:
                import os
                os.unlink(tmp_path)

            return result

    def dump_to_fileobj(self, obj, file, **kwargs):
        """Dump to file object.

        Args:
            obj: Dictionary of tensors to save
            file: File object or file path
            **kwargs: Additional arguments
        """
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError(
                "safetensors package is required to save .safetensors files.\n"
                "Install with: pip install safetensors"
            )

        # If file is a file object with a name attribute, use the path
        if hasattr(file, 'name'):
            save_file(obj, file.name, **kwargs)
        # If file is already a path string
        elif isinstance(file, (str, Path)):
            save_file(obj, str(file), **kwargs)
        else:
            raise NotImplementedError("SafeTensors can only save to file paths, not file objects")

    def dump_to_str(self, obj, **kwargs):
        """Dump to string (not supported for safetensors)."""
        raise NotImplementedError("SafeTensors dumping to string is not supported")
