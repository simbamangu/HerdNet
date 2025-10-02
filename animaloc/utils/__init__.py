__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"

from .device import (
    get_device,
    is_gpu_available,
    get_device_type,
    setup_cudnn,
    get_memory_allocated,
    get_max_memory_allocated,
    empty_cache,
    supports_pinned_memory,
    get_autocast_context,
)