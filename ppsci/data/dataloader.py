"""Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ppsci.utils import logger


class InfiniteDataLoader(object):
    """A wrapper for infinite dataloader

    Args:
        dataloader (DataLoader): A finite dataloader to be wrapped.
    """

    def __init__(self, dataloader, cache=False):
        self.dataloader = dataloader
        self.cache = cache
        if self.cache:
            logger.info("Dataloader cache enabled")
        self.cache_data = None

    def __iter__(self):
        while True:
            if self.cache and self.cache_data is not None:
                yield self.cache_data
                continue
            dataloader_iter = iter(self.dataloader)
            for batch in dataloader_iter:
                yield batch
                if self.cache and self.cache_data is None:
                    self.cache_data = batch
