# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Context Manager
from pathlib import Path
from typing import Optional, Dict, Any
import os

from synthetic_data_kit.utils.config import DEFAULT_CONFIG_PATH

class AppContext:
    """Context manager for global app state"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize app context"""
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {}
        
        # Ensure data directories exist
        self._ensure_data_dirs()
        
    # Why have separeate folders? Yes ideally you should just be able to ingest an input folder and have everything being ingested and converted BUT
    # Managing context window is hard and there are more edge cases which needs to be handled carefully
    # it's also easier to debug in alpha if we have multiple files. 
    def _ensure_data_dirs(self):
        """Ensure data directories exist"""
        dirs = [
            "data/pdf",
            "data/html",
            "data/youtube",
            "data/docx",
            "data/ppt",
            "data/txt",
            "data/output",
            "data/generated",
            "data/cleaned",
            "data/final",
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)