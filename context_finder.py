# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

from core import (
    models,
    files
)

InputFolder = 'input-DMKD-300'

current_folder = os.path.dirname(os.path.abspath(__file__))
print('current_folder:',current_folder)

# TODO: input_files = files.load_list
input_files = files.load_list(
        path=current_folder,
        folder=InputFolder #config['IO']['InputFolder']
    )
files_list=input_files

#input_file = current_folder + '/' +  InputFolder
print('input_files:', input_files)

def read_tokens(input_file):
        return list(
            files.read_file(
                filename=input_file.path
            )
        )

# read tokens
for file in files_list:
        print('file:', file)
        input_tokens = read_tokens(
            input_file=file
        )

print('input_tokens:', input_tokens)
