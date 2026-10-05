# Third-party notices

This repository includes code adapted from the projects below.

| Component | Source | Used in | License |
| --- | --- | --- | --- |
| Transformers | https://github.com/huggingface/transformers | `src/overrep/common/utils.py` (cosine schedule), `src/overrep/models/reparam_model.py` (decoder layers), `src/overrep/models/reparam_module_qwen.py` (attention) | Apache-2.0 |
| LM Evaluation Harness | https://github.com/EleutherAI/lm-evaluation-harness | `src/overrep/eval/harness.py` (`task_manage`) | MIT |

## Apache License 2.0 (Transformers)

Copyright The HuggingFace Team. Licensed under the Apache License, Version 2.0; the full text is in
[LICENSE](LICENSE). The adapted code has been modified for this project.

## MIT License (LM Evaluation Harness)

```text
Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
