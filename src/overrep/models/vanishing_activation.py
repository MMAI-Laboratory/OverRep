import math

import torch
import torch.nn as nn


def run_module_function(model, total_steps=None, warmup_steps_ratio=None, patient_steps_ratio=None,
                        step=False, current_step=None):
    for m in model.modules():
        if isinstance(m, VanishingActivation):
            if total_steps is not None:
                m.set_total_steps(total_steps, warmup_steps_ratio, patient_steps_ratio)
            if current_step is not None:
                m.set_current_step(current_step)
            if step:
                m.va_step()


def va_step(model):
    for m in model.modules():
        if isinstance(m, VanishingActivation):
            m.va_step()


class VanishingActivation(nn.Module):
    """Annealed activation `act(x) * (1 - alpha) + x * alpha`; alpha reaches 1 before training ends."""

    def __init__(self, total_steps=None, warmup_steps_ratio=0.01, patient_steps_ratio=0.2, act_fn=nn.SiLU):
        super().__init__()
        self.act = act_fn()

        self.total_steps = total_steps
        self.warmup_steps_ratio = warmup_steps_ratio
        self.patient_steps_ratio = patient_steps_ratio
        self.current_step = 0

        self.register_buffer('alpha_table', None)

    def _build_schedule(self):
        """warmup: alpha 1 -> 0 (linear), cosine: alpha 0 -> 1, patient: alpha = 1."""
        assert self.warmup_steps_ratio + self.patient_steps_ratio < 1.0, \
            "warmup_steps_ratio + patient_steps_ratio must be < 1.0"

        warmup_steps = int(self.total_steps * self.warmup_steps_ratio)
        patient_steps = int(self.total_steps * self.patient_steps_ratio)
        cosine_steps = self.total_steps - warmup_steps - patient_steps

        if warmup_steps > 1:
            steps = torch.arange(warmup_steps, dtype=torch.float32)
            warmup_phase = 1.0 - steps / (warmup_steps - 1)
        elif warmup_steps == 1:
            warmup_phase = torch.tensor([1.0])
        else:
            warmup_phase = torch.tensor([])

        if cosine_steps > 1:
            steps = torch.arange(cosine_steps, dtype=torch.float32)
            cosine_phase = 0.5 * (1 - torch.cos(math.pi * steps / (cosine_steps - 1)))
        elif cosine_steps == 1:
            cosine_phase = torch.tensor([0.0])
        else:
            cosine_phase = torch.tensor([])

        patient_phase = torch.ones(patient_steps)

        self.alpha_table = torch.cat([warmup_phase, cosine_phase, patient_phase])

        self.alpha_table = torch.clamp(self.alpha_table, 0.0, 1.0)

    def va_step(self):
        if self.current_step < self.total_steps - 1:
            self.current_step += 1

    def forward(self, x):
        alpha = self.alpha
        return self.act(x) * (1. - alpha) + x * alpha

    def set_total_steps(self, total_steps, warmup_steps_ratio=None, patient_steps_ratio=None):
        self.total_steps = total_steps
        if warmup_steps_ratio is not None:
            self.warmup_steps_ratio = warmup_steps_ratio
        if patient_steps_ratio is not None:
            self.patient_steps_ratio = patient_steps_ratio
        self.current_step = 0
        self._build_schedule()

    def set_current_step(self, current_step):
        self.current_step = current_step

    @property
    def alpha(self):
        if self.current_step >= self.alpha_table.size(0):
            alpha = 1.0
        else:
            alpha = self.alpha_table[self.current_step]
        return alpha
