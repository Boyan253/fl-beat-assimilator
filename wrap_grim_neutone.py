"""Wrap our exported RAVE GRIM model into a Neutone FX model (.nm) so it loads in Neutone FX inside FL.
Based on neutone_sdk's official example_rave.py. Run with the rave-env python."""
import logging, os
from pathlib import Path
from typing import Dict, List
import torch
from torch import Tensor
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter, ContinuousNeutoneParameter
from neutone_sdk.utils import save_neutone_model

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

IN_TS = "/opt/rave_data/GRIM/runs/GRIM_melody_0fd60b4ddc/GRIM_melody_0fd60b4ddc_streaming.ts"
OUT_DIR = "/opt/rave_data/GRIM/neutone_export"
OUT_NAME = "GRIM_melody"


class RAVEModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str: return "GRIM.melody"
    def get_model_authors(self) -> List[str]: return ["Boyan"]
    def get_model_short_description(self) -> str: return "GRIM phonk melody timbre (RAVE)."
    def get_model_long_description(self) -> str: return "RAVE timbre-transfer model trained on the GRIM phonk melody. Feed it a tone/melody, it re-voices it in GRIM's sound."
    def get_technical_description(self) -> str: return "RAVE v2 trained locally on the 7900 XTX on the GRIM melody stem."
    def get_technical_links(self) -> Dict[str, str]: return {"Code": "https://github.com/acids-ircam/RAVE"}
    def get_tags(self) -> List[str]: return ["timbre transfer", "RAVE", "phonk", "GRIM"]
    def get_model_version(self) -> str: return "1.0.0"
    def is_experimental(self) -> bool: return True
    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            ContinuousNeutoneParameter(name="Chaos", description="Magnitude of latent noise", default_value=0.0),
            ContinuousNeutoneParameter(name="Z edit index", description="Index of latent dimension to edit", default_value=0.0),
            ContinuousNeutoneParameter(name="Z scale", description="Scale of latent variable", default_value=0.5),
            ContinuousNeutoneParameter(name="Z offset", description="Offset of latent variable", default_value=0.5),
        ]
    def is_input_mono(self) -> bool: return True
    def is_output_mono(self) -> bool: return True
    def get_native_sample_rates(self) -> List[int]: return [44100]
    def get_native_buffer_sizes(self) -> List[int]: return [2048]
    def calc_model_delay_samples(self) -> int: return 2048

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> Tensor:
        z = self.model.encode(x.unsqueeze(1))
        noise_amp = params["Chaos"]
        z = torch.randn_like(z) * noise_amp + z
        idx_z = int(torch.clamp(params["Z edit index"], min=0.0, max=0.99) * self.model.latent_size)
        z_scale = params["Z scale"] * 2
        z_offset = params["Z offset"] * 2 - 1
        z[:, idx_z] = z[:, idx_z] * z_scale + z_offset
        out = self.model.decode(z)
        out = out.squeeze(1)
        return out


if __name__ == "__main__":
    model = torch.jit.load(IN_TS)
    wrapper = RAVEModelWrapper(model)
    save_neutone_model(wrapper, Path(OUT_DIR) / OUT_NAME, freeze=False, dump_samples=False, submission=False)
    print("WRAPPED OK ->", str(Path(OUT_DIR) / OUT_NAME), flush=True)
