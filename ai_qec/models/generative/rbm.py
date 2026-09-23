"""Joint error/syndrome restricted Boltzmann machine of Torlai & Melko (2017).

The visible layer is ``v = [e | S]``. With ``weight = [W | U]`` and
``visible_bias = [b | d]`` the energy of Eq. (2) is

    E(e, S, h) = -h.W e - h.U S - b.e - c.h - d.S

and the free energy after summing out ``h`` is
``F(v) = -v.visible_bias - sum_i softplus(c_i + (weight v)_i)``.
"""

from collections.abc import Mapping

import torch

from ai_qec.registry.validation import ConfigurationError
from ai_qec.models.spec import ModelSpec
from ai_qec.registry.catalog import MODELS
from ai_qec.utils.hashing import sha256_json

FAMILY_ID = "joint-error-syndrome-rbm"
ARCHITECTURE_VERSION = "1"
PAYLOAD_SCHEMA = "joint-rbm-model-payload-v1"


class JointErrorSyndromeRBM(torch.nn.Module):
    """Paper Fig. 2: syndrome layer S and error layer e, both fully connected to h."""

    def __init__(
        self, num_error_units: int, num_syndrome_units: int, num_hidden_units: int
    ) -> None:
        super().__init__()
        self.num_error_units = num_error_units
        self.num_syndrome_units = num_syndrome_units
        self.num_hidden_units = num_hidden_units
        visible = num_error_units + num_syndrome_units
        self.weight = torch.nn.Parameter(torch.zeros(num_hidden_units, visible))
        self.visible_bias = torch.nn.Parameter(torch.zeros(visible))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(num_hidden_units))

    def reset_parameters(self, init_std: float, generator) -> None:
        with torch.no_grad():
            self.weight.normal_(0.0, init_std, generator=generator)
            self.visible_bias.zero_()
            self.hidden_bias.zero_()

    def hidden_logits(self, visible):
        return torch.nn.functional.linear(visible, self.weight, self.hidden_bias)

    def hidden_probability(self, visible):
        return torch.sigmoid(self.hidden_logits(visible))

    def visible_probability(self, hidden):
        return torch.sigmoid(hidden @ self.weight + self.visible_bias)

    def sample_hidden(self, visible, generator):
        return torch.bernoulli(self.hidden_probability(visible), generator=generator)

    def sample_visible(self, hidden, generator):
        return torch.bernoulli(self.visible_probability(hidden), generator=generator)

    def free_energy(self, visible):
        return -(visible @ self.visible_bias) - torch.nn.functional.softplus(
            self.hidden_logits(visible)
        ).sum(-1)

    def reconstruction_bce(self, visible):
        """Mean-field one-step reconstruction cross entropy per visible unit."""

        reconstruction = self.visible_probability(self.hidden_probability(visible))
        return torch.nn.functional.binary_cross_entropy(reconstruction, visible)

    @property
    def error_weight(self):
        return self.weight[:, : self.num_error_units]

    @property
    def syndrome_weight(self):
        return self.weight[:, self.num_error_units :]

    @property
    def error_bias(self):
        return self.visible_bias[: self.num_error_units]


class JointErrorSyndromeRBMFamily:
    """Creates, serializes, and decodes with the joint ``[e | S]`` RBM."""

    family_id = FAMILY_ID
    input_fields = ("physical_errors", "detector_events")

    def parameters(self, spec: ModelSpec) -> dict[str, object]:
        if spec.family != FAMILY_ID:
            raise ConfigurationError(f"[model.family] expected {FAMILY_ID!r}, got {spec.family!r}")
        if spec.architecture_version != ARCHITECTURE_VERSION:
            raise ConfigurationError(
                f"[model.architecture_version] {FAMILY_ID} supports {ARCHITECTURE_VERSION!r}, "
                f"got {spec.architecture_version!r}"
            )
        unknown = sorted(set(spec.parameters) - {"hidden_units", "init_std"})
        if unknown:
            raise ConfigurationError(f"[model.parameters] unknown keys: {unknown}")
        hidden = spec.parameters.get("hidden_units")
        init_std = spec.parameters.get("init_std")
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ConfigurationError(
                f"[model.parameters.hidden_units] must be a positive integer, got {hidden!r}"
            )
        if isinstance(init_std, bool) or not isinstance(init_std, (int, float)) or not init_std > 0:
            raise ConfigurationError(
                f"[model.parameters.init_std] must be a positive number, got {init_std!r}"
            )
        return {"hidden_units": hidden, "init_std": float(init_std)}

    def decoding_parameters(self, decoding: Mapping[str, object]) -> dict[str, int]:
        unknown = sorted(set(decoding) - {"burn_in", "max_steps"})
        if unknown:
            raise ConfigurationError(f"[model.decoding] unknown keys: {unknown}")
        values = {}
        for name in ("burn_in", "max_steps"):
            value = decoding.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigurationError(
                    f"[model.decoding.{name}] must be a non-negative integer, got {value!r}"
                )
            values[name] = value
        if values["max_steps"] <= values["burn_in"]:
            raise ConfigurationError("[model.decoding] max_steps must exceed burn_in")
        return values

    def identity(self, spec: ModelSpec, widths: Mapping[str, int]) -> str:
        return sha256_json(
            {
                "family": FAMILY_ID,
                "architecture_version": ARCHITECTURE_VERSION,
                "parameters": self.parameters(spec),
                "widths": dict(widths),
            }
        )

    def create(self, spec: ModelSpec, *, widths: Mapping[str, int], seed: int):
        parameters = self.parameters(spec)
        model = JointErrorSyndromeRBM(
            widths["physical_errors"], widths["detector_events"], parameters["hidden_units"]
        )
        model.reset_parameters(parameters["init_std"], torch.Generator().manual_seed(seed))
        return model

    def visible(self, tensors: Mapping[str, object]):
        return torch.cat([tensors["physical_errors"], tensors["detector_events"]], dim=1).float()

    def payload(
        self, model, spec: ModelSpec, *, widths: Mapping[str, int], dataset_artifact_id: str
    ) -> dict:
        return {
            "schema_version": PAYLOAD_SCHEMA,
            "family": FAMILY_ID,
            "architecture_version": ARCHITECTURE_VERSION,
            "parameters": self.parameters(spec),
            "widths": dict(widths),
            "model_identity": self.identity(spec, widths),
            "dataset_artifact_id": dataset_artifact_id,
            "state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            },
        }

    def load(self, payload: Mapping[str, object], *, device: str):
        if payload.get("schema_version") != PAYLOAD_SCHEMA or payload.get("family") != FAMILY_ID:
            raise ValueError("checkpoint is not a joint-error-syndrome-rbm model payload")
        widths = payload["widths"]
        model = JointErrorSyndromeRBM(
            widths["physical_errors"],
            widths["detector_events"],
            payload["parameters"]["hidden_units"],
        )
        model.load_state_dict(payload["state_dict"])
        return model.to(device).eval()

    def build_decoder(
        self, model, *, code, decoding: Mapping[str, object], pipeline, device: str, seed: int
    ):
        from ai_qec.models.decoders.generative.rbm_gibbs import RBMGibbsDecoder

        values = self.decoding_parameters(decoding)
        return RBMGibbsDecoder(
            decoder_id=FAMILY_ID,
            model=model,
            code=code,
            burn_in=values["burn_in"],
            max_steps=values["max_steps"],
            pipeline=pipeline,
            device=device,
            seed=seed,
        )


@MODELS.register(FAMILY_ID)
def build_joint_rbm_family() -> JointErrorSyndromeRBMFamily:
    return JointErrorSyndromeRBMFamily()
