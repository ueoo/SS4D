from .lora_mixin import LoRAMixin
from .sparse_structure_flow_4d import SparseStructureFlow4DModel
from .structured_latent_flow_4d import ElasticSLatFlow4DModel, SLatFlow4DModel


class LoRASparseStructureFlow4DModel(LoRAMixin, SparseStructureFlow4DModel):
    pass


class LoRASLatFlow4DModel(LoRAMixin, SLatFlow4DModel):
    pass


class LoRAElasticSLatFlow4DModel(LoRAMixin, ElasticSLatFlow4DModel):
    pass
