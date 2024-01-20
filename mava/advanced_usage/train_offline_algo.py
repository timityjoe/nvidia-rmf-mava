


# from flashbax.vault import Vault
from og_marl_tjt.og_marl.jax.systems.maicq_vault2 import train_maicq_system
from og_marl_tjt.og_marl.loggers import TerminalLogger, WandbLogger

# Download the dataset
# flashbax_vault = Vault(
#     vault_name="ff_ippo_rware",
#     vault_uid="20240120165454",
# )
# buffer_state = flashbax_vault.read()

# Instantiate environment for evaluation
# env = SMACv1("8m")

# Setup a logger to write to terminal
# logger = TerminalLogger()
logger = WandbLogger()

# Train system
train_maicq_system(logger)