#!/usr/bin/env python3
"""Add a GPT-2-compatible weights checkpoint writer to upstream train_gpt2.c."""
from pathlib import Path
import sys

path = Path(sys.argv[1] if len(sys.argv) > 1 else "train_gpt2.c")
source = path.read_text()

function = r"""
// Save updated model weights in the same binary format used by gpt2_build_from_checkpoint.
static void gpt2_save_weights(const GPT2 *model, const char *checkpoint_path) {
    int header[256] = {0};
    header[0] = 20240326;
    header[1] = 3;
    header[2] = (int)model->config.max_seq_len;
    header[3] = (int)model->config.vocab_size;
    header[4] = (int)model->config.num_layers;
    header[5] = (int)model->config.num_heads;
    header[6] = (int)model->config.channels;
    header[7] = (int)model->config.padded_vocab_size;

    FILE *file = fopen(checkpoint_path, "wb");
    if (file == NULL) {
        perror("Could not open checkpoint for writing");
        exit(1);
    }
    if (fwrite(header, sizeof(int), 256, file) != 256 ||
        fwrite(model->params_memory, sizeof(float), model->num_parameters, file) != model->num_parameters) {
        perror("Could not write model checkpoint");
        fclose(file);
        exit(1);
    }
    if (fclose(file) != 0) {
        perror("Could not close model checkpoint");
        exit(1);
    }
    printf("Saved trained model weights: %s (%zu parameters)\\n", checkpoint_path, model->num_parameters);
}

"""

if "static void gpt2_save_weights(" in source:
    raise SystemExit("Checkpoint writer already appears to be installed; refusing to patch twice.")
if "int main() {" not in source:
    raise SystemExit("Could not find main() insertion point.")
source = source.replace("int main() {", function + "int main() {", 1)

needle = """    // free
    dataloader_free(&train_loader);"""
replacement = """    // Save the fine-tuned weights in the original GPT-2 checkpoint format.
    gpt2_save_weights(&model, "gpt2_124M_finetuned.bin");

    // free
    dataloader_free(&train_loader);"""
if needle not in source:
    raise SystemExit("Could not find training cleanup insertion point.")
source = source.replace(needle, replacement, 1)
path.write_text(source)
print(f"Patched {path}; checkpoint will be gpt2_124M_finetuned.bin")
