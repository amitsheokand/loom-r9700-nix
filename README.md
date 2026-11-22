# Loom on an R9700: Inkling 2 Mini, 4 slots x 80K, drafter and vision

A NixOS module that serves Inkling 2 Mini through AMD Loom (`hrx-loom` 0.2.0) on a
Radeon AI PRO R9700 (RDNA4, gfx1201), as an OpenAI-compatible API on `127.0.0.1:8000`.

What it adds on top of the upstream Loom package:

- The four custom flash2 attention kernels ported from RDNA3 to dual-target RDNA3 + RDNA4,
  compile-verified at Nix build time (`loom-r9700/port-inkling-rdna4.py`).
- The built-in DSpark speculative drafter (`wire-inkling-dspark.py`, weights converted by
  `convert-dspark-gguf.py`).
- The built-in vision encoder (`wire-inkling-vision.py`).
- Preset `chat-r9700`: 4 connections x 80K context (320K shared pool), about 30.5 GiB peak VRAM.
- A sandboxed systemd unit: dynamic user, loopback only, GPU device nodes only.

## Use

```nix
{
  imports = [ ./loom-recipe/module.nix ];
  services.loom-r9700 = {
    enable = true;
    modelDir = "/var/lib/loom-r9700";
    owner = "youruser";
  };
}
```

Set `pci_slot` and `hip_device` in `loom.toml` for your card, then download the three
weight files into `modelDir` (the commands are in the header of `module.nix`; the unit
prints the missing one if it cannot start).

Check it: `curl -s http://127.0.0.1:8000/v1/models`.

## Notes

- x86_64 Linux only; the `hrx-loom` binaries come from the upstream PyPI wheel.
- The unit refuses to start while another process holds more than 2 GiB VRAM on the card.
- Model and Loom package revisions are pinned in `loom.toml` and by hash in `module.nix`.

## Licence

MIT, see `LICENSE`. It covers this module and the scripts only. `hrx-loom`, the upstream
Loom model package and the model weights are downloaded at build or setup time and keep
their own licences.
