# AMD Loom (hrx-loom 0.2.0, RDNA4 gfx1201) serving Inkling 2 Mini on a Radeon AI PRO R9700.
#
# Model, preset, port, slots and per-slot context come from ./loom.toml; this module
# packages hrx-loom + an RDNA4-ported Inkling 2 Mini Loom program declaratively and
# wires them to a native systemd service.
#
# - OpenAI-compatible API on 127.0.0.1:<port> only (never 0.0.0.0).
# - GGUF weights live in services.loom-r9700.modelDir (not in the Nix store):
#     hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf \
#       --revision ea0c79b2ecd73669e774c8519b79bfabfddbc027 --local-dir <modelDir>
#     hf download unsloth/Inkling-2-Mini-GGUF mmproj-F16.gguf \
#       --revision ae9327c5fbc7333116ba32eee78bb5d2bbefae4f --local-dir <modelDir>
#     hf download thinkingmachines/Inkling-2-Mini mtp.safetensors \
#       --revision b85c3cfa615fc413cc9af3379f491b963f3b586a, then convert it with
#       loom-r9700/convert-dspark-gguf.py into <modelDir>/Inkling-2-Mini-DSpark-Q8_0.gguf
# - Refuses to start while another process holds >2 GiB VRAM on the card and retries
#   every minute.
# - Ported & tuned on an R9700 (2026-11-21):
#   4 custom flash2 attention kernels ported from RDNA3 (gfx11) to dual-target RDNA3+RDNA4
#   (gfx12) WMMA fragment layouts and compile-verified to HSACO at Nix build time.
#   Preset chat-r9700: 4 slots x 80K (context_capacity=327680), prefill_capacity=2048,
#   max_tokens=16384, draft_depth=3 (DSpark), vision (inkling2v mmproj-F16.gguf),
#   --pool_backing=fixed, --parameter_mode=preload (~30.5 GiB peak VRAM).
{ config, lib, pkgs, ... }:

let
  cfg = config.services.loom-r9700;
  d = builtins.fromTOML (builtins.readFile ./loom.toml);
  b = d.backend;
  scratch = cfg.modelDir;
  totalCtx = b.slots * b.context_per_slot;

  # Prebuilt native binaries from the official hrx-loom PyPI wheel (x86_64).
  # Zero ROCm/Python runtime dependencies: loom-serve talks directly to /dev/kfd + /dev/dri
  # via embedded libamdf in libiree.so and links only glibc + libstdc++/libgcc_s.
  hrxLoom = pkgs.stdenvNoCC.mkDerivation {
    pname = "hrx-loom";
    version = b.hrx_loom_version;
    src = pkgs.fetchurl {
      url = "https://files.pythonhosted.org/packages/2a/d5/4bf52ac2c2f4c94530233d6dc19c7247462d9279731f6553e8de15e59d4a/hrx_loom-0.2.0-py3-none-manylinux_2_39_x86_64.whl";
      hash = "sha256-UUj546Jkr0AOxLrji9ewa/VR0ozDO3ph1n97zvpZBDg=";
    };
    nativeBuildInputs = [ pkgs.unzip pkgs.autoPatchelfHook ];
    buildInputs = [ pkgs.stdenv.cc.cc.lib ];
    dontUnpack = true;
    dontConfigure = true;
    dontBuild = true;
    installPhase = ''
      runHook preInstall
      mkdir -p work "$out/bin" "$out/lib"
      unzip -q "$src" "hrx_loom-${b.hrx_loom_version}.data/scripts/*" -d work
      scripts="work/hrx_loom-${b.hrx_loom_version}.data/scripts"
      install -m 0755 "$scripts"/lib*.so "$out/lib/"
      for f in "$scripts"/*; do
        case "$(basename "$f")" in
          lib*.so) continue ;;
        esac
        install -m 0755 "$f" "$out/bin/"
      done
      runHook postInstall
    '';
  };

  loomBase = "https://huggingface.co/${b.loom_repo}/resolve/${b.loom_rev}";
  fetchLoomFile = rel: hash: pkgs.fetchurl {
    url = "${loomBase}/${rel}";
    inherit hash;
  };

  upstreamFiles = {
    "model.json" = fetchLoomFile "model.json" "sha256-w0UZf2XlN/AtVBE1g7Zk+lWrYMP40JUq/dYDotIkwkU=";
    "llm_config.json" = fetchLoomFile "llm_config.json" "sha256-Y+YUPPISHy04D0T+KshQqA9FEf4PrlbYYDBodcLnCC4=";
    "tokenizer.json" = fetchLoomFile "tokenizer.json" "sha256-pF/cTn5moDXeI20mlHQv7VZFMnBRyqzl+DEgE6h5+bE=";
    "weights.loom" = fetchLoomFile "weights.loom" "sha256-+zyW4TEoP5949Fsv/3m9Oy6G8pLgEvHzfXUuFe2vyRY=";
    "program/control.loom" = fetchLoomFile "program/control.loom" "sha256-qxDvVBfy5kHFyYbbot9R2816E+YbVyX2HtA+DulWt3A=";
    "program/prepare.loom" = fetchLoomFile "program/prepare.loom" "sha256-rUP5gamlotUxMfO4Kl1Len3Ygu08OYdbZuAQcF52zps=";
    "program/service.loom" = fetchLoomFile "program/service.loom" "sha256-Xc0IOybZjzmzIV+fR7ms+qCOYt9AZxjNuSqgaEMj99s=";
    "program/storage.loom" = fetchLoomFile "program/storage.loom" "sha256-3RxgwXRZLk6jIlbe49n14rT+b/ncEFgV4j0HHVwzdi4=";
    "program/llm.loom" = fetchLoomFile "program/llm.loom" "sha256-vs1bm5n26oX1DwLNQZY9CG74/JHYQXCnKYo7t4JN2wU=";
  };

  # Dual-target (RDNA3 + RDNA4) Inkling 2 Mini Loom model package, compile-verified at build time.
  modelPkg = pkgs.runCommand "inkling-2-mini-loom-rdna4" {
    nativeBuildInputs = [ pkgs.python3 ];
  } ''
    mkdir -p src/program
    ${lib.concatStringsSep "\n" (lib.mapAttrsToList (rel: drv:
      "cp ${drv} src/${rel}"
    ) upstreamFiles)}
    python3 ${./loom-r9700}/port-inkling-rdna4.py \
      --src-dir src \
      --out-dir "$out" \
      --model-id "${d.model_id}" \
      --model-name "${b.name}" \
      --gguf-file "${b.model}" \
      --gguf-path "${scratch}/${b.model}" \
      --dspark-gguf-file "${b.dspark_model}" \
      --dspark-gguf-path "${scratch}/${b.dspark_model}" \
      --draft-depth "${toString b.draft_depth}" \
      --vision-gguf-file "${b.vision_model}" \
      --vision-gguf-path "${scratch}/${b.vision_model}" \
      --preset "${b.preset}" \
      --rows "${toString b.slots}" \
      --context-capacity "${toString totalCtx}" \
      --prefill-capacity 2048 \
      --max-tokens "${toString b.max_tokens}" \
      --default-temperature 1.0 \
      --default-top-p 0.95 \
      --loom-opt "${hrxLoom}/bin/loom-opt" \
      --loom-compile "${hrxLoom}/bin/loom-compile"
  '';

  vramGuard = pkgs.writeShellScript "loom-r9700-vram-guard" ''
    # Refuse to start while something else holds the card (PCI ${d.pci_slot}).
    for d in /sys/bus/pci/devices/${d.pci_slot}/mem_info_vram_used; do
      [ -r "$d" ] || continue
      used=$(cat "$d")
      limit=$((2 * 1024 * 1024 * 1024))
      if [ "$used" -gt "$limit" ]; then
        echo "loom-r9700: VRAM in use ($used bytes > 2 GiB); another backend still holds the card. Retrying." >&2
        exit 1
      fi
    done
    if [ ! -r "${scratch}/${b.model}" ]; then
      echo "loom-r9700: missing ${scratch}/${b.model}; run: hf download ${b.hf_repo} ${b.model} --revision ${b.hf_rev} --local-dir ${scratch}" >&2
      exit 1
    fi
    if [ ! -r "${scratch}/${b.dspark_model}" ]; then
      echo "loom-r9700: missing ${scratch}/${b.dspark_model}; download ${b.dspark_hf_repo}@${b.dspark_hf_rev}:mtp.safetensors and convert with loom-r9700/convert-dspark-gguf.py" >&2
      exit 1
    fi
    if [ ! -r "${scratch}/${b.vision_model}" ]; then
      echo "loom-r9700: missing ${scratch}/${b.vision_model}; run: hf download ${b.hf_repo} ${b.vision_model} --revision ${b.vision_hf_rev} --local-dir ${scratch}" >&2
      exit 1
    fi
  '';

  serve = pkgs.writeShellScript "loom-r9700-serve" ''
    set -euo pipefail
    ${hrxLoom}/bin/loom-serve \
      --model="${modelPkg}" \
      --preset="${b.preset}" \
      --listen="127.0.0.1:${toString d.port}" \
      --connections=${toString b.slots} \
      --held_requests=0 \
      --device="amdgpu://${toString d.hip_device}?completion_spin_ns=4294967295" \
      --pool_backing=fixed \
      --parameter_mode=preload \
      --heartbeat_ms=0 &
    loom_pid=$!
    trap 'kill -TERM "$loom_pid" 2>/dev/null || true; wait "$loom_pid" 2>/dev/null || true' TERM INT

    # Monitor model health: if the model enters phase "failed", exit 1 so systemd restarts the service.
    sleep 5
    while kill -0 "$loom_pid" 2>/dev/null; do
      if status=$(${pkgs.curl}/bin/curl -s -m 2 http://127.0.0.1:${toString d.port}/v1/status 2>/dev/null); then
        if echo "$status" | ${pkgs.gnugrep}/bin/grep -q '"phase":"failed"'; then
          echo "loom-r9700: model entered failed phase; terminating for systemd restart" >&2
          kill -TERM "$loom_pid" 2>/dev/null || true
          wait "$loom_pid" 2>/dev/null || true
          exit 1
        fi
      fi
      sleep 10
    done
    wait "$loom_pid"
  '';
in
{
  options.services.loom-r9700 = {
    enable = lib.mkEnableOption "AMD Loom serving Inkling 2 Mini on an R9700";
    modelDir = lib.mkOption {
      type = lib.types.path;
      example = "/var/lib/loom-r9700";
      description = "Directory holding the GGUF weights (about 18 GiB). Not in the Nix store.";
    };
    owner = lib.mkOption {
      type = lib.types.str;
      default = "root";
      description = "User that owns modelDir and downloads the weights into it.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.tmpfiles.rules = [
      "d ${scratch} 0755 ${cfg.owner} users -"
    ];

    systemd.services.loom-r9700 = {
      description = "${b.name} (127.0.0.1:${toString d.port})";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];
      unitConfig = {
        RequiresMountsFor = [ scratch ];
        StartLimitIntervalSec = 0;
      };
      serviceConfig = {
        # Prebuilt PyPI binary: never run as the human user; isolate from home, host
        # filesystem, and all non-loopback networking.
        DynamicUser = true;
        SupplementaryGroups = [ "video" "render" ];
        # Filesystem sandbox: root read-only, home hidden, the model dir bound read-only,
        # and a private ephemeral /tmp.
        ProtectSystem = "strict";
        ProtectHome = true;
        BindReadOnlyPaths = [ scratch ];
        PrivateTmp = true;
        # Network sandbox: loopback only; no LAN or internet.
        IPAddressDeny = "any";
        IPAddressAllow = "localhost";
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
        # Device & privilege sandbox: only GPU KFD/DRM nodes; drop all capabilities.
        DevicePolicy = "closed";
        DeviceAllow = [
          "/dev/kfd rw"
          "char-drm rw"
        ];
        NoNewPrivileges = true;
        CapabilityBoundingSet = "";
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectKernelLogs = true;
        ProtectControlGroups = true;
        ProtectClock = true;
        ProtectHostname = true;
        RestrictSUIDSGID = true;
        RestrictNamespaces = true;
        RestrictRealtime = true;
        LockPersonality = true;
        SystemCallArchitectures = "native";
        ExecStartPre = [ "${vramGuard}" ];
        ExecStart = "${serve}";
        Restart = "on-failure";
        RestartSec = "60";
        TimeoutStartSec = "300";
      };
    };
  };
}
