# macOS quickstart without Homebrew or CMake

CortexLite's native C11/C++17 inference engine can build with the Apple Command Line Tools alone. The project does not require an API key, GPU, Docker, Homebrew, or CMake for this path.

## Requirements

```bash
xcode-select -p
cc --version
c++ --version
python3 --version
```

If `cc`/`c++` are missing, install Apple's Command Line Tools through `xcode-select --install` and follow the macOS installation dialog. Check the commands again afterward.

## Clone, build and run

```bash
git clone https://github.com/naim-munshi/cortexlite-ai-engine.git
cd cortexlite-ai-engine
bash scripts/showcase.sh
```

- With CMake available, the showcase uses CMake and CTest, plus the independent Python reference check.
- Without CMake, it automatically calls `scripts/build-native.sh` and `scripts/test-native.sh`. Native artifacts go under `build-native/` and do not interfere with CMake artifacts in `build/`.
- Force the fallback even when CMake is installed: `CORTEXLITE_FORCE_NATIVE=1 bash scripts/showcase.sh`.
- Each route runs the included native checks and prints an example prediction/evaluation and synthetic nearest-neighbor benchmark. The optional PyTorch parity check remains separate.

## Homebrew SSL certificate error

If `brew install cmake` fails with `SSL certificate problem: self signed certificate in certificate chain`, you **do not need to fix Homebrew to run CortexLite**. The error concerns your local system/network certificate trust chain, potentially a campus/office HTTPS-inspecting proxy, VPN, or missing trusted certificate. Using a trusted mobile hotspot can help isolate whether the network is involved.

Do **not** work around this by disabling TLS verification, using `curl -k`, or setting `HOMEBREW_CURLRC` to insecure options. Consult your network administrator or repair the local trust configuration if you need Homebrew later.

## Troubleshooting

| Symptom | Action |
|---|---|
| `cmake: command not found` | Use `bash scripts/showcase.sh`; it detects and uses native compilation |
| `c++: command not found` | Install Xcode Command Line Tools |
| `python3: command not found` | Install a supported Python 3 interpreter before running the bundled validation scripts |
| Compiled binaries not found | Run `bash scripts/build-native.sh`, then `bash scripts/test-native.sh` |
| macOS security warning | Do not bypass Gatekeeper to run untrusted binaries. Compile locally from reviewed source instead |

For measured benchmark caveats, see [benchmark methodology](benchmarks.md).
