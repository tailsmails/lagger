# lagger

## Overview
lagger is a dynamic network latency, packet loss simulation, and side-channel behavioral analysis proxy designed to emulate real-world network degradation at the application layer. Operating as a TCP, UDP, or SOCKS5 proxy, it introduces mathematically structured jitter, serialization delays, and bursty packet loss.

Unlike simple random-delay simulators, lagger utilizes statistical distributions, network state machines, and the **vnm (V Neural Network Module) library** to build a **2-Hidden-Layer Deep Competitive Neural Network (DCNN)** designed to segment and target traffic behaviors adaptively. By evaluating real-time temporal and payload features without decrypting packets, lagger can isolate specific traffic states (such as active streaming, chat messaging, or control signals) and apply customized network impairment profiles only when those behaviors are detected.

Additionally, lagger supports **Active Traffic Morphing**, allowing it to reshape the size and pacing of a program's traffic to match the statistical profile of another recorded application in real-time.

---

## Technical Specifications

### Simulation Vectors
*   **Layer 3/4 Packet Loss (Gilbert-Elliott Model):** Rather than dropping packets with flat, uniform probability, the proxy implements a stateful two-state Markov chain. It switches between `Good` and `Bad` link states, mimicking the bursty, localized packet drops typical of wireless networks and sudden congestion.
*   **Layer 1/2 Serialization Delay:** Computes dynamic transmission overhead based on actual packet size (in bits) against a user-defined physical bandwidth limit (in Mbps), introducing realistic delay differences between small keep-alive packets and large data blocks.
*   **Heavy-Tailed Queuing Delay (Pareto & Gaussian Jitter):** Simulates hardware-level routing congestion using the Pareto distribution alongside Box-Muller Gaussian jitter, recreating the occasional high-latency spikes (spikes/tails) seen in congested WAN routes.
*   **Autocorrelation (EMA Filter):** Employs an Exponential Moving Average (EMA) to ensure consecutive packets experience correlated delays rather than independent, chaotic fluctuations, emulating the smooth latency drift of physical paths.

### Active Traffic Morphing & High-Entropy Obfuscation
When running in morphing mode, the proxy acts as a traffic shaper that reshapes packets to mimic the loaded behavior profiles without breaking application stability:
*   **Cryptographic-Safe Padding:** Rather than using zero-padding (which reduces entropy and is easily flagged by DPI engines), lagger appends pseudo-random bytes generated on-the-fly (`FastRng.next_u8()`). This preserves a high-entropy profile indistinguishable from standard encrypted TLS streams.
*   **Smart Segmentation:** Oversized TCP payloads are dynamically split to match target cluster sizes. To protect stream stability and prevent packet explosion, splits are capped at a minimum of 512 bytes and a maximum of 4 fragments per read.
*   **Adaptive Packet Pacing:** Inter-packet arrival times are dynamically buffered and spaced to align with the target PPS (Packets Per Second) of the target state. A pacing cap of 500ms prevents infinite queue lag and connection dropouts.
*   **Handshake Bypass Protection:** The first 15 packets of any connection bypass morphing and pacing. This allows critical connection-negotiation phases (e.g., SOCKS5 authentication, TLS Client/Server Hello) to establish natively before active shaping begins.
*   **UDP Integrity Preservation:** UDP traffic is shaped strictly through high-entropy padding. Since splitting UDP datagrams degrades application payload structures, lagger bypasses fragmentation on UDP routes to maintain stability.

### Encrypted Content Side-Channel Analysis (28D Representation)
To bypass packet encryption and obfuscation without decryption, lagger extracts multi-layered side-channel features and projects them into a **28-Dimensional competitive learning space** (representing 32 behavioral modes):

1.  **Temporal Profile Layer (IAT Moments & Velocity):**
    *   `PPS` (Packets Per Second)
    *   `Jitter` (IAT Standard Deviation)
    *   `IAT Skewness` (3rd statistical moment measuring distribution asymmetry)
    *   `IAT Kurtosis` (4th statistical moment measuring bursty/extreme delay outliers)
    *   `PPS Derivative (Acceleration)` (Velocity and acceleration of packet sequence arrivals)
2.  **Structural & Sizing Profile Layer (Packet Ratios & Runs):**
    *   `Average Packet Size`
    *   `Size Standard Deviation` (Differentiates steady media streams from volatile web-browsing sizes)
    *   `Large Packet Ratio` (MTU-like packets larger than 1200 bytes)
    *   `Medium Packet Ratio` (Interactive and control payloads between 128 and 1024 bytes)
    *   `Small Packet Ratio` (Control, keep-alive, or metadata packets smaller than 128 bytes)
    *   `Constant Run Ratio` (Consecutive packets with identical length, indicating specific tunnel behaviors)
3.  **Cryptographic Block & Padding Alignment Layer:**
    *   `DES/Blowfish Alignment (8-byte)` (Ratio of packet sizes aligned to standard 8-byte boundaries)
    *   `AES Block Alignment (16-byte)` (16-byte alignment tracking, highly typical of block ciphers)
    *   `Tunneling Alignment (32-byte)` (32-byte alignment typical of modern tunneling/VPN schemes)
    *   `AVX/Cache Alignment (64-byte)` (64-byte boundary padding representation)
    *   `Tail Repetitive Pattern Ratio` (Ratio of packets matching repeating trailing byte structures, indicative of trailing padding leaks)
4.  **Advanced Entropy Scaling Layer:**
    *   `Shannon Entropy` (Payload randomness evaluation, $O(N)$ complexity)
    *   `Entropy Volatility` (Standard deviation of Shannon Entropy over the window, identifying phase changes)
    *   `Internal Rolling Entropy Variance` (Spatial entropy variation within 64-byte payload chunks of a single packet)
    *   `Byte Frequency Uniformity` (Deviation of byte distributions from perfect cryptographic randomness)
    *   `Rényi Order-2 Entropy` (Collision entropy assessing maximum unpredictability peaks)
    *   `Min-Entropy` (Worst-case randomness bound, leaking encryption boundaries)
5.  **Markovian / Sequential Multi-Lag Layer:**
    *   `Size Autocorrelation Lag 1` (Consecutive packet sizing correlation)
    *   `Size Autocorrelation Lag 2` (Sizing correlation across two-step gaps)
    *   `Size Autocorrelation Lag 3` (Sizing correlation across three-step gaps)

### Deep Competitive Neural Network & vnm Backprop Engine
Rather than using a flat nearest-neighbor lookup or custom matrix calculations, lagger integrates the **vnm (V Neural Network Module)** library to run an optimized, structured feed-forward neural network pipeline:
*   **Layer Architecture:** $28 \text{ inputs} \to 64 \text{ neurons (ReLU)} \to 32 \text{ neurons (ReLU)} \to 32 \text{ competitive output nodes (Linear)}$.
*   **vnm Backpropagation & SGD Optimization:** When active learning is enabled, classification error propagates back through the layers using the library's gradient descent pipeline (`vnm.NeuralNetwork.train_step`). Weights and biases are updated dynamically via Stochastic Gradient Descent (SGD) to reinforce non-linear feature representations.
*   **Type Compatibility:** Designed to scale dynamically with the library's type definition (`vnm.Fnn`), ensuring safety and performance whether the library is compiled in `f32` or `f64` precision mode.

### Dimension-Invariant Geometric Calibration
To counter the "curse of dimensionality" over 28 dimensions, lagger implements two geometric optimizations to compute modeling confidence:
*   **Dynamic Seeding (First-Activation Initialization):** Centroids are dynamically seeded with the active input vector on their first classification match, entirely bypassing cold-start convergence phases.
*   **Dimension-Invariant Normalization:** The Euclidean distance ($d$) is normalized against the dimension size ($N=28$):
    $$d_{\text{norm}} = \frac{d}{\sqrt{28}}$$
    $$\text{Confidence} = e^{-d_{\text{norm}} \cdot \gamma} \times 100$$
    This ensures confidence scores scale predictably from 0% to 100%, preventing dimensional inflation from masking model certainty.

### Grammatical & Run-Length Encoded State Machine
Instead of simple state concatenation (which degrades into illegible strings over time), lagger features a **Run-Length Encoded (RLE) & Grammatical Pair Merger** (`StateCompressor`).
*   **Run-Length Compression:** Consecutive repeating states are summarized as a multiplier (e.g., state `4` repeated 3 times prints as `4n3`).
*   **Structural Chaining:** Frequently co-occurring distinct sequences are combined using grammatical addition (e.g., when `"12n2"` and `"25n2"` occur together repeatedly, they merge into `"12n2+25n2"`).
*   **Zero-Flapping Filter:** Transition logs and state updates are completely gated by the confidence threshold. Low-confidence cluster boundaries are ignored, preserving the previous high-confidence macro-state and keeping terminal output extremely quiet and readable.

---

## Application Contexts (Dual-Use Framework)

The mathematical precision of lagger makes it a highly capable tool across a spectrum of benign, defensive, and controversial application scenarios.

### Benign & Defensive Applications
*   **Robustness & QA Testing:** Crucial for evaluating how applications (e.g., VoIP clients, multiplayer netcode, video streaming services, or distributed database replicas) handle severe connection degradation and sudden packet loss.
*   **Traffic Obfuscation & Anti-Fingerprinting:** Modern passive network analysis often relies on **Website/Application Fingerprinting** by observing Inter-Packet Arrival Times (IAT) and packet size distributions. By routing traffic through `lagger` and injecting customized Gaussian/Pareto jitter, these temporal side-channels are scrambled, neutralizing statistical network fingerprinting and enhancing user privacy.
*   **Network Path Emulation:** Provides local developers with a reliable way to reproduce the exact packet characteristics of 3G, satellite, or congested WAN links without physical testing infrastructure.

### Controversial & Exploitative Applications
*   **Multiplayer Game Exploits (Lag Switching):** In peer-to-peer or server-authoritative multiplayer games, artificially delaying upstream traffic while preserving downstream flow can be used to exploit client-side prediction, reconciliation, or rollback netcode. This tactic, commonly known as "lag switching," can grant unfair gameplay advantages.
*   **Rate-Limiter Evasion:** Introducing highly irregular, autocorrelated delays can sometimes bypass primitive intrusion detection or rate-limiting systems that flag uniform, high-frequency connection patterns.

---

## Quick start (copy - paste - enter)
```bash
pkg update -y && pkg install -y git clang make && if ! command -v v >/dev/null 2>&1; then git clone --depth=1 https://github.com/vlang/v && cd v && make && ./v symlink && cd ..; fi && git clone --depth=1 https://github.com/tailsmails/lagger && v install --git https://github.com/tailsmails/vnm && cd lagger && v -prod lagger.v -d vnm_f64 -o lagger && ln -sf $(pwd)/lagger $PREFIX/bin/lagger
```

---

## Requirements
*   **Operating System:** Cross-platform (Linux, macOS, Windows) as it relies on standard BSD socket APIs.
*   **Privileges:** Standard user privileges (root/sudo is **not** required, as it runs as a user-space proxy).
*   **Compiler:** V programming language compiler.
*   **Project Structure:** To compile successfully, the `vnm` module must be installed:
    ```bash
    v install --git https://github.com/tailsmails/vnm
    ```

---

## Installation

Compile the source code using the V compiler:
```bash
v -prod lagger.v -o lagger
```

---

## Usage

Start the proxy by specifying the protocol, port, target, and physical properties:

```bash
./lagger --proto socks5 --port 1080 --up-bandwidth 10 --down-bandwidth 50 --up-min 100 --up-max 200 --up-natural
```

### Key CLI Parameters
*   `--proto`: Proxy protocol to run (`socks5`, `tcp`, or `udp`).
*   `--port`: Local listening port (defaults to `1080`).
*   `--target`: Destination target (required for `tcp` and `udp` forwarding).
*   `--up-bandwidth` / `--down-bandwidth`: Simulated upload/download bandwidth limits in Mbps (defaults to `50.0`).
*   `--up-min` / `--up-max`: Minimum and maximum baseline latency bounds in milliseconds.
*   `--up-natural` / `--down-natural`: Enables the physical layer emulation (Gilbert-Elliott loss, Pareto noise, and Gaussian jitter).
*   `--up-jitter`: Standard deviation of the Gaussian jitter.
*   `--up-correlation`: Autocorrelation factor between successive packet delays (0.0 to 1.0).
*   `-a`, `--analyze`: Runs in analyze-only mode (bypasses all lagging and drops, solely displaying live behavioral state transitions).
*   `-v`, `--save-model`: Name of the `.lgr` file to save the learned clustering centroids and state lag templates upon exiting (defaults to `model.lgr`).
*   `-f`, `--load-model`: Path to a `.lgr` file to load learned centroids and custom state-specific lag configurations.
*   `-g`, `--lag-on`: Only apply lag/loss on these comma-separated states (e.g., `--lag-on 2,3`). Bypasses lagging on any unlisted state.
*   `-c`, `--conf-threshold`: Minimum clustering model confidence percentage required to trigger lagging **and** gate transition analysis printing (0.0 to 100.0, defaults to `0.0`). Filtering out low-confidence state transitions eliminates boundary noise.
*   `-t`, `--target-filter`: Comma-separated filter of target domains or IPs (e.g., `telegram,149.154`). Only analyzes and lags matching hosts, letting other background sockets bypass the proxy unhindered.
*   `-X`, `--morph`: Enables active traffic morphing. When a model is loaded, this actively shapes packet sizes (using high-entropy random padding or controlled segmentation) and paces packet intervals to match the loaded behavior profile while maintaining read-only model classification.

---

## Advanced targeted lagging & workspaces

Instead of applying a flat lag across the entire connection, you can assign **independent, customized lag workspaces** to any of the 32 states. For example, you can set a minor latency oscillation for text chatting (e.g., State 2) and a heavy, lossy latency wave for media streaming (e.g., State 3), while leaving standard keep-alives untouched.

### Step 1: Profiling the Application (Analyze Mode)
Run the proxy in analyze-only mode, filtering specifically for your target app:
```bash
./lagger --proto socks5 --port 1080 --analyze --save-model my_profile.lgr --target-filter "telegram,149.154"
```
Once you are done capturing traffic patterns, hit `Ctrl+C` to gracefully stop. A human-readable, pretty-printed JSON config will be written to `my_profile.lgr`.

### Step 2: Customizing the Workspaces
Open the generated `my_profile.lgr` file in any text editor. You will see the serialization format mapping to `vnm`'s architecture, including learned centroids in their 28-dimensional format and a customizable `lag_configs` block. Assign separate `WaveConfig` workspaces to different modes:

```json
{
  "model": {
    "seq": {
      "net": {
        "layers": [
          { "weights": { "rows": 64, "cols": 28, "data": [...] }, "biases": { "rows": 64, "cols": 1, "data": [...] } },
          { "weights": { "rows": 32, "cols": 64, "data": [...] }, "biases": { "rows": 32, "cols": 1, "data": [...] } },
          { "weights": { "rows": 32, "cols": 32, "data": [...] }, "biases": { "rows": 32, "cols": 1, "data": [...] } }
        ]
      }
    },
    "centroids": [
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.01, 0.05, 0.05, 0.85, 0.02, 0.01, 0.02, 0.05, 0.1, 0.05, 0.9, 0.3, 0.55, 0.2, 0.1, 0.1, 0.1, 0.05, 0.5, 0.5, 0.1, 0.1, 0.0, 0.5, 0.1, 0.0, 0.0, 0.5],
      [0.08, 0.12, 0.4, 0.88, 0.15, 0.05, 0.05, 0.2, 0.25, 0.15, 0.8, 0.4, 0.6, 0.3, 0.15, 0.2, 0.25, 0.1, 0.5, 0.5, 0.2, 0.2, 0.1, 0.52, 0.15, 0.05, 0.05, 0.52]
    ]
  },
  "lag_configs": {
    "2": {
      "min_lat": 40.0,
      "max_lat": 100.0,
      "sync": false,
      "sync_inverse": false,
      "pattern": "sine",
      "period": 5.0,
      "custom": [],
      "natural": true,
      "jitter": 5.0,
      "correlation": 0.8,
      "inverse": false,
      "last_lat": 0.0,
      "is_bad_state": false,
      "loss_enabled": false,
      "bandwidth_mbps": 50.0,
      "analyze_only": false
    },
    "3": {
      "min_lat": 400.0,
      "max_lat": 800.0,
      "sync": false,
      "sync_inverse": false,
      "pattern": "random",
      "period": 2.0,
      "custom": [],
      "natural": true,
      "jitter": 40.0,
      "correlation": 0.6,
      "inverse": false,
      "last_lat": 0.0,
      "is_bad_state": false,
      "loss_enabled": true,
      "bandwidth_mbps": 1.5,
      "analyze_only": false
    }
  }
}
```

### Step 3: Run Active Traffic Morphing
To actively shape the traffic of a target program to look like the recorded behavioral profile of the loaded model:
```bash
./lagger --proto socks5 --port 1080 --load-model my_profile.lgr --target-filter "149.154" --morph --conf-threshold 90.0
```

---

## Disclaimer
This software is provided "as is" and is intended solely for educational purposes, software quality assurance, network robustness auditing, and privacy research. The developer assumes no liability and accepts no responsibility for any misuse, unauthorized network manipulation, exploitation in online platforms/gaming environments, or damage caused by the use of this tool. Users are entirely responsible for ensuring their actions comply with relevant local laws, service agreements, and regulations.

---

## License
![License](https://img.shields.io/badge/License-GPLv3-blue.svg)
