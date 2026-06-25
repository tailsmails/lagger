# lagger

## Overview
lagger is a dynamic network latency, packet loss simulation, and side-channel behavioral analysis proxy designed to emulate real-world network degradation at the application layer. Operating as a TCP, UDP, or SOCKS5 proxy, it introduces mathematically structured jitter, serialization delays, and bursty packet loss.

Unlike simple random-delay simulators, lagger utilizes statistical distributions, network state machines, and an **unsupervised competitive learning clustering engine** to segment and target traffic behaviors adaptively. By evaluating real-time temporal and payload features without decrypting packets, lagger can isolate specific traffic states (such as active streaming, chat messaging, or control signals) and apply customized network impairment profiles only when those behaviors are detected.

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

### Encrypted Content Side-Channel Analysis (11D Clustering)
To bypass packet encryption and obfuscation without decryption, lagger extracts multi-layered side-channel features and projects them into an **11-Dimensional competitive learning space** (Mode 0 to Mode 11):

1.  **Macro-Metadata Layer:**
    *   `PPS` (Packets Per Second)
    *   `Average Packet Size`
    *   `Jitter` (Inter-Arrival Time Standard Deviation)
    *   `Shannon Entropy` (Payload randomness evaluation, $O(N)$ complexity)
2.  **Granular Traffic Layer:**
    *   `Size Standard Deviation` (Measures uniform media streams vs. jittery web-browsing sizes)
    *   `Large Packet Ratio` (Proportion of MTU-like packets larger than 1200 bytes)
    *   `Entropy Volatility` (Standard deviation of entropy, identifying encryption phase changes)
    *   `Burst Density` (Ratio of packets arriving within highly dense < 8ms windows)
3.  **Encrypted Content/Cryptographic Layer:**
    *   `Block Padding Ratio` (Detects structural 16-byte alignment patterns typical of AES block ciphers)
    *   `Internal Rolling Entropy Variance` (Spatial entropy variation within 64-byte chunks of a single packet)
    *   `Byte Frequency Uniformity` (Statistical evaluation measuring the deviation of byte distribution from perfect cryptographic randomness)

### Radial Basis Function (RBF) Confidence Calibration
Due to the high-dimensional feature space, linear distance metrics suffer from the "curse of dimensionality." lagger implements a non-linear **Exponential RBF Kernel mapping** to compute clustering confidence:
$$\text{Confidence} = e^{-\lambda \cdot d} \times 100$$
This ensures mathematically smooth, stable, and naturally scaled confidence ratings, allowing more accurate behavior classification without false transitions near cluster boundaries.

### Grammatical & Run-Length Encoded State Machine
Instead of simple state concatenation (which degrades into illegible strings over time), lagger features a **Run-Length Encoded (RLE) & Grammatical Pair Merger** (`StateCompressor`).
*   **Run-Length Compression:** Consecutive repeating states are summarized as a multiplier (e.g., state `1` repeated 4 times prints as `1n4`).
*   **Structural Chaining:** Frequently co-occurring distinct sequences are combined using grammatical addition (e.g., when `"95n2"` and `"50n2"` occur together repeatedly, they merge into `"95n2+50n2"`).
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
pkg update -y && pkg install -y git clang make && if ! command -v v >/dev/null 2>&1; then git clone --depth=1 https://github.com/vlang/v && cd v && make && ./v symlink && cd ..; fi && git clone --depth=1 https://github.com/tailsmails/lagger && cd lagger && v -prod lagger.v -o lagger && ln -sf $(pwd)/lagger $PREFIX/bin/lagger
```

---

## Requirements
*   **Operating System:** Cross-platform (Linux, macOS, Windows) as it relies on standard BSD socket APIs.
*   **Privileges:** Standard user privileges (root/sudo is **not** required, as it runs as a user-space proxy).
*   **Compiler:** V programming language compiler.

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

Instead of applying a flat lag across the entire connection, you can assign **independent, customized lag workspaces** to any of the 12 states. For example, you can set a minor latency oscillation for text chatting (e.g., State 2) and a heavy, lossy latency wave for media streaming (e.g., State 3), while leaving standard keep-alives untouched.

### Step 1: Profiling the Application (Analyze Mode)
Run the proxy in analyze-only mode, filtering specifically for your target app:
```bash
./lagger --proto socks5 --port 1080 --analyze --save-model my_profile.lgr --target-filter "telegram,149.154"
```
Once you are done capturing traffic patterns, hit `Ctrl+C` to gracefully stop. A human-readable, pretty-printed JSON config will be written to `my_profile.lgr`.

### Step 2: Customizing the Workspaces
Open the generated `my_profile.lgr` file in any text editor. You will see the learned centroids in their 11-dimensional format and a customizable `lag_configs` block. Assign separate `WaveConfig` workspaces to different modes:

```json
{
  "model": {
    "centroids": [
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.01, 0.05, 0.05, 0.85, 0.02, 0.01, 0.02, 0.05, 0.1, 0.05, 0.9],
      [0.08, 0.12, 0.4, 0.88, 0.15, 0.05, 0.05, 0.2, 0.25, 0.15, 0.8],
      [0.85, 0.95, 0.05, 0.98, 0.05, 0.9, 0.01, 0.02, 0.95, 0.02, 0.99],
      [0.55, 0.35, 0.6, 0.92, 0.25, 0.2, 0.1, 0.35, 0.4, 0.1, 0.75],
      [0.05, 0.8, 0.0, 0.95, 0.1, 0.75, 0.02, 0.05, 0.8, 0.05, 0.98],
      [0.25, 0.15, 0.75, 0.86, 0.3, 0.1, 0.15, 0.45, 0.2, 0.25, 0.7],
      [0.9, 0.1, 0.25, 0.95, 0.05, 0.02, 0.03, 0.1, 0.1, 0.05, 0.95],
      [0.02, 0.3, 0.2, 0.9, 0.12, 0.15, 0.08, 0.15, 0.3, 0.1, 0.85],
      [0.4, 0.7, 0.3, 0.95, 0.2, 0.6, 0.05, 0.25, 0.7, 0.08, 0.95],
      [0.3, 0.1, 0.9, 0.85, 0.35, 0.05, 0.18, 0.5, 0.15, 0.3, 0.65],
      [0.7, 0.5, 0.5, 0.95, 0.22, 0.45, 0.08, 0.3, 0.5, 0.12, 0.88]
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
