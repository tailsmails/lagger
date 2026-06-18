# lagger

## Overview
lagger is a dynamic network latency and packet loss simulation proxy designed to emulate real-world network degradation at the application layer. Operating as a TCP, UDP, or SOCKS5 proxy, it introduces mathematically structured jitter, serialization delays, and bursty packet loss to simulate highly variable network environments.

Unlike simple random-delay simulators, lagger utilizes statistical distributions and network state machines to model physical link constraints, router queuing delays, and transport-layer characteristics.

---

## Technical Specifications

### Simulation Vectors
*   **Layer 3/4 Packet Loss (Gilbert-Elliott Model):** Rather than dropping packets with flat, uniform probability, the proxy implements a stateful two-state Markov chain. It switches between `Good` and `Bad` link states, mimicking the bursty, localized packet drops typical of wireless networks and sudden congestion.
*   **Layer 1/2 Serialization Delay:** Computes dynamic transmission overhead based on actual packet size (in bits) against a user-defined physical bandwidth limit (in Mbps), introducing realistic delay differences between small keep-alive packets and large data blocks.
*   **Heavy-Tailed Queuing Delay (Pareto & Gaussian Jitter):** Simulates hardware-level routing congestion using the Pareto distribution alongside Box-Muller Gaussian jitter, recreating the occasional high-latency spikes (spikes/tails) seen in congested WAN routes.
*   **Autocorrelation (EMA Filter):** Employs an Exponential Moving Average (EMA) to ensure consecutive packets experience correlated delays rather than independent, chaotic fluctuations, emulating the smooth latency drift of physical paths.

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

---

## Disclaimer
This software is provided "as is" and is intended solely for educational purposes, software quality assurance, network robustness auditing, and privacy research. The developer assumes no liability and accepts no responsibility for any misuse, unauthorized network manipulation, exploitation in online platforms/gaming environments, or damage caused by the use of this tool. Users are entirely responsible for ensuring their actions comply with relevant local laws, service agreements, and regulations.

---

## License
![License](https://img.shields.io/badge/License-MIT-blue.svg)