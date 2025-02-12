# Graph-State BMC encoder

Encoder for encoding graph-state synthesis problems in CNF (only encoding, no solving), written in Rust. Can be used as a binary tool, or using included Python bindings.


## Building

```shell
# install maturin
python -m venv .venv
source .venv/bin/activate
pip install maturin

# to build the binary for main.rs in debug (release)
cargo build [-r]

# compile rust + python bindings in debug (release)
maturin develop [-r]
```
