# ZoKrates Implementation - Zero-Knowledge Weather Prediction

Generate and test zero-knowledge circuits for privacy-preserving rainfall prediction.

## Quick Start

```bash
# Install ZoKrates
curl -LSfs get.zokrat.es | sh
export PATH=$PATH:~/.zokrates/bin

# Test circuit compilation and proof generation
cd bst1/
python test.py
```

## What This Enables

- **Privacy-Preserving Predictions**: Weather stations can prove rainfall without revealing exact measurements
- **Blockchain Integration**: Smart contracts can verify weather predictions cryptographically  
- **Decentralized Weather Oracles**: Multiple stations contribute to consensus without data sharing

## Directory Structure

```
zok_simulation/
├── bst1/              # Model 1 implementation (single radar scan)
│   ├── programs/      # Generated .zok circuit files
│   ├── test.py       # Circuit compilation and proof testing
│   ├── abi.json      # Circuit interface definition
│   ├── proof.json    # Generated zero-knowledge proof
│   └── verifier.sol  # Solidity contract for on-chain verification
├── bst2/             # Model 2 implementation (1-4 scans)
├── bst3/             # Model 3 implementation (3-8 scans)  
├── bst4/             # Model 4 implementation (7-18 scans)
└── bst5/             # Model 5 implementation (17-1000 scans)
```

## Usage Workflow

### 1. Circuit Compilation
```bash
cd bst1/programs/
zokrates compile -i bst1_10.zok  # Compile 10-tree model to circuit
```

### 2. Proof Generation
```bash
# Generate witness (private computation)
zokrates compute-witness --args /* your 116 private weather measurements */

# Generate zero-knowledge proof
zokrates generate-proof

# Export verification contract
zokrates export-verifier
```

### 3. Blockchain Deployment
```bash
# Deploy verifier.sol to blockchain
# Smart contracts can now verify weather predictions without seeing raw data
```

## Generated Files

- **`.zok` circuits**: Generated from `../PSE/outputs/zokrates/`
- **`abi.json`**: Circuit interface for external integration
- **`proof.json`**: Zero-knowledge proof that computation was performed correctly
- **`verifier.sol`**: Solidity smart contract for on-chain verification
- **`proving.key` / `verification.key`**: Cryptographic keys for proof system

## Circuit Details

- **Input**: 116 private meteorological measurements (weather radar data)
- **Output**: Public rainfall prediction in mm
- **Proof**: Cryptographic guarantee that prediction was computed correctly without revealing inputs
- **Tree Variants**: 10-300 decision trees (different accuracy/performance tradeoffs)

## Integration Examples

### Weather Oracle Smart Contract
```solidity
contract WeatherOracle {
    IVerifier immutable verifier;
    
    function submitPrediction(
        uint256[8] memory proof,
        uint256[1] memory publicInputs  // rainfall prediction
    ) external {
        require(verifier.verifyTx(proof, publicInputs), "Invalid proof");
        // Accept prediction as verified
    }
}
```

### Privacy-Preserving Weather API
```python
# Weather station generates proof locally
proof = zokrates_prove(private_measurements)

# Submit only proof + prediction (measurements stay private)
weather_api.submit_verified_prediction(proof, prediction)
```

## Testing

The `test.py` script in each model directory performs:
1. Circuit compilation verification
2. Witness generation with test data
3. Proof generation and validation
4. Verifier contract export

## Performance

- **Circuit Size**: Varies by tree count (10-300 trees)
- **Proof Generation**: ~10-60 seconds depending on circuit complexity
- **Proof Verification**: ~1-5 milliseconds on-chain
- **Proof Size**: ~256 bytes (constant regardless of input size)

## Troubleshooting

- **ZoKrates Installation**: Ensure PATH includes `~/.zokrates/bin`
- **Circuit Compilation**: Use exact file paths for .zok files
- **Memory Issues**: Use fewer trees (e.g., bst1_10.zok instead of bst1_300.zok)
- **Proof Generation**: Ensure exactly 116 input features are provided