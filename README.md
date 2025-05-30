# XGBoost Rainfall Prediction - Multi-Language Deployment System

A high-performance machine learning deployment system that converts trained XGBoost rainfall prediction models into production code for zero-knowledge proof systems and native applications.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+ with XGBoost, pandas, numpy
- Rust (for Rust deployment) - [Install Rust](https://rustup.rs/)
- ZoKrates (for zero-knowledge proofs) - [Install ZoKrates](https://zokrates.github.io/)

### Basic Setup

**Note**: We have pre-prepared the `feature_names_cache.json` file for ease of reproduction.

```bash
# Convert models to target languages (cache already prepared)
cd converter/
jupyter notebook model_converter.ipynb
# Execute cells 0-6 for complete conversion
```

**Optional**: If you need to regenerate the cache file or processed data, refer to [Devin Anzelmo's Kaggle project](https://www.kaggle.com/c/how-much-did-it-rain/discussion/16260) to generate all files in the `../processed/` folder, then run:
```bash
python -c "import cached_features as cf; cf.save_feature_names_cache()"
```

### Test Deployments

**Rust (High-Performance Native)**
```bash
cd rust_simulation/
cargo build --release
cargo run --bin predict -- --demo
```

**ZoKrates (Zero-Knowledge Proofs)**
```bash
cd zok_simulation/bst1/
curl -LSfs get.zokrat.es | sh
export PATH=$PATH:~/.zokrates/bin
python test.py
```

## 📖 What This Project Does

This system converts trained XGBoost decision tree models for rainfall prediction into multiple programming languages, enabling:

- **🔐 Privacy-Preserving Predictions**: Deploy ML models in zero-knowledge proof systems
- **⚡ High-Performance Deployment**: Generate optimized Rust code for production applications  
- **🌐 Blockchain Integration**: Create verifiable weather predictions without revealing input data
- **🔧 Multi-Language Support**: Extensible template system for additional target languages

### Input → Output
- **Input**: XGBoost models trained on 116 meteorological features from weather radar
- **Output**: Production-ready code in Rust, ZoKrates, and other target languages

## 🏗️ Architecture

### Core Components

```
converter/                    # Main conversion system
├── cached_features.py        # 100x performance speedup through intelligent caching
├── xgb_language_converter.py # Template-based multi-language converter  
├── functions.py             # Model loading and data processing utilities
├── model_converter.ipynb    # Main workflow (7 cells)
├── language_configs/        # Syntax definitions for target languages
├── language_templates/      # Code structure templates
└── outputs/                 # Generated code (Rust, ZoKrates)

models/                      # 5 specialized XGBoost models
├── bst1_1_final_subm       # Single radar scan (count = 1)
├── bst2_1_final_subm       # Low scans (1 < count < 4)  
├── bst3_1_final_subm       # Medium scans (3 < count < 8)
├── bst4_1_final_subm       # High scans (7 < count < 18)
└── bst5_1_final_subm       # Very high scans (17 < count < 1000)

rust_simulation/             # Rust testing and validation
zok_simulation/             # ZoKrates testing and proof generation
```

### Multi-Model Ensemble
The system uses **5 specialized models** trained on different radar scan count ranges, allowing optimal predictions based on data availability:

- **Model 1**: Optimized for single radar measurements
- **Models 2-3**: Handle sparse to moderate radar coverage  
- **Models 4-5**: Leverage dense radar networks (with optional extra meteorological features)

## 🛠️ Usage

### Model Conversion Workflow

**Step 1: Convert Models (Cache Pre-prepared)**
```python
# Load from cache (extremely fast)
feature_names, instr = cf.load_feature_names_and_instr_from_cache(model_num, use_xtra_features)

# Convert to target language
from xgb_language_converter import XGBoostLanguageConverter
converter = XGBoostLanguageConverter('rust')  # or 'zokrates'
code = converter.convert_xgboost_to_code(bst, feature_names, tree_limit)
```

**Step 2: Deploy Generated Code**

**Rust Integration**
```rust
use rainfall_prediction::{xgboost_predict, to_fixed_point, from_fixed_point};

// Convert 116 meteorological features
let features: Vec<f64> = /* your radar measurements */;
let scaled_features: Vec<i64> = features.iter()
    .map(|&x| to_fixed_point(x))
    .collect();

// Predict rainfall
let prediction = xgboost_predict(&scaled_features);
let rainfall_mm = from_fixed_point(prediction);
```

**ZoKrates Zero-Knowledge Proofs**
```bash
# Compile to circuit
zokrates compile -i bst1_10.zok

# Generate proof (input stays private)
zokrates compute-witness --args /* private weather data */
zokrates generate-proof

# Verify on blockchain
zokrates export-verifier
```

## ⚡ Performance

### Speed Optimizations
- **Caching System**: 100x speedup for feature extraction
- **Zero Dependencies**: Rust implementation uses only standard library

## 🔧 Development Guide

### Adding New Target Languages

The template system makes it easy to add support for new programming languages:

1. **Create Language Config**: `language_configs/{language}_config.json`
2. **Add Templates**: `language_templates/{language}_*.template`  
3. **Implement Handlers**: Add conversion methods to `XGBoostLanguageConverter`

See `LANGUAGE_TEMPLATE_GUIDE.md` for detailed instructions.

### Testing and Validation

**Comprehensive Test Suite**
```bash
# Rust build and test
cd rust_simulation/  
cargo test                           # Unit tests
cargo run --bin predict -- --test   # Integration tests

# ZoKrates circuit validation
cd zok_simulation/bst1/
python test.py                       # End-to-end circuit tests
```

### Key Implementation Details

**Precision Consistency**: All numeric conversions use exactly **10^10 multiplier** for cross-language compatibility:
```python
# Critical for maintaining identical results across languages
int(np.round(value * 10000000000, 0))
```

**Array Indexing**: XGBoost feature names (`f34`) are automatically converted to numeric indices (`f[34]`) for target language array compatibility.

## 📁 Generated Output

The system generates complete, ready-to-deploy implementations:

```
converter/outputs/
├── rust/
│   ├── bst1_10.rs, bst1_20.rs, ..., bst1_300.rs    # Different tree counts
│   ├── bst1_feature_names.txt                       # Feature mappings
│   ├── bst1_instr.txt                              # Processing instructions
│   ├── Cargo.toml                                  # Complete Rust project
│   └── [models 2-5 with variants]
└── zokrates/
    ├── bst1_10.zok, bst1_20.zok, ...              # ZoKrates circuits
    └── [feature and instruction files]
```

## 🔍 Troubleshooting

### Common Issues

**Rust Build Problems**
- ✅ **All resolved**: Generated code compiles cleanly without errors
- Run `cargo build --release` - should complete without errors
- All compilation issues have been fixed in the latest implementation

**Performance Issues**  
- The cache file `feature_names_cache.json` is pre-prepared in converter directory
- Use `--release` mode for production performance
- For data regeneration, refer to the Kaggle project link above

**Feature Count Errors**
- Always provide exactly **116 meteorological features**
- Pad with zeros if you have fewer measurements
- Check that input arrays match the expected feature count

## 📊 Real-World Applications

### Weather Service Integration
- **Radar Processing**: Convert real-time Doppler radar data to rainfall predictions
- **Batch Processing**: Process historical weather data for climate analysis
- **API Services**: Embed prediction models in web services and mobile apps

### Blockchain Weather Oracles
- **Privacy-Preserving**: Weather stations can prove rainfall without revealing exact measurements  
- **Verifiable Predictions**: Smart contracts can verify weather predictions cryptographically
- **Decentralized Networks**: Multiple weather stations can contribute to consensus without data sharing

### Research Applications
- **Climate Modeling**: High-performance batch processing for climate research
- **Meteorological Studies**: Validate radar-based prediction accuracy
- **Algorithm Development**: Test new weather prediction approaches

## 📚 Documentation

- **[rust_simulation/README.md](rust_simulation/README.md)** - Rust implementation guide  
- **[zok_simulation/README.md](zok_simulation/README.md)** - ZoKrates zero-knowledge circuits guide
- **[docs/LANGUAGE_TEMPLATE_GUIDE.md](docs/LANGUAGE_TEMPLATE_GUIDE.md)** - Adding new target languages

## 🤝 Contributing

This system is designed for extensibility:

1. **New Languages**: Follow the template system to add support for additional programming languages
2. **Performance**: Contribute caching optimizations or algorithmic improvements  
3. **Features**: Add support for different ML models or meteorological data types
4. **Testing**: Expand the validation suite with additional test cases

## 📄 License

Check individual component licenses. Generated code inherits from original project terms.

---

**🌧️ Accurate weather prediction through privacy-preserving machine learning deployment**