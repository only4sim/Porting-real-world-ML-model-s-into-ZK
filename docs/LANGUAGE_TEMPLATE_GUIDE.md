# Language Template System Guide

This guide explains how to add support for new programming languages to the XGBoost decision tree converter.

## Architecture Overview

The system uses a hybrid template-based architecture with three main components:

1. **Language Configuration File** (`language_configs/{language}_config.json`) - Syntax rules and type definitions
2. **Template Files** (`language_templates/{language}_*.template`) - Code structure patterns
3. **Language-Specific Handlers** (in `XGBoostLanguageConverter` class) - Semantic conversion logic

This design ensures proper separation of concerns while maintaining correctness for language-specific requirements like ZoKrates' numeric array indexing.

## Adding a New Language

### Step 1: Create Language Configuration

Create `language_configs/{language}_config.json` with the following structure:

```json
{
  "language": "your_language_name",
  "file_extension": ".ext",
  "data_types": {
    "fixed_point": {
      "type_name": "your_fixed_point_type",
      "struct_definition": "optional struct definition",
      "zero_value": "zero representation",
      "precision_multiplier": 10000000000
    },
    "boolean": "bool_type",
    "array": "array_syntax_pattern"
  },
  "operators": {
    "less_equal": {
      "function_name": "le_function",
      "call_format": "function_call_pattern"
    },
    "add": {
      "function_name": "add_function",
      "call_format": "function_call_pattern"
    }
  },
  "control_structures": {
    "if_condition": "if syntax",
    "else_clause": "else syntax",
    "end_block": "block end syntax",
    "assignment": "assignment syntax",
    "mutable_declaration": "mutable variable syntax"
  },
  "function_syntax": {
    "definition": "function definition syntax",
    "return": "return statement syntax",
    "end_function": "function end syntax",
    "private_param": "private parameter keyword",
    "public_param": "public parameter keyword"
  },
  "comments": {
    "single_line": "//",
    "multi_line_start": "/*",
    "multi_line_end": "*/"
  },
  "indentation": {
    "type": "spaces|tabs",
    "size": 4
  }
}
```

### Step 2: Create Template Files

Create these template files in `language_templates/`:

#### `{language}_header.template`
Contains helper functions, data type definitions, and utility code needed for the target language.

Example for ZoKrates:
```zokrates
struct i64{
    bool sgn;
    u64 v;
}

def i64_le(i64 lth, i64 rth) -> bool{
    // comparison logic
}

def i64_add(i64 lth, i64 rth) -> i64{
    // addition logic
}
```

#### `{language}_main.template`
Main function template with placeholders:

```
def main(private i64[{num_features}] f) -> i64 {{
    // variable declarations
{tree_code}
    return result;
}}
```

#### `{language}_tree.template`
Template for individual tree processing:

```
  // Tree {tree_idx}
{tree_logic}
```

### Step 3: Implement Language-Specific Methods

In `xgb_language_converter.py`, add language-specific implementations in these methods:

```python
def convert_number_to_fixed_point_from_scaled(self, scaled_value: int) -> str:
    if self.language == "your_language":
        # Implement your language's fixed-point conversion from pre-scaled integer
        # Note: scaled_value is already multiplied by 10^10 for precision
        is_positive = scaled_value >= 0
        abs_value = abs(scaled_value)
        return f"your_language_format({abs_value}, {'positive' if is_positive else 'negative'})"
    
def convert_number_to_field(self, value: float) -> str:
    if self.language == "your_language":
        # Use 10^10 precision multiplier (CRITICAL for compatibility)
        converted_value = int(np.round(value * 10000000000, 0))
        # Implement your language's field format
        pass

def convert_number_to_input(self, value: float) -> str:
    if self.language == "your_language":
        # Use 10^10 precision multiplier (CRITICAL for compatibility)
        converted_value = int(np.round(value * 10000000000, 0))
        # Implement your language's input format
        pass
```

**CRITICAL**: Always use the exact precision multiplier `10000000000` (10^10) to maintain compatibility with the original implementation.

### Step 4: Usage

```python
# Create converter for your language
converter = XGBoostLanguageConverter('your_language')

# Convert XGBoost model
code = converter.convert_xgboost_to_code(booster, feature_names, num_trees)

# Save to file
converter.save_code_to_file(code, 'output_file')
```

## Template Placeholders

Available placeholders in templates:

- `{num_features}`: Number of input features
- `{tree_code}`: Generated code for all trees
- `{tree_idx}`: Current tree index
- `{tree_logic}`: Generated logic for current tree

## Key Design Principles

1. **Separation of Concerns**: Configuration handles syntax, templates handle structure, code handles semantics
2. **Extensibility**: Easy to add new languages without modifying existing code
3. **Consistency**: All languages follow the same conversion pipeline with standardized precision
4. **Maintainability**: Changes to one language don't affect others
5. **Compatibility**: Maintains exact numeric precision (10^10) and array indexing compatibility

## Critical Implementation Requirements

### Array Indexing
- **XGBoost Format**: Features are named `f34`, `f22`, `f85` (with 'f' prefix)
- **Target Language**: Must use numeric indices `f[34]`, `f[22]`, `f[85]` for array access
- **Conversion**: The `feature_names_to_indices()` method handles this mapping automatically
- **Example**: XGBoost `f34` becomes ZoKrates `f[34]` for array compatibility

### Precision Requirements
- **Multiplier**: Always use exactly `10000000000` (10^10) for all numeric conversions
- **Rounding**: Use `int(np.round(value * 10000000000, 0))` for consistency
- **Purpose**: Maintains compatibility with original implementation and ensures identical results

## Testing New Languages

1. **Basic Functionality**: Create test cases with simple XGBoost models
2. **Code Compilation**: Verify generated code compiles in target language
3. **Array Indexing**: Ensure feature access uses proper numeric indices (not names)
4. **Precision Validation**: Test numeric conversion with known values
5. **Tree Structure**: Test with various tree depths and feature counts
6. **Documentation Testing**: Ensure doc-tests and examples compile and run (for languages that support it)
7. **Reference Comparison**: Compare outputs with `plot_model_zok_old.ipynb` for validation

## Validation Checklist

For any new language implementation, verify:

- [ ] Array indexing uses numeric format: `array[index]` not `array[name]`
- [ ] Precision multiplier is exactly `10000000000` (10^10)
- [ ] Generated code compiles without syntax errors
- [ ] Numeric conversions match expected precision
- [ ] Tree logic correctly handles leaf nodes and decision nodes
- [ ] Feature mapping works for all model types (1-5)
- [ ] Output format is compatible with target deployment environment
- [ ] Documentation examples include proper imports and correct feature counts
- [ ] All tests pass for the target language

## Reference Implementations

### ZoKrates Reference Implementation

The complete ZoKrates implementation demonstrates best practices:

#### Configuration
- `converter/language_configs/zokrates_config.json`: Complete syntax and type definitions

#### Templates  
- `converter/language_templates/zokrates_header.template`: Helper functions and data structures
- `converter/language_templates/zokrates_main.template`: Main function structure
- `converter/language_templates/zokrates_tree.template`: Individual tree formatting

#### Code Implementation
- `XGBoostLanguageConverter` class: All ZoKrates-specific conversion methods
- Array indexing: `f[34]` format for proper ZoKrates compatibility
- Data precision: 10^10 multiplier for fixed-point arithmetic

#### Validation
- Built-in test cell in `converter/model_converter.ipynb`: Automated verification
- Cross-language validation with reference implementations

### Rust Reference Implementation

The Rust implementation showcases custom fixed-point arithmetic without external dependencies:

#### Configuration
- `converter/language_configs/rust_config.json`: Rust syntax definitions with custom fixed-point types
- Uses `i64` type instead of external crate types
- Operators: `fixed_le()`, `fixed_add()` for custom arithmetic

#### Templates
- `converter/language_templates/rust_header.template`: Custom fixed-point arithmetic functions
- `converter/language_templates/rust_main.template`: Main prediction function with comprehensive documentation
- `converter/language_templates/rust_tree.template`: Individual tree processing with proper scope
- `converter/language_templates/rust_cargo.template`: Cargo.toml with no external dependencies
- `converter/language_templates/rust_test.template`: Comprehensive test suite

#### Key Features
- **No External Dependencies**: Uses only Rust standard library
- **Custom Fixed-Point**: Implementation with `const PRECISION_MULTIPLIER: i64 = 10_000_000_000`
- **Thread Safety**: All functions are thread-safe with `#[inline]` optimizations
- **Overflow Protection**: Uses `saturating_add()` for safe arithmetic
- **Comprehensive Testing**: Built-in unit tests for all arithmetic operations
- **Build Verified**: All compilation errors resolved, builds cleanly with latest Rust

#### Code Structure
```rust
// Fixed-point constants and functions
const PRECISION_MULTIPLIER: i64 = 10_000_000_000;
fn fixed_le(a: i64, b: i64) -> bool { a <= b }
fn fixed_add(a: i64, b: i64) -> i64 { a.saturating_add(b) }
fn from_scaled_i64(scaled_value: i64) -> i64 { scaled_value }

// Public utility functions (exported from library)
pub fn to_fixed_point(value: f64) -> i64 { /* convert float to scaled i64 */ }
pub fn from_fixed_point(fixed_value: i64) -> f64 { /* convert back to float */ }

// Main prediction function
pub fn xgboost_predict(features: &[i64]) -> i64 { /* decision tree logic */ }
```

#### Build Requirements and Fixes Applied
- **Function Visibility**: All utility functions marked as `pub` for proper library export
- **Type Annotations**: Explicit type annotations added to resolve compiler ambiguity
- **Template Updates**: Header template updated to generate public functions
- **Doc-Test Fix**: Main template updated with proper imports and 116-feature examples
- **Verification**: Build test program included for validation

#### Validation and Testing
- Built-in test module with comprehensive arithmetic validation
- Round-trip testing for fixed-point conversion accuracy
- Prediction testing with known input vectors
- Performance testing for batch operations
- Build verification with integrated test programs
- Clean compilation verified with `cargo build --release`
- Documentation examples compile and run successfully
- Generated files include proper imports and feature vector examples

Both reference implementations have been validated and produce identical results with proper array indexing and precision handling.