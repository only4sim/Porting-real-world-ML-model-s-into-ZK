// Test program for XGBoost rainfall prediction model
// This demonstrates how to use the generated Rust prediction function

use std::env;
use std::io::{self, Write};

// Import the generated rainfall prediction library
use rainfall_prediction::{xgboost_predict, from_fixed_point, to_fixed_point};

/// Convert array of float features to scaled integers
fn prepare_features(float_features: &[f64]) -> Vec<i64> {
    float_features.iter()
        .map(|&x| to_fixed_point(x))
        .collect()
}

fn main() {
    println!("XGBoost Rainfall Prediction Test Program");
    println!("========================================");

    let args: Vec<String> = env::args().collect();

    match args.len() {
        1 => {
            // Interactive mode
            interactive_mode();
        }
        2 if args[1] == "--test" => {
            // Run built-in tests
            run_tests();
        }
        2 if args[1] == "--demo" => {
            // Run demonstration with sample data
            run_demo();
        }
        117.. => {
            // Command line mode with 116 features
            command_line_mode(&args[1..117]);
        }
        _ => {
            print_usage(&args[0]);
        }
    }
}

fn interactive_mode() {
    println!("Interactive mode - Enter rainfall prediction features");
    println!("Note: This is a simplified example. Real features should come from radar data.");
    println!();

    loop {
        println!("Choose an option:");
        println!("1. Test with sample data");
        println!("2. Enter custom features");
        println!("3. Batch prediction demo");
        println!("4. Performance test");
        println!("5. Exit");
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        match input.trim() {
            "1" => test_with_sample_data(),
            "2" => custom_feature_input(),
            "3" => batch_prediction_demo(),
            "4" => performance_test(),
            "5" => break,
            _ => println!("Invalid option. Please try again."),
        }
        println!();
    }
}

fn test_with_sample_data() {
    println!("Testing with sample meteorological data...");
    
    // Sample features representing typical radar/weather measurements
    // These would normally come from processed radar scans
    let sample_features = vec![
        0.0220286213,  // Reflectivity feature
        0.045,         // Velocity feature  
        -0.018,        // Spectrum width
        0.12,          // Differential reflectivity
        -0.005,        // Correlation coefficient
        0.089,         // Specific differential phase
        0.234,         // Hydrometeor type indicator
        0.156,         // Rain rate estimate
        0.078,         // Liquid water content
        0.045,         // Ice water content
    ];

    // Pad with zeros to reach 116 features
    let mut full_features = sample_features;
    full_features.resize(116, 0.0);

    let prediction = make_prediction(&full_features);
    
    println!("Sample Input Features (first 10):");
    for (i, &feature) in full_features.iter().take(10).enumerate() {
        println!("  Feature {}: {:.6}", i, feature);
    }
    println!("  ... (106 more features)");
    println!();
    println!("Prediction Result:");
    println!("  Rainfall: {:.6} mm", prediction);
    
    // Interpret the result
    if prediction < 0.1 {
        println!("  Interpretation: No significant rainfall expected");
    } else if prediction < 1.0 {
        println!("  Interpretation: Light rainfall expected");
    } else if prediction < 5.0 {
        println!("  Interpretation: Moderate rainfall expected");
    } else if prediction < 20.0 {
        println!("  Interpretation: Heavy rainfall expected");
    } else {
        println!("  Interpretation: Very heavy rainfall expected");
    }
}

fn custom_feature_input() {
    println!("Custom feature input (simplified - enter 5 key features):");
    println!("This is a demonstration. Real usage requires all 116 radar features.");
    
    let feature_names = [
        "Reflectivity (dBZ, typical range: -20 to 60)",
        "Radial Velocity (m/s, typical range: -30 to 30)", 
        "Spectrum Width (m/s, typical range: 0 to 10)",
        "Differential Reflectivity (dB, typical range: -2 to 5)",
        "Correlation Coefficient (unitless, range: 0 to 1)",
    ];

    let mut features = vec![0.0; 116];
    
    for (i, name) in feature_names.iter().enumerate() {
        print!("Enter {}: ", name);
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        match input.trim().parse::<f64>() {
            Ok(value) => features[i] = value,
            Err(_) => {
                println!("Invalid input, using 0.0");
                features[i] = 0.0;
            }
        }
    }

    let prediction = make_prediction(&features);
    println!();
    println!("Prediction with custom features: {:.6} mm", prediction);
}

fn batch_prediction_demo() {
    println!("Batch prediction demonstration...");
    
    // Generate sample batch data
    let batch_data = vec![
        vec![0.02, 0.04, -0.01, 0.1, 0.08],     // Light rain scenario
        vec![0.15, 0.12, 0.02, 0.25, 0.15],     // Moderate rain scenario  
        vec![0.45, 0.30, 0.08, 0.40, 0.35],     // Heavy rain scenario
        vec![0.01, 0.005, -0.002, 0.02, 0.01],  // Clear weather scenario
    ];

    let scenarios = ["Light rain", "Moderate rain", "Heavy rain", "Clear weather"];

    println!("Batch processing {} scenarios:", batch_data.len());
    println!();

    for (i, data) in batch_data.iter().enumerate() {
        let mut full_features = data.clone();
        full_features.resize(116, 0.0);
        
        let prediction = make_prediction(&full_features);
        
        println!("Scenario {}: {} -> {:.6} mm", 
                i + 1, scenarios[i], prediction);
    }
}

fn performance_test() {
    println!("Performance test - processing 1000 predictions...");
    
    let start = std::time::Instant::now();
    
    // Generate random-ish test data
    let mut predictions = Vec::new();
    for i in 0..1000 {
        let mut features = vec![0.0; 116];
        // Fill with some variation
        for j in 0..10 {
            features[j] = (i as f64 * 0.001 + j as f64 * 0.01) % 1.0;
        }
        
        let prediction = make_prediction(&features);
        predictions.push(prediction);
    }
    
    let duration = start.elapsed();
    
    println!("Processed 1000 predictions in {:?}", duration);
    println!("Average time per prediction: {:?}", duration / 1000);
    println!("Predictions per second: {:.0}", 1000.0 / duration.as_secs_f64());
    
    // Show some statistics
    let avg = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let min = predictions.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = predictions.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    println!();
    println!("Prediction statistics:");
    println!("  Average: {:.6} mm", avg);
    println!("  Minimum: {:.6} mm", min);
    println!("  Maximum: {:.6} mm", max);
}

fn command_line_mode(feature_args: &[String]) {
    println!("Command line mode with {} features", feature_args.len());
    
    let features: Result<Vec<f64>, _> = feature_args.iter()
        .map(|s| s.parse::<f64>())
        .collect();
    
    match features {
        Ok(feature_values) => {
            let prediction = make_prediction(&feature_values);
            println!("Prediction: {:.6} mm", prediction);
        }
        Err(e) => {
            eprintln!("Error parsing features: {}", e);
            std::process::exit(1);
        }
    }
}

fn run_tests() {
    println!("Running built-in tests...");
    println!();

    // Test 1: Fixed-point conversion
    println!("Test 1: Fixed-point arithmetic");
    let test_values = [0.0, 1.0, -1.0, 0.5, -0.5, 123.456789, -987.654321];
    
    for &value in &test_values {
        let fixed = to_fixed_point(value);
        let back = from_fixed_point(fixed);
        let error = (value - back).abs() as f64;
        
        println!("  {} -> {} -> {} (error: {:.2e})", 
                value, fixed, back, error);
        
        assert!(error < 1e-9, "Conversion error too large for {}", value);
    }
    println!("  ✓ Fixed-point conversion tests passed");
    println!();

    // Test 2: Feature preparation
    println!("Test 2: Feature preparation");
    let float_features = vec![0.1, -0.2, 0.0, 1.5, -2.8];
    let scaled_features = prepare_features(&float_features);
    
    println!("  Float features: {:?}", float_features);
    println!("  Scaled features: {:?}", scaled_features);
    
    for (i, (&original, &scaled)) in float_features.iter().zip(scaled_features.iter()).enumerate() {
        let expected = to_fixed_point(original);
        assert_eq!(scaled, expected, "Feature {} scaling mismatch", i);
    }
    println!("  ✓ Feature preparation tests passed");
    println!();

    // Test 3: Prediction consistency
    println!("Test 3: Prediction consistency");
    let test_features = vec![0.0; 116];
    
    let prediction1 = make_prediction(&test_features);
    let prediction2 = make_prediction(&test_features);
    
    assert_eq!(prediction1, prediction2, "Predictions should be deterministic");
    println!("  ✓ Prediction consistency test passed");
    println!();

    println!("All tests passed! ✓");
}

fn run_demo() {
    println!("XGBoost Rainfall Prediction Demo");
    println!("===============================");
    println!();
    
    println!("This demo shows how the generated Rust code works:");
    println!("1. Input: 116 meteorological features from radar scans");
    println!("2. Processing: XGBoost decision trees with fixed-point arithmetic");
    println!("3. Output: Rainfall prediction in millimeters");
    println!();
    
    test_with_sample_data();
    println!();
    
    println!("Key technical details:");
    println!("- Uses custom fixed-point arithmetic (i64 with 10^10 precision)");
    println!("- No external dependencies (pure Rust standard library)");
    println!("- Thread-safe and suitable for high-performance applications");
    println!("- Generated from XGBoost model with {} decision trees", 10);
    println!();
    
    println!("For production use:");
    println!("- Replace this main.rs with your application logic");
    println!("- Import the prediction function from the generated lib.rs");
    println!("- Ensure input features are properly scaled and validated");
}

fn make_prediction(features: &[f64]) -> f64 {
    // Ensure we have enough features
    let mut full_features = features.to_vec();
    full_features.resize(116, 0.0);
    
    // Scale features to fixed-point
    let scaled_features = prepare_features(&full_features);
    
    // Make prediction (this would call the generated function in real usage)
    let prediction_scaled = xgboost_predict(&scaled_features);
    
    // Convert back to float
    from_fixed_point(prediction_scaled)
}

fn print_usage(program_name: &str) {
    println!("Usage:");
    println!("  {}                          # Interactive mode", program_name);
    println!("  {} --test                   # Run built-in tests", program_name);
    println!("  {} --demo                   # Run demonstration", program_name);
    println!("  {} <f1> <f2> ... <f116>     # Command line with 116 features", program_name);
    println!();
    println!("Examples:");
    println!("  {} --demo", program_name);
    println!("  {} --test", program_name);
    println!("  {} $(python -c \"print(' '.join(['0.1'] * 116))\")", program_name);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_roundtrip() {
        let test_values = [0.0, 1.0, -1.0, 0.123456789, -987.654321];
        
        for &value in &test_values {
            let fixed = to_fixed_point(value);
            let back = from_fixed_point(fixed);
            let error = (value - back).abs() as f64;
            
            assert!(error < 1e-9, "Round-trip error too large for {}: error = {}", value, error);
        }
    }

    #[test]
    fn test_feature_preparation() {
        let features = vec![0.1, -0.2, 0.0, 1.5];
        let scaled = prepare_features(&features);
        
        assert_eq!(scaled.len(), features.len());
        
        for (i, (&original, &scaled_val)) in features.iter().zip(scaled.iter()).enumerate() {
            let expected = to_fixed_point(original);
            assert_eq!(scaled_val, expected, "Feature {} scaling incorrect", i);
        }
    }

    #[test]
    fn test_prediction_deterministic() {
        let features = vec![0.0; 116];
        let pred1 = make_prediction(&features);
        let pred2 = make_prediction(&features);
        
        assert_eq!(pred1, pred2, "Predictions should be deterministic");
    }
}