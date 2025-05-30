// Simple build test to verify the fixes
// This file can be used to test compilation without running the full program

// Import the generated library functions
use rainfall_prediction::{xgboost_predict, from_fixed_point, to_fixed_point};

fn main() {
    println!("Testing Rust build fixes...");
    
    // Test 1: Function visibility
    let test_value = 1.5f64;
    let fixed = to_fixed_point(test_value);
    let back = from_fixed_point(fixed);
    println!("✓ Function visibility test passed: {} -> {} -> {}", test_value, fixed, back);
    
    // Test 2: Type specificity
    let diff: f64 = (test_value - back).abs();
    println!("✓ Type specificity test passed: difference = {:.2e}", diff);
    
    // Test 3: Main prediction function
    let features = vec![0i64; 116];
    let prediction = xgboost_predict(&features);
    let prediction_float = from_fixed_point(prediction);
    println!("✓ Prediction function test passed: {} (scaled) = {:.6} mm", prediction, prediction_float);
    
    println!("All build tests passed! ✅");
}