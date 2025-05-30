// XGBoost Decision Tree Implementation in Rust
// Uses custom fixed-point arithmetic with i64 for precision compatible with zero-knowledge proofs
// All values are scaled by 10^10 for precision (maintaining compatibility with original implementation)

/// Fixed-point arithmetic constants
const PRECISION_MULTIPLIER: i64 = 10_000_000_000; // 10^10 for precision

/// Fixed-point less-than-or-equal comparison
/// 
/// # Arguments
/// * `a` - First value (scaled by 10^10)
/// * `b` - Second value (scaled by 10^10)
/// 
/// # Returns
/// * `bool` - true if a <= b
#[inline]
fn fixed_le(a: i64, b: i64) -> bool {
    a <= b
}

/// Fixed-point addition with overflow protection
/// 
/// # Arguments
/// * `a` - First value (scaled by 10^10)
/// * `b` - Second value (scaled by 10^10)
/// 
/// # Returns
/// * `i64` - Sum (scaled by 10^10), saturated on overflow
#[inline]
fn fixed_add(a: i64, b: i64) -> i64 {
    a.saturating_add(b)
}

/// Convert floating-point value to fixed-point representation
/// 
/// # Arguments
/// * `value` - Floating-point value
/// 
/// # Returns
/// * `i64` - Fixed-point value (scaled by 10^10)
#[inline]
pub fn to_fixed_point(value: f64) -> i64 {
    (value * PRECISION_MULTIPLIER as f64).round() as i64
}

/// Convert fixed-point value back to floating-point
/// 
/// # Arguments
/// * `fixed_value` - Fixed-point value (scaled by 10^10)
/// 
/// # Returns
/// * `f64` - Floating-point value
#[inline]
pub fn from_fixed_point(fixed_value: i64) -> f64 {
    fixed_value as f64 / PRECISION_MULTIPLIER as f64
}

/// Create fixed-point value from pre-scaled integer (already multiplied by 10^10)
/// This is used for values that are already in the correct scale from the training data
/// 
/// # Arguments
/// * `scaled_value` - Pre-scaled integer value
/// 
/// # Returns
/// * `i64` - Fixed-point value ready for computation
#[inline]
fn from_scaled_i64(scaled_value: i64) -> i64 {
    scaled_value
}
/// Main XGBoost prediction function
/// 
/// # Arguments
/// * `features` - Input feature vector as slice of i64 values (scaled by 10^10)
/// 
/// # Returns
/// * `i64` - Prediction result (scaled by 10^10 for precision)
/// 
/// # Example
/// ```rust
/// use rainfall_prediction::{xgboost_predict, from_fixed_point};
/// 
/// // Create a feature vector with 116 features (meteorological measurements)
/// let mut features = vec![0i64; 116]; // Initialize with zeros
/// features[0] = 220286213;  // Example: scaled reflectivity value
/// features[1] = 450000000;  // Example: scaled velocity value  
/// features[2] = -180000000; // Example: scaled spectrum width value
/// 
/// let prediction = xgboost_predict(&features);
/// println!("Prediction: {}", prediction);
/// 
/// // Convert back to floating point if needed
/// let float_prediction = from_fixed_point(prediction);
/// println!("Float prediction: {:.6} mm", float_prediction);
/// ```
pub fn xgboost_predict(features: &[i64]) -> i64 {
    // Ensure we have the expected number of features
    assert!(features.len() >= 116, 
            "Expected at least {} features, got {}", 116, features.len());
    
    // Features are already in fixed-point format (scaled by 10^10)
    let f = features;
    
    // Initialize accumulator for tree predictions
    let mut y = 0i64;
    
    // Tree 0
    {
    let tree_result = if fixed_le(f[34], from_scaled_i64(120000000000)) {
        if fixed_le(f[22], from_scaled_i64(8450000290)) {
            if fixed_le(f[34], from_scaled_i64(85000000000)) {
                from_scaled_i64(220286213)
            } else {
                if fixed_le(f[85], from_scaled_i64(10316699700)) {
                    if fixed_le(f[54], from_scaled_i64(10000000000)) {
                        from_scaled_i64(216100514)
                    } else {
                        from_scaled_i64(177788269)
                    }
                } else {
                    from_scaled_i64(204757601)
                }
            }
        } else {
            from_scaled_i64(200073291)
        }
    } else {
        if fixed_le(f[71], from_scaled_i64(110000000000)) {
            if fixed_le(f[54], from_scaled_i64(10000000000)) {
                if fixed_le(f[85], from_scaled_i64(9316669700)) {
                    from_scaled_i64(216697901)
                } else {
                    if fixed_le(f[56], from_scaled_i64(46250000000)) {
                        if fixed_le(f[41], from_scaled_i64(200000000000)) {
                            if fixed_le(f[56], from_scaled_i64(-28125000000)) {
                                from_scaled_i64(214853249)
                            } else {
                                if fixed_le(f[77], from_scaled_i64(210000000000)) {
                                    from_scaled_i64(196630303)
                                } else {
                                    from_scaled_i64(182448309)
                                }
                            }
                        } else {
                            from_scaled_i64(148634585)
                        }
                    } else {
                        from_scaled_i64(212080162)
                    }
                }
            } else {
                if fixed_le(f[98], from_scaled_i64(47177402)) {
                    if fixed_le(f[77], from_scaled_i64(510000000000)) {
                        if fixed_le(f[34], from_scaled_i64(180000000000)) {
                            from_scaled_i64(160672814)
                        } else {
                            from_scaled_i64(178509150)
                        }
                    } else {
                        from_scaled_i64(135967126)
                    }
                } else {
                    if fixed_le(f[34], from_scaled_i64(240000000000)) {
                        from_scaled_i64(213882346)
                    } else {
                        from_scaled_i64(170198008)
                    }
                }
            }
        } else {
            if fixed_le(f[77], from_scaled_i64(350000000000)) {
                if fixed_le(f[56], from_scaled_i64(-25000000000)) {
                    from_scaled_i64(131937172)
                } else {
                    from_scaled_i64(189821832)
                }
            } else {
                if fixed_le(f[34], from_scaled_i64(210000000000)) {
                    if fixed_le(f[98], from_scaled_i64(47177402)) {
                        from_scaled_i64(113076912)
                    } else {
                        from_scaled_i64(150089012)
                    }
                } else {
                    from_scaled_i64(65454538)
                }
            }
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 1
    {
    let tree_result = if fixed_le(f[34], from_scaled_i64(120000000000)) {
        if fixed_le(f[41], from_scaled_i64(30000000000)) {
            if fixed_le(f[34], from_scaled_i64(90000000000)) {
                if fixed_le(f[71], from_scaled_i64(10000000000)) {
                    from_scaled_i64(-109014511)
                } else {
                    from_scaled_i64(-92314146)
                }
            } else {
                if fixed_le(f[85], from_scaled_i64(10149999900)) {
                    from_scaled_i64(-104940450)
                } else {
                    from_scaled_i64(-96795242)
                }
            }
        } else {
            if fixed_le(f[77], from_scaled_i64(520000000000)) {
                from_scaled_i64(-98628206)
            } else {
                from_scaled_i64(-82567809)
            }
        }
    } else {
        if fixed_le(f[41], from_scaled_i64(145000000000)) {
            if fixed_le(f[56], from_scaled_i64(16875000000)) {
                if fixed_le(f[77], from_scaled_i64(560000000000)) {
                    if fixed_le(f[98], from_scaled_i64(9097869990)) {
                        if fixed_le(f[71], from_scaled_i64(135000000000)) {
                            from_scaled_i64(-92081446)
                        } else {
                            from_scaled_i64(-68823537)
                        }
                    } else {
                        if fixed_le(f[62], from_scaled_i64(10000000000)) {
                            from_scaled_i64(-89030378)
                        } else {
                            from_scaled_i64(-60643782)
                        }
                    }
                } else {
                    if fixed_le(f[102], from_scaled_i64(2891510130)) {
                        if fixed_le(f[102], from_scaled_i64(1042150040)) {
                            from_scaled_i64(-76886648)
                        } else {
                            from_scaled_i64(-105834836)
                        }
                    } else {
                        from_scaled_i64(-55822791)
                    }
                }
            } else {
                from_scaled_i64(-103122499)
            }
        } else {
            if fixed_le(f[77], from_scaled_i64(460000000000)) {
                from_scaled_i64(-80401516)
            } else {
                if fixed_le(f[71], from_scaled_i64(195000000000)) {
                    from_scaled_i64(-56133452)
                } else {
                    from_scaled_i64(-11083750)
                }
            }
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 2
    {
    let tree_result = if fixed_le(f[54], from_scaled_i64(10000000000)) {
        if fixed_le(f[18], from_scaled_i64(13174599400)) {
            if fixed_le(f[34], from_scaled_i64(115000000000)) {
                from_scaled_i64(-111318324)
            } else {
                if fixed_le(f[85], from_scaled_i64(9216669800)) {
                    from_scaled_i64(-110063581)
                } else {
                    from_scaled_i64(-101747019)
                }
            }
        } else {
            from_scaled_i64(-33535536)
        }
    } else {
        if fixed_le(f[98], from_scaled_i64(47177402)) {
            if fixed_le(f[71], from_scaled_i64(175000000000)) {
                if fixed_le(f[77], from_scaled_i64(190000000000)) {
                    from_scaled_i64(-90477774)
                } else {
                    from_scaled_i64(-75595314)
                }
            } else {
                from_scaled_i64(-23944960)
            }
        } else {
            from_scaled_i64(-109209102)
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 3
    {
    let tree_result = if fixed_le(f[34], from_scaled_i64(115000000000)) {
        if fixed_le(f[102], from_scaled_i64(3017739950)) {
            if fixed_le(f[34], from_scaled_i64(95000000000)) {
                if fixed_le(f[54], from_scaled_i64(10000000000)) {
                    if fixed_le(f[71], from_scaled_i64(75000000000)) {
                        from_scaled_i64(215834305)
                    } else {
                        from_scaled_i64(180885270)
                    }
                } else {
                    from_scaled_i64(176237877)
                }
            } else {
                from_scaled_i64(204360951)
            }
        } else {
            from_scaled_i64(201743413)
        }
    } else {
        if fixed_le(f[54], from_scaled_i64(10000000000)) {
            if fixed_le(f[28], from_scaled_i64(-128489046000)) {
                if fixed_le(f[85], from_scaled_i64(8583329920)) {
                    from_scaled_i64(212018602)
                } else {
                    if fixed_le(f[56], from_scaled_i64(46250000000)) {
                        if fixed_le(f[56], from_scaled_i64(-30000000000)) {
                            from_scaled_i64(214402825)
                        } else {
                            if fixed_le(f[71], from_scaled_i64(110000000000)) {
                                if fixed_le(f[77], from_scaled_i64(160000000000)) {
                                    from_scaled_i64(196175501)
                                } else {
                                    if fixed_le(f[85], from_scaled_i64(10516699600)) {
                                        from_scaled_i64(188515410)
                                    } else {
                                        from_scaled_i64(172999110)
                                    }
                                }
                            } else {
                                from_scaled_i64(166723803)
                            }
                        }
                    } else {
                        from_scaled_i64(211293362)
                    }
                }
            } else {
                if fixed_le(f[77], from_scaled_i64(210000000000)) {
                    from_scaled_i64(188542046)
                } else {
                    if fixed_le(f[41], from_scaled_i64(240000000000)) {
                        if fixed_le(f[85], from_scaled_i64(10249999800)) {
                            if fixed_le(f[56], from_scaled_i64(-625000000)) {
                                from_scaled_i64(141366646)
                            } else {
                                from_scaled_i64(177461114)
                            }
                        } else {
                            from_scaled_i64(115124555)
                        }
                    } else {
                        from_scaled_i64(68099876)
                    }
                }
            }
        } else {
            if fixed_le(f[98], from_scaled_i64(47177402)) {
                if fixed_le(f[77], from_scaled_i64(500000000000)) {
                    if fixed_le(f[71], from_scaled_i64(140000000000)) {
                        if fixed_le(f[77], from_scaled_i64(70000000000)) {
                            from_scaled_i64(183908530)
                        } else {
                            if fixed_le(f[34], from_scaled_i64(175000000000)) {
                                from_scaled_i64(151470201)
                            } else {
                                from_scaled_i64(169683266)
                            }
                        }
                    } else {
                        from_scaled_i64(110215759)
                    }
                } else {
                    from_scaled_i64(125246290)
                }
            } else {
                from_scaled_i64(203246623)
            }
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 4
    {
    let tree_result = if fixed_le(f[34], from_scaled_i64(105000000000)) {
        if fixed_le(f[102], from_scaled_i64(3587639930)) {
            if fixed_le(f[71], from_scaled_i64(30000000000)) {
                from_scaled_i64(-107858507)
            } else {
                from_scaled_i64(-83645908)
            }
        } else {
            from_scaled_i64(-95674908)
        }
    } else {
        if fixed_le(f[41], from_scaled_i64(140000000000)) {
            if fixed_le(f[56], from_scaled_i64(11250000000)) {
                if fixed_le(f[98], from_scaled_i64(7917590140)) {
                    if fixed_le(f[60], from_scaled_i64(-32500000000)) {
                        from_scaled_i64(-104452092)
                    } else {
                        if fixed_le(f[77], from_scaled_i64(550000000000)) {
                            from_scaled_i64(-90929847)
                        } else {
                            from_scaled_i64(-80107646)
                        }
                    }
                } else {
                    if fixed_le(f[34], from_scaled_i64(155000000000)) {
                        from_scaled_i64(-82954019)
                    } else {
                        from_scaled_i64(-51729423)
                    }
                }
            } else {
                if fixed_le(f[85], from_scaled_i64(9083330040)) {
                    from_scaled_i64(-106970109)
                } else {
                    from_scaled_i64(-96427705)
                }
            }
        } else {
            if fixed_le(f[77], from_scaled_i64(340000000000)) {
                if fixed_le(f[77], from_scaled_i64(50000000000)) {
                    from_scaled_i64(-97370520)
                } else {
                    from_scaled_i64(-75484496)
                }
            } else {
                if fixed_le(f[71], from_scaled_i64(190000000000)) {
                    if fixed_le(f[32], from_scaled_i64(-132573223000)) {
                        from_scaled_i64(-89075370)
                    } else {
                        from_scaled_i64(-59816572)
                    }
                } else {
                    from_scaled_i64(-23698979)
                }
            }
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 5
    {
    let tree_result = if fixed_le(f[54], from_scaled_i64(10000000000)) {
        if fixed_le(f[28], from_scaled_i64(-106306915000)) {
            if fixed_le(f[34], from_scaled_i64(115000000000)) {
                from_scaled_i64(-110623874)
            } else {
                if fixed_le(f[85], from_scaled_i64(9283329840)) {
                    from_scaled_i64(-109469993)
                } else {
                    from_scaled_i64(-100927744)
                }
            }
        } else {
            from_scaled_i64(-44213431)
        }
    } else {
        if fixed_le(f[71], from_scaled_i64(175000000000)) {
            if fixed_le(f[98], from_scaled_i64(47177402)) {
                if fixed_le(f[77], from_scaled_i64(100000000000)) {
                    from_scaled_i64(-91569303)
                } else {
                    if fixed_le(f[22], from_scaled_i64(80000000000)) {
                        if fixed_le(f[34], from_scaled_i64(130000000000)) {
                            from_scaled_i64(-92457486)
                        } else {
                            from_scaled_i64(-70770509)
                        }
                    } else {
                        from_scaled_i64(-35172943)
                    }
                }
            } else {
                from_scaled_i64(-106590400)
            }
        } else {
            from_scaled_i64(-11418733)
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 6
    {
    let tree_result = if fixed_le(f[34], from_scaled_i64(120000000000)) {
        if fixed_le(f[34], from_scaled_i64(80000000000)) {
            if fixed_le(f[41], from_scaled_i64(-15000000000)) {
                from_scaled_i64(211216267)
            } else {
                from_scaled_i64(199115016)
            }
        } else {
            if fixed_le(f[71], from_scaled_i64(95000000000)) {
                if fixed_le(f[102], from_scaled_i64(4632590120)) {
                    from_scaled_i64(203387998)
                } else {
                    if fixed_le(f[77], from_scaled_i64(580000000000)) {
                        from_scaled_i64(192870963)
                    } else {
                        from_scaled_i64(156547148)
                    }
                }
            } else {
                from_scaled_i64(177206714)
            }
        }
    } else {
        if fixed_le(f[71], from_scaled_i64(185000000000)) {
            if fixed_le(f[54], from_scaled_i64(10000000000)) {
                if fixed_le(f[98], from_scaled_i64(2408719960)) {
                    if fixed_le(f[85], from_scaled_i64(9250000120)) {
                        from_scaled_i64(208464283)
                    } else {
                        if fixed_le(f[28], from_scaled_i64(-128126249000)) {
                            if fixed_le(f[56], from_scaled_i64(26250000000)) {
                                if fixed_le(f[56], from_scaled_i64(-20625000000)) {
                                    from_scaled_i64(203255098)
                                } else {
                                    from_scaled_i64(184152368)
                                }
                            } else {
                                from_scaled_i64(201723371)
                            }
                        } else {
                            from_scaled_i64(144163128)
                        }
                    }
                } else {
                    if fixed_le(f[77], from_scaled_i64(570000000000)) {
                        if fixed_le(f[77], from_scaled_i64(110000000000)) {
                            from_scaled_i64(195179284)
                        } else {
                            if fixed_le(f[62], from_scaled_i64(10000000000)) {
                                from_scaled_i64(179194454)
                            } else {
                                from_scaled_i64(158879962)
                            }
                        }
                    } else {
                        if fixed_le(f[32], from_scaled_i64(-130523911000)) {
                            from_scaled_i64(174450502)
                        } else {
                            if fixed_le(f[65], from_scaled_i64(-78750000000)) {
                                from_scaled_i64(150654847)
                            } else {
                                from_scaled_i64(112415636)
                            }
                        }
                    }
                }
            } else {
                if fixed_le(f[98], from_scaled_i64(47177402)) {
                    if fixed_le(f[77], from_scaled_i64(550000000000)) {
                        if fixed_le(f[22], from_scaled_i64(145000000000)) {
                            if fixed_le(f[34], from_scaled_i64(165000000000)) {
                                from_scaled_i64(154358177)
                            } else {
                                from_scaled_i64(176384971)
                            }
                        } else {
                            from_scaled_i64(120440479)
                        }
                    } else {
                        if fixed_le(f[77], from_scaled_i64(580000000000)) {
                            from_scaled_i64(104939900)
                        } else {
                            from_scaled_i64(139307147)
                        }
                    }
                } else {
                    from_scaled_i64(200974531)
                }
            }
        } else {
            if fixed_le(f[77], from_scaled_i64(510000000000)) {
                if fixed_le(f[32], from_scaled_i64(-95737161600)) {
                    from_scaled_i64(172833018)
                } else {
                    from_scaled_i64(100538107)
                }
            } else {
                from_scaled_i64(64264256)
            }
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 7
    {
    let tree_result = if fixed_le(f[34], from_scaled_i64(105000000000)) {
        if fixed_le(f[98], from_scaled_i64(5610970260)) {
            from_scaled_i64(-107177328)
        } else {
            if fixed_le(f[77], from_scaled_i64(560000000000)) {
                if fixed_le(f[60], from_scaled_i64(-10625000000)) {
                    from_scaled_i64(-79097264)
                } else {
                    from_scaled_i64(-99452883)
                }
            } else {
                from_scaled_i64(-75558540)
            }
        }
    } else {
        if fixed_le(f[41], from_scaled_i64(125000000000)) {
            if fixed_le(f[56], from_scaled_i64(21875000000)) {
                if fixed_le(f[60], from_scaled_i64(-30000000000)) {
                    from_scaled_i64(-105727250)
                } else {
                    if fixed_le(f[77], from_scaled_i64(420000000000)) {
                        from_scaled_i64(-92028007)
                    } else {
                        if fixed_le(f[34], from_scaled_i64(140000000000)) {
                            from_scaled_i64(-89877127)
                        } else {
                            if fixed_le(f[98], from_scaled_i64(8946099880)) {
                                from_scaled_i64(-81418483)
                            } else {
                                from_scaled_i64(-43446147)
                            }
                        }
                    }
                }
            } else {
                from_scaled_i64(-102541065)
            }
        } else {
            if fixed_le(f[77], from_scaled_i64(160000000000)) {
                from_scaled_i64(-93834037)
            } else {
                if fixed_le(f[22], from_scaled_i64(185000000000)) {
                    if fixed_le(f[71], from_scaled_i64(95000000000)) {
                        from_scaled_i64(-77793938)
                    } else {
                        if fixed_le(f[77], from_scaled_i64(560000000000)) {
                            from_scaled_i64(-70290868)
                        } else {
                            from_scaled_i64(-37369209)
                        }
                    }
                } else {
                    if fixed_le(f[48], from_scaled_i64(-8587239980)) {
                        from_scaled_i64(-25736820)
                    } else {
                        from_scaled_i64(-65213507)
                    }
                }
            }
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 8
    {
    let tree_result = if fixed_le(f[54], from_scaled_i64(10000000000)) {
        if fixed_le(f[34], from_scaled_i64(115000000000)) {
            from_scaled_i64(-110157225)
        } else {
            if fixed_le(f[41], from_scaled_i64(250000000000)) {
                if fixed_le(f[85], from_scaled_i64(10083299900)) {
                    from_scaled_i64(-105701117)
                } else {
                    from_scaled_i64(-98838126)
                }
            } else {
                from_scaled_i64(-56816954)
            }
        }
    } else {
        if fixed_le(f[98], from_scaled_i64(47177402)) {
            if fixed_le(f[71], from_scaled_i64(175000000000)) {
                if fixed_le(f[77], from_scaled_i64(510000000000)) {
                    from_scaled_i64(-81313355)
                } else {
                    if fixed_le(f[34], from_scaled_i64(135000000000)) {
                        from_scaled_i64(-89351647)
                    } else {
                        from_scaled_i64(-50091222)
                    }
                }
            } else {
                from_scaled_i64(-16243105)
            }
        } else {
            from_scaled_i64(-107505322)
        }
    };

        y = fixed_add(y, tree_result);
    }
    // Tree 9
    {
    let tree_result = if fixed_le(f[34], from_scaled_i64(120000000000)) {
        if fixed_le(f[41], from_scaled_i64(25000000000)) {
            if fixed_le(f[34], from_scaled_i64(80000000000)) {
                if fixed_le(f[71], from_scaled_i64(0)) {
                    from_scaled_i64(207114760)
                } else {
                    from_scaled_i64(183432624)
                }
            } else {
                if fixed_le(f[85], from_scaled_i64(10316699700)) {
                    if fixed_le(f[54], from_scaled_i64(10000000000)) {
                        from_scaled_i64(203589965)
                    } else {
                        from_scaled_i64(167924576)
                    }
                } else {
                    from_scaled_i64(192529727)
                }
            }
        } else {
            if fixed_le(f[77], from_scaled_i64(520000000000)) {
                from_scaled_i64(194547437)
            } else {
                if fixed_le(f[41], from_scaled_i64(115000000000)) {
                    from_scaled_i64(180566125)
                } else {
                    from_scaled_i64(153003298)
                }
            }
        }
    } else {
        if fixed_le(f[71], from_scaled_i64(175000000000)) {
            if fixed_le(f[54], from_scaled_i64(10000000000)) {
                if fixed_le(f[85], from_scaled_i64(9049999710)) {
                    from_scaled_i64(202941615)
                } else {
                    if fixed_le(f[41], from_scaled_i64(200000000000)) {
                        if fixed_le(f[56], from_scaled_i64(41250000000)) {
                            if fixed_le(f[77], from_scaled_i64(340000000000)) {
                                from_scaled_i64(183587614)
                            } else {
                                if fixed_le(f[98], from_scaled_i64(1500000060)) {
                                    from_scaled_i64(179055259)
                                } else {
                                    if fixed_le(f[69], from_scaled_i64(-30000000000)) {
                                        from_scaled_i64(182974041)
                                    } else {
                                        from_scaled_i64(154395122)
                                    }
                                }
                            }
                        } else {
                            from_scaled_i64(201292746)
                        }
                    } else {
                        from_scaled_i64(134445932)
                    }
                }
            } else {
                if fixed_le(f[98], from_scaled_i64(47177402)) {
                    if fixed_le(f[77], from_scaled_i64(550000000000)) {
                        if fixed_le(f[34], from_scaled_i64(195000000000)) {
                            if fixed_le(f[77], from_scaled_i64(70000000000)) {
                                from_scaled_i64(172630139)
                            } else {
                                from_scaled_i64(146571761)
                            }
                        } else {
                            from_scaled_i64(173615366)
                        }
                    } else {
                        from_scaled_i64(121499514)
                    }
                } else {
                    if fixed_le(f[34], from_scaled_i64(220000000000)) {
                        from_scaled_i64(202018451)
                    } else {
                        from_scaled_i64(168260261)
                    }
                }
            }
        } else {
            if fixed_le(f[77], from_scaled_i64(350000000000)) {
                if fixed_le(f[56], from_scaled_i64(-1875000000)) {
                    from_scaled_i64(113139534)
                } else {
                    from_scaled_i64(170389228)
                }
            } else {
                if fixed_le(f[71], from_scaled_i64(220000000000)) {
                    from_scaled_i64(121934097)
                } else {
                    from_scaled_i64(49785199)
                }
            }
        }
    };

        y = fixed_add(y, tree_result);
    }
    
    // Return result in fixed-point format
    y
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xgboost_predict_basic() {
        // Test with sample feature vector (values already scaled by 10^10)
        let test_features = vec![
            220286213,   // Example scaled values
            450000000,
            -180000000,
            100000000,
            -50000000
        ];
        
        // Extend with zeros if we need more features
        let mut features = test_features;
        features.resize(116, 0);
        
        let prediction = xgboost_predict(&features);
        
        // Basic sanity checks
        assert!(prediction.abs() < 1000000000000); // Reasonable range for scaled values
        println!("Prediction for test case: {} (scaled)", prediction);
        println!("Prediction as float: {:.6}", from_fixed_point(prediction));
    }
    
    #[test]
    fn test_xgboost_predict_zeros() {
        // Test with all zero features
        let features = vec![0i64; 116];
        let prediction = xgboost_predict(&features);
        
        // Should return some prediction for zero input
        println!("Prediction for zero input: {} (scaled)", prediction);
        println!("Prediction as float: {:.6}", from_fixed_point(prediction));
    }
    
    #[test] 
    fn test_fixed_point_conversion() {
        // Test fixed-point conversion functions
        let original_float = 0.0220286213;
        let scaled_value = to_fixed_point(original_float);
        let back_to_float = from_fixed_point(scaled_value);
        
        // Should round-trip correctly (within precision limits)
        let diff = (original_float - back_to_float).abs();
        assert!(diff < 1e-9); // Allow for small rounding errors
        
        println!("Round-trip test: {} -> {} -> {}", original_float, scaled_value, back_to_float);
        
        // Test from_scaled_i64 function
        let test_scaled = 220286213i64;
        let result = from_scaled_i64(test_scaled);
        assert_eq!(test_scaled, result);
        println!("from_scaled_i64 test: {} -> {}", test_scaled, result);
    }
    
    #[test]
    fn test_fixed_point_arithmetic() {
        // Test fixed-point arithmetic functions
        let a = to_fixed_point(1.5);  // 15000000000
        let b = to_fixed_point(2.3);  // 23000000000
        
        // Test less-than-or-equal
        assert!(fixed_le(a, b));
        assert!(!fixed_le(b, a));
        assert!(fixed_le(a, a));
        
        // Test addition
        let sum = fixed_add(a, b);
        let expected = to_fixed_point(3.8);
        assert_eq!(sum, expected);
        
        println!("Arithmetic test: {} + {} = {} (expected {})", a, b, sum, expected);
        println!("As floats: {:.1} + {:.1} = {:.1}", 
                from_fixed_point(a), from_fixed_point(b), from_fixed_point(sum));
    }
}