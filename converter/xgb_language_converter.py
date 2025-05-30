import json
import os
import numpy as np
from typing import Dict, Any, List, Tuple

class XGBoostLanguageConverter:
    """
    Generalized XGBoost to programming language converter using template system.
    """
    
    def __init__(self, language: str, config_dir: str = "language_configs", template_dir: str = "language_templates"):
        """
        Initialize converter for specific language.
        
        Args:
            language: Target language name (e.g., 'zokrates')
            config_dir: Directory containing language configuration files
            template_dir: Directory containing language template files
        """
        self.language = language
        self.config_dir = config_dir
        self.template_dir = template_dir
        
        # Load language configuration
        config_path = os.path.join(config_dir, f"{language}_config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load templates
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, str]:
        """Load all template files for the language."""
        templates = {}
        template_files = [
            f"{self.language}_header.template",
            f"{self.language}_main.template", 
            f"{self.language}_tree.template"
        ]
        
        # Add language-specific additional templates
        if self.language == "rust":
            template_files.extend([
                f"{self.language}_cargo.template",
                f"{self.language}_test.template"
            ])
        
        for template_file in template_files:
            template_path = os.path.join(self.template_dir, template_file)
            if os.path.exists(template_path):
                with open(template_path, 'r') as f:
                    template_name = template_file.replace(f"{self.language}_", "").replace(".template", "")
                    templates[template_name] = f.read()
        
        return templates
    
    def convert_number_to_fixed_point(self, value: float) -> str:
        """Convert a number to target language's fixed-point representation."""
        fixed_point_config = self.config["data_types"]["fixed_point"]
        multiplier = fixed_point_config["precision_multiplier"]
        converted_value = int(np.round(value * multiplier, 0))
        
        # Handle sign
        is_positive = converted_value >= 0
        abs_value = abs(converted_value)
        
        if self.language == "zokrates":
            return f"i64{{sgn:{'true' if is_positive else 'false'}, v: {abs_value}}}"
        elif self.language == "rust":
            return f"from_scaled_i64({converted_value})"
        else:
            # For other languages, implement their specific format
            raise NotImplementedError(f"Fixed point conversion not implemented for {self.language}")
    
    def convert_number_to_fixed_point_from_scaled(self, scaled_value: int) -> str:
        """Convert an already scaled integer value to target language's fixed-point representation."""
        # Handle sign
        is_positive = scaled_value >= 0
        abs_value = abs(scaled_value)
        
        if self.language == "zokrates":
            return f"i64{{sgn:{'true' if is_positive else 'false'}, v: {abs_value}}}"
        elif self.language == "rust":
            return f"from_scaled_i64({scaled_value})"
        else:
            # For other languages, implement their specific format
            raise NotImplementedError(f"Fixed point conversion not implemented for {self.language}")
    
    def convert_number_to_field(self, value: float) -> str:
        """Convert number to field format for language input."""
        # Use same precision as old notebook: 10^10
        converted_value = int(np.round(value * 10000000000, 0))
        
        is_positive = converted_value >= 0
        abs_value = abs(converted_value)
        
        if self.language == "zokrates":
            return f'{'1' if is_positive else '0'}", "{abs_value}'
        elif self.language == "rust":
            return f"{converted_value}"
        else:
            raise NotImplementedError(f"Field conversion not implemented for {self.language}")
    
    def convert_number_to_input(self, value: float) -> str:
        """Convert number to input format for language."""
        # Use same precision as old notebook: 10^10
        converted_value = int(np.round(value * 10000000000, 0))
        
        is_positive = converted_value >= 0
        abs_value = abs(converted_value)
        
        if self.language == "zokrates":
            return f'{{"sgn":{'true' if is_positive else 'false'}, "v": "{abs_value}"}}'
        elif self.language == "rust":
            return f"{converted_value}"
        else:
            raise NotImplementedError(f"Input conversion not implemented for {self.language}")
    
    def convert_test_data_to_field_list(self, test_data: List[float]) -> str:
        """Convert test data array to field list format."""
        field_parts = []
        for value in test_data:
            field_parts.append(self.convert_number_to_field(value))
        
        if self.language == "zokrates":
            return '"' + '", "'.join(field_parts) + '"'
        elif self.language == "rust":
            return "vec![" + ', '.join(field_parts) + "]"
        else:
            return ', '.join(field_parts)
    
    def _generate_tree_logic(self, tree_structure: Dict, feature_indices: List[str], depth: int) -> str:
        """Generate code for a single decision tree."""
        indent_config = self.config["indentation"]
        indent_char = " " if indent_config["type"] == "spaces" else "\t"
        indent = indent_char * (indent_config["size"] * depth)
        
        # Base case: leaf node
        if 'leaf' in tree_structure:
            # Convert leaf value with precision multiplier (10^10 as in old notebook)
            leaf_value_scaled = int(np.round(tree_structure['leaf'] * 10000000000, 0))
            leaf_value = self.convert_number_to_fixed_point_from_scaled(leaf_value_scaled)
            if self.language == "rust":
                return f"{indent}{leaf_value}\n"
            else:
                return f"{indent} {leaf_value}\n"
        
        # Split node
        feature_idx = int(tree_structure['split'][1:])  # Remove 'f' prefix
        feature_index = feature_indices[feature_idx]  # This should be a string like "0", "1", "2"...
        
        # Convert threshold with precision multiplier
        threshold_scaled = int(np.round(tree_structure['split_condition'] * 10000000000, 0))
        threshold = self.convert_number_to_fixed_point_from_scaled(threshold_scaled)
        
        # Generate condition - use numeric index for array access
        if self.language == "rust":
            if depth == 1:
                condition_template = "let tree_result = if fixed_le(f[{}], {}) {{"
            else:
                condition_template = "if fixed_le(f[{}], {}) {{"
        else:  # ZoKrates
            if depth == 1:
                condition_template = "x = if i64_le(f[{}], {}) {{"
            else:
                condition_template = "if i64_le(f[{}], {}){{"
        
        result = f"{indent}{condition_template.format(feature_index, threshold)}\n"
        
        # Left branch (yes)
        yes_branch = tree_structure['children'][0]
        result += self._generate_tree_logic(yes_branch, feature_indices, depth + 1)
        result += f"{indent}}} else {{\n"
        
        # Right branch (no)
        no_branch = tree_structure['children'][1]
        result += self._generate_tree_logic(no_branch, feature_indices, depth + 1)
        
        if self.language == "rust":
            if depth == 1:
                result += f"{indent}}};\n"
            else:
                result += f"{indent}}}\n"
        else:  # ZoKrates
            if depth == 1:
                result += f"{indent} }};\n"
                result += f"{indent} y = i64_add(y, x);\n"
            else:
                result += f"{indent} }}\n"
        
        return result
    
    def convert_xgboost_to_code(self, booster, feature_names: List[str], num_trees: int) -> str:
        """
        Convert XGBoost booster to target language code.
        
        Args:
            booster: XGBoost booster object
            feature_names: List of feature names (will be converted to indices)
            num_trees: Maximum number of trees to convert
            
        Returns:
            Generated code string
        """
        # Convert feature names to indices for ZoKrates array access
        feature_indices = self.feature_names_to_indices(feature_names)
        
        # Get tree dumps
        trees_dump = booster.get_dump(dump_format='json')
        
        # Start building code
        code_parts = []
        
        # Add header (helper functions, data types)
        if "header" in self.templates:
            code_parts.append(self.templates["header"])
        
        # Generate tree code
        tree_codes = []
        for tree_idx, tree_dump in enumerate(trees_dump):
            if tree_idx >= num_trees:
                break
                
            tree_structure = json.loads(tree_dump)
            tree_logic = self._generate_tree_logic(tree_structure, feature_indices, 1)
            
            if "tree" in self.templates:
                tree_code = self.templates["tree"].format(
                    tree_idx=tree_idx,
                    tree_logic=tree_logic
                )
                tree_codes.append(tree_code)
        
        # Generate main function
        if "main" in self.templates:
            main_code = self.templates["main"].format(
                num_features=len(feature_names),
                tree_code="\n".join(tree_codes)
            )
            code_parts.append(main_code)
        
        # Add tests for Rust
        if self.language == "rust" and "test" in self.templates:
            test_code = self.templates["test"].format(
                num_features=len(feature_names)
            )
            code_parts.append(test_code)
        
        return "\n".join(code_parts)
    
    def feature_names_to_indices(self, feature_names: List[str]) -> List[str]:
        """Convert feature names to index format."""
        return [str(i) for i in range(len(feature_names))]
    
    def save_code_to_file(self, code: str, filename: str) -> None:
        """Save generated code to file with appropriate extension."""
        extension = self.config["file_extension"]
        if not filename.endswith(extension):
            filename += extension
        
        with open(filename, 'w') as f:
            f.write(code)
        
        # For Rust, also generate Cargo.toml
        if self.language == "rust" and "cargo" in self.templates:
            directory = os.path.dirname(filename) if os.path.dirname(filename) else "."
            cargo_path = os.path.join(directory, "Cargo.toml")
            with open(cargo_path, 'w') as f:
                f.write(self.templates["cargo"])