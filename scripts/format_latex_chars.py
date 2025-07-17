import re
import sys

def format_latex_chars(content):
    # Define basic UTF to LaTeX conversions
    replacements = [
        # Greek letters
        ('α', '\\alpha'),
        ('β', '\\beta'),
        ('γ', '\\gamma'),
        ('δ', '\\delta'),
        ('ε', '\\epsilon'),
        ('θ', '\\theta'),
        ('λ', '\\lambda'),
        ('μ', '\\mu'),
        ('π', '\\pi'),
        ('σ', '\\sigma'),
        ('τ', '\\tau'),
        ('ω', '\\omega'),
        ('Δ', '\\Delta'),
        ('Σ', '\\Sigma'),
        ('Ω', '\\Omega'),
        
        # Handle subscripts/superscripts

        # Handle superscript characters
        ('²', '$^2$'),
        
        # Handle arrows
        ('→', '$\\rightarrow$'),
        ('‑', '-')
    ]
    
    modified_content = content
    for utf_char, latex_char in replacements:
        modified_content = modified_content.replace(utf_char, latex_char)
    
    return modified_content

def process_file(filepath):
    try:
        # Read the file
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Format characters
        modified_content = format_latex_chars(content)
        
        # Write back to file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(modified_content)
            
        print(f"Successfully processed {filepath}")
        
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python format_latex_chars.py <tex_file>", file=sys.stderr)
        sys.exit(1)
    
    process_file(sys.argv[1])
