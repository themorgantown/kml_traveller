#!/usr/bin/env python3
"""
Fix ampersand symbols in KML file for Google Earth compatibility.
"""

def fix_kml_ampersands(input_file, output_file=None):
    """Fix unescaped ampersands in KML file."""
    if output_file is None:
        output_file = input_file
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace unescaped ampersands with proper XML entities
    # We need to be careful not to double-escape already escaped ampersands
    # First, temporarily replace already escaped ampersands
    content = content.replace('&amp;', '<<<TEMP_AMP>>>')
    content = content.replace('&lt;', '<<<TEMP_LT>>>')
    content = content.replace('&gt;', '<<<TEMP_GT>>>')
    content = content.replace('&quot;', '<<<TEMP_QUOT>>>')
    content = content.replace('&apos;', '<<<TEMP_APOS>>>')
    
    # Now replace all remaining ampersands
    content = content.replace('&', '&amp;')
    
    # Restore the temporarily replaced entities
    content = content.replace('<<<TEMP_AMP>>>', '&amp;')
    content = content.replace('<<<TEMP_LT>>>', '&lt;')
    content = content.replace('<<<TEMP_GT>>>', '&gt;')
    content = content.replace('<<<TEMP_QUOT>>>', '&quot;')
    content = content.replace('<<<TEMP_APOS>>>', '&apos;')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed ampersands in {input_file}")
    if output_file != input_file:
        print(f"Output saved to {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_kml.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_kml_ampersands(input_file, output_file)
