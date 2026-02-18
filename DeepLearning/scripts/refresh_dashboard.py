
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from reporting.html_report import AdvancedReportGenerator

def main():
    reports_dir = Path(__file__).parent.parent / 'outputs' / 'reports'
    if not reports_dir.exists():
        print(f"Reports directory not found: {reports_dir}")
        return
    
    print(f"Refreshing dashboard and reports in: {reports_dir}")
    generator = AdvancedReportGenerator(reports_dir)
    
    # Regenerate Dashboard
    generator.generate_dashboard()
    
    # Regenerate All Detailed Reports
    import json
    json_files = list(reports_dir.glob("*.json"))
    print(f"Found {len(json_files)} run files. Regenerating detailed reports...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                run_data = json.load(f)
            
            output_path = reports_dir / (json_file.stem + ".html")
            generator.generate_detailed_report(run_data, output_path)
        except Exception as e:
            print(f"Failed to regenerate {json_file.name}: {e}")

    print("Done! All reports updated.")

if __name__ == '__main__':
    main()
