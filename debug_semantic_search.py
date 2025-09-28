#!/usr/bin/env python3
"""
Test semantic search directly without needing the server
"""
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_search_directly():
    """Test the search function directly"""
    
    test_cases = [
        {
            "name": "Strep Throat",
            "notes": "Patient presents with sore throat, fever, and fatigue for 3 days. Rapid strep test positive for Group A Streptococcus."
        },
        {
            "name": "Diabetes Check", 
            "notes": "Patient with Type 2 diabetes mellitus for routine follow-up. Blood glucose levels stable."
        },
        {
            "name": "Headache",
            "notes": "Patient complains of severe headache for 2 days. No fever or neck stiffness."
        },
        {
            "name": "Hypertension",
            "notes": "Patient with high blood pressure, needs medication adjustment. BP 150/90."
        }
    ]
    
    try:
        from app.ai.semantic_search import enhanced_semantic_search
        
        print("TESTING SEMANTIC SEARCH DIRECTLY")
        print("=" * 60)
        
        for i, case in enumerate(test_cases):
            print(f"\n{i+1}. {case['name']}")
            print(f"Input: {case['notes'][:60]}...")
            
            try:
                icd_results, proc_results, hcpcs_results = enhanced_semantic_search(case['notes'], top_n=3)
                
                print("ICD Results:")
                if icd_results:
                    for j, result in enumerate(icd_results):
                        sim = result.get('similarity', 0) * 100
                        print(f"  {j+1}. {result['code']} - {result['description'][:50]}... (sim: {sim:.1f}%)")
                else:
                    print("  No ICD results")
                
                print("CPT Results:")
                if proc_results:
                    for j, result in enumerate(proc_results):
                        sim = result.get('similarity', 0) * 100
                        print(f"  {j+1}. {result['code']} - {result['description'][:50]}... (sim: {sim:.1f}%)")
                else:
                    print("  No CPT results")
                
                if hcpcs_results:
                    print("HCPCS Results:")
                    for j, result in enumerate(hcpcs_results):
                        sim = result.get('similarity', 0) * 100
                        print(f"  {j+1}. {result['code']} - {result['description'][:50]}... (sim: {sim:.1f}%)")
                
                # Check if results are different
                current_icd_codes = [r['code'] for r in icd_results]
                current_cpt_codes = [r['code'] for r in proc_results]
                
                if i == 0:
                    first_icd_codes = current_icd_codes
                    first_cpt_codes = current_cpt_codes
                else:
                    icd_same = set(current_icd_codes) == set(first_icd_codes)
                    cpt_same = set(current_cpt_codes) == set(first_cpt_codes)
                    
                    if icd_same and cpt_same:
                        print("  ⚠️  WARNING: Same codes as first test case!")
                    else:
                        print("  ✓ Different codes from first test case")
                
            except Exception as e:
                print(f"  ERROR in search: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("If all test cases return identical codes, the search logic is broken")
        print("If codes vary by medical condition, the search logic is working")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_database_connection():
    """Test if we can load the medical code tables"""
    print("\nTESTING DATABASE CONNECTION")
    print("=" * 40)
    
    try:
        from app.ai.utils import load_table
        
        tables = ['icd10_codes', 'procedural_codes', 'hcpcs_codes']
        
        for table in tables:
            df = load_table(table)
            print(f"{table}: {len(df)} records")
            
            if not df.empty:
                # Show a few sample codes
                print(f"  Sample codes: {df['code'].head(3).tolist()}")
            else:
                print(f"  ⚠️  WARNING: {table} is empty!")
    
    except Exception as e:
        print(f"Database connection error: {e}")

if __name__ == "__main__":
    test_database_connection()
    test_search_directly()