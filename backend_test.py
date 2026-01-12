import requests
import sys
import time
from datetime import datetime

class ABTestAPITester:
    def __init__(self, base_url="https://finance-testing.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict) and 'message' in response_data:
                        print(f"   Message: {response_data['message']}")
                    return True, response_data
                except:
                    return True, {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_detail = response.json()
                    print(f"   Error: {error_detail}")
                except:
                    print(f"   Response: {response.text[:200]}")
                self.failed_tests.append(f"{name}: Expected {expected_status}, got {response.status_code}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout after {timeout}s")
            self.failed_tests.append(f"{name}: Request timeout")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.failed_tests.append(f"{name}: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test API root endpoint"""
        return self.run_test("API Root", "GET", "", 200)

    def test_generate_data(self):
        """Test data generation endpoint"""
        print("\n📊 Testing data generation (this may take 10-15 seconds)...")
        success, response = self.run_test(
            "Generate Experiment Data", 
            "POST", 
            "experiment/generate", 
            200,
            timeout=60
        )
        if success and response:
            sample_size = response.get('sample_size', 0)
            if sample_size == 20000:
                print(f"   ✅ Correct sample size: {sample_size}")
            else:
                print(f"   ⚠️  Unexpected sample size: {sample_size}")
        return success

    def test_experiment_config(self):
        """Test experiment configuration endpoint"""
        success, response = self.run_test(
            "Get Experiment Config", 
            "GET", 
            "experiment/config", 
            200
        )
        if success and response:
            required_fields = ['name', 'hypothesis', 'sample_size', 'control_variant', 'treatment_variant']
            missing_fields = [field for field in required_fields if field not in response]
            if not missing_fields:
                print(f"   ✅ All required fields present")
                print(f"   Sample size: {response.get('sample_size', 'N/A')}")
            else:
                print(f"   ⚠️  Missing fields: {missing_fields}")
        return success

    def test_metrics_summary(self):
        """Test metrics summary endpoint"""
        success, response = self.run_test(
            "Get Metrics Summary", 
            "GET", 
            "metrics/summary", 
            200
        )
        if success and response:
            required_metrics = [
                'conversion_rate_control', 'conversion_rate_treatment',
                'ctr_control', 'ctr_treatment',
                'revenue_per_user_control', 'revenue_per_user_treatment',
                'relative_uplift', 'absolute_uplift'
            ]
            missing_metrics = [metric for metric in required_metrics if metric not in response]
            if not missing_metrics:
                print(f"   ✅ All metrics present")
                print(f"   Control conversion: {response.get('conversion_rate_control', 0):.3f}")
                print(f"   Treatment conversion: {response.get('conversion_rate_treatment', 0):.3f}")
                print(f"   Relative uplift: {response.get('relative_uplift', 0):.1f}%")
            else:
                print(f"   ⚠️  Missing metrics: {missing_metrics}")
        return success

    def test_statistical_tests(self):
        """Test statistical tests endpoint"""
        success, response = self.run_test(
            "Get Statistical Tests", 
            "GET", 
            "analysis/statistical-tests", 
            200
        )
        if success and response:
            tests = response.get('tests', [])
            if tests:
                print(f"   ✅ Found {len(tests)} statistical tests")
                significant_tests = [t for t in tests if t.get('significant', False)]
                print(f"   Significant tests: {len(significant_tests)}/{len(tests)}")
                for test in tests:
                    p_val = test.get('p_value', 1)
                    sig = "✅" if test.get('significant') else "❌"
                    print(f"   {sig} {test.get('test_name', 'Unknown')}: p={p_val:.4f}")
            else:
                print(f"   ⚠️  No tests found")
        return success

    def test_funnel_analysis(self):
        """Test funnel analysis endpoint"""
        success, response = self.run_test(
            "Get Funnel Analysis", 
            "GET", 
            "analysis/funnel", 
            200
        )
        if success and response:
            funnel = response.get('funnel', [])
            if funnel:
                print(f"   ✅ Found {len(funnel)} funnel steps")
                for step in funnel:
                    step_name = step.get('step', 'Unknown')
                    control_rate = step.get('control_rate', 0) * 100
                    treatment_rate = step.get('treatment_rate', 0) * 100
                    print(f"   {step_name}: Control {control_rate:.1f}%, Treatment {treatment_rate:.1f}%")
            else:
                print(f"   ⚠️  No funnel data found")
        return success

    def test_segment_analysis(self):
        """Test segment analysis endpoint"""
        success, response = self.run_test(
            "Get Segment Analysis", 
            "GET", 
            "analysis/segments", 
            200
        )
        if success and response:
            segments = response.get('segments', [])
            if segments:
                print(f"   ✅ Found {len(segments)} segments")
                segment_types = set(s.get('segment_name') for s in segments)
                print(f"   Segment types: {', '.join(segment_types)}")
            else:
                print(f"   ⚠️  No segment data found")
        return success

    def test_timeseries_analysis(self):
        """Test timeseries analysis endpoint"""
        success, response = self.run_test(
            "Get Timeseries Analysis", 
            "GET", 
            "analysis/timeseries", 
            200
        )
        if success and response:
            timeseries = response.get('timeseries', [])
            if timeseries:
                print(f"   ✅ Found {len(timeseries)} time points")
                days = [t.get('day') for t in timeseries]
                print(f"   Days range: {min(days)} to {max(days)}")
            else:
                print(f"   ⚠️  No timeseries data found")
        return success

    def test_business_recommendations(self):
        """Test business recommendations endpoint"""
        success, response = self.run_test(
            "Get Business Recommendations", 
            "GET", 
            "recommendations", 
            200
        )
        if success and response:
            recommendations = response.get('recommendations', [])
            if recommendations:
                print(f"   ✅ Found {len(recommendations)} recommendations")
                priorities = [r.get('priority') for r in recommendations]
                priority_counts = {p: priorities.count(p) for p in set(priorities)}
                print(f"   Priority breakdown: {priority_counts}")
            else:
                print(f"   ⚠️  No recommendations found")
        return success

def main():
    print("🚀 Starting A/B Test Analytics API Testing")
    print("=" * 60)
    
    tester = ABTestAPITester()
    
    # Test sequence
    print("\n📋 PHASE 1: Basic API Health Check")
    tester.test_root_endpoint()
    
    print("\n📋 PHASE 2: Data Generation")
    data_generated = tester.test_generate_data()
    
    if not data_generated:
        print("\n❌ Data generation failed. Cannot proceed with other tests.")
        print(f"\n📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
        if tester.failed_tests:
            print("\n❌ Failed Tests:")
            for failure in tester.failed_tests:
                print(f"   - {failure}")
        return 1
    
    # Wait a moment for data to be fully processed
    print("\n⏳ Waiting 2 seconds for data processing...")
    time.sleep(2)
    
    print("\n📋 PHASE 3: Configuration & Metrics")
    tester.test_experiment_config()
    tester.test_metrics_summary()
    
    print("\n📋 PHASE 4: Analysis Endpoints")
    tester.test_statistical_tests()
    tester.test_funnel_analysis()
    tester.test_segment_analysis()
    tester.test_timeseries_analysis()
    tester.test_business_recommendations()
    
    # Final results
    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.failed_tests:
        print("\n❌ FAILED TESTS:")
        for failure in tester.failed_tests:
            print(f"   - {failure}")
        return 1
    else:
        print("\n✅ ALL TESTS PASSED! Backend API is working correctly.")
        return 0

if __name__ == "__main__":
    sys.exit(main())