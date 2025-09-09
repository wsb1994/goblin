#!/usr/bin/env python3
"""
End-to-end test of goblin integration
"""

import sys
import os
import json
import subprocess
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from model.integration import ModelRegistry
from model.pipeline import create_composable_model


def setup_test_models():
    """Create and register test models"""
    print("🏗️  Setting up test models...")
    
    registry = ModelRegistry()
    
    # Create a simple test model
    try:
        model = create_composable_model(
            dataset_type='synthetic',
            feature_type='statistical',
            architecture_type='logistic',
            model_name='test_simple_model',
            dataset_config={
                'num_samples': 200,
                'positive_ratio': 0.3,
                'random_state': 42
            },
            feature_config={},
            architecture_config={'C': 1.0}
        )
        
        print("   Training test model...")
        model.train()
        
        # Save model
        os.makedirs('model_outputs', exist_ok=True)
        model_path = 'model_outputs/test_simple_model.json'
        model.save(model_path)
        
        # Register model
        registry.register_model(
            model_name='test_simple_model',
            model_type='composable',
            model_path=model_path,
            config=model.config.__dict__,
            description='Test model for goblin integration',
            tags=['test', 'synthetic', 'statistical', 'logistic']
        )
        
        print("   ✅ Test model created and registered")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create test model: {e}")
        return False


def test_goblin_adapter():
    """Test the goblin adapter directly"""
    print("\n🔧 Testing Goblin Adapter...")
    
    test_cases = [
        {
            'name': 'List Models',
            'command': ['python', '-m', 'model.integration.goblin_adapter', 'test_simple_model', '--list-models'],
            'expect': 'models'
        },
        {
            'name': 'Test Model Loading',
            'command': ['python', '-m', 'model.integration.goblin_adapter', 'test_simple_model', '--test'],
            'expect': 'true'
        },
        {
            'name': 'Simple Prediction',
            'command': ['python', '-m', 'model.integration.goblin_adapter', 'test_simple_model', '--input', 'This is a test message'],
            'expect': 'predictions'
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"   Testing: {test_case['name']}")
        
        try:
            result = subprocess.run(
                test_case['command'],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                
                if test_case['expect'] in output:
                    print(f"      ✅ Passed")
                    results.append(True)
                else:
                    print(f"      ❌ Failed - Expected '{test_case['expect']}' in output")
                    print(f"         Output: {output[:100]}...")
                    results.append(False)
            else:
                print(f"      ❌ Failed - Return code: {result.returncode}")
                print(f"         Error: {result.stderr[:100]}...")
                results.append(False)
                
        except subprocess.TimeoutExpired:
            print(f"      ❌ Failed - Timeout")
            results.append(False)
        except Exception as e:
            print(f"      ❌ Failed - Exception: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results)
    print(f"   📊 Adapter tests: {sum(results)}/{len(results)} passed ({success_rate:.1%})")
    
    return success_rate > 0.8


def test_goblin_engine_integration():
    """Test integration with goblin engine"""
    print("\n⚙️  Testing Goblin Engine Integration...")
    
    try:
        # Import goblin engine
        sys.path.append(os.path.join(os.getcwd(), 'goblin_backend'))
        from goblin_backend.engine.engine import Engine
        
        # Create engine instance
        engine = Engine()
        
        # Define scripts for our test model
        script_config = '''
        [test_simple_model]
        name = "test_simple_model"
        command = "python -m model.integration.goblin_adapter test_simple_model --input \\"Test message for hate detection\\""
        timeout = 30000
        test_command = "python -m model.integration.goblin_adapter test_simple_model --test"
        require_test = false
        
        [compare_models]
        name = "compare_models"  
        command = "python goblin_backend/scripts/compare_models/main.py"
        timeout = 10000
        '''
        
        print("   Loading script configuration...")
        engine.load_scripts(script_config)
        
        # Define execution plan - just test the model, skip comparison for now
        plan_config = '''
        [[steps]]
        name = "TestModel"
        function = "test_simple_model"
        inputs = ["default_input"]
        '''
        
        print("   Loading execution plan...")
        plan = engine.load_plan(plan_config)
        
        print("   Executing plan...")
        results = engine.execute_plan(plan)
        
        print("   ✅ Goblin engine integration successful")
        
        # Print results summary
        if results:
            for step_name, result in results.items():
                print(f"      {step_name}: {len(str(result))} chars output")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Goblin engine integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_comparison_workflow():
    """Test the A/B comparison workflow"""
    print("\n🔬 Testing Model Comparison Workflow...")
    
    try:
        # Create a second test model for comparison
        registry = ModelRegistry()
        
        model2 = create_composable_model(
            dataset_type='synthetic',
            feature_type='statistical', 
            architecture_type='svm',
            model_name='test_svm_model',
            dataset_config={
                'num_samples': 200,
                'positive_ratio': 0.3,
                'random_state': 42
            },
            feature_config={},
            architecture_config={'C': 0.1, 'kernel': 'linear'}
        )
        
        print("   Training second model...")
        model2.train()
        
        model2_path = 'model_outputs/test_svm_model.json'
        model2.save(model2_path)
        
        registry.register_model(
            model_name='test_svm_model',
            model_type='composable',
            model_path=model2_path,
            config=model2.config.__dict__,
            description='Second test model for comparison',
            tags=['test', 'synthetic', 'statistical', 'svm']
        )
        
        # Test both models on same input
        test_input = ["This is hate speech", "This is normal text", "What a beautiful day"]
        
        print("   Running model comparisons...")
        
        # Get predictions from both models
        cmd1 = ['python', '-m', 'model.integration.goblin_adapter', 'test_simple_model', '--input', json.dumps(test_input)]
        cmd2 = ['python', '-m', 'model.integration.goblin_adapter', 'test_svm_model', '--input', json.dumps(test_input)]
        
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        
        if result1.returncode == 0 and result2.returncode == 0:
            print("   ✅ Both models ran successfully")
            
            # Parse results
            try:
                pred1 = json.loads(result1.stdout)
                pred2 = json.loads(result2.stdout)
                
                print(f"      Model 1 avg prob: {pred1['summary']['average_probability']:.3f}")
                print(f"      Model 2 avg prob: {pred2['summary']['average_probability']:.3f}")
                
                return True
            except json.JSONDecodeError:
                print("   ❌ Failed to parse model outputs")
                return False
        else:
            print("   ❌ Model comparison failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Comparison workflow failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    try:
        import shutil
        
        # Remove test models
        if os.path.exists('model_outputs/test_simple_model.json'):
            os.remove('model_outputs/test_simple_model.json')
        if os.path.exists('model_outputs/test_svm_model.json'):
            os.remove('model_outputs/test_svm_model.json')
        
        # Remove related files
        for pattern in ['*_feature_extractor.pkl', '*_architecture.pkl', '*_lgb.txt']:
            import glob
            for file in glob.glob(f'model_outputs/{pattern}'):
                os.remove(file)
        
        # Remove registry if it was created for testing
        if os.path.exists('model_registry.json'):
            with open('model_registry.json', 'r') as f:
                registry_data = json.load(f)
            
            # Remove test models from registry
            test_models = ['test_simple_model', 'test_svm_model']
            for model in test_models:
                registry_data.pop(model, None)
            
            # Write back cleaned registry
            with open('model_registry.json', 'w') as f:
                json.dump(registry_data, f, indent=2)
        
        print("   ✅ Cleanup completed")
        
    except Exception as e:
        print(f"   ⚠️  Cleanup had issues: {e}")


def main():
    """Run all integration tests"""
    print("🧪 Goblin Integration Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Run tests in sequence
    tests = [
        ("Setup Test Models", setup_test_models),
        ("Goblin Adapter", test_goblin_adapter),
        ("Goblin Engine Integration", test_goblin_engine_integration),
        ("Model Comparison Workflow", test_model_comparison_workflow)
    ]
    
    for test_name, test_func in tests:
        print(f"\n🏃 Running: {test_name}")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            test_results.append((test_name, False))
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    total = len(test_results)
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 All integration tests passed! Goblin integration is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)