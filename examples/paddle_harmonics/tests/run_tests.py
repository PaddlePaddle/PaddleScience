import unittest
import os

def load_tests(loader, standard_tests, pattern):
    top_level_dir = None
    package_dir = os.path.dirname(__file__)
    print(f"Loading tests from: {package_dir}")
    
    for root, dirs, files in os.walk(package_dir):
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                module_name = file[:-3]
                module = f"{root.replace(os.sep, '.')}.{module_name}"
                test = loader.loadTestsFromName(module)
                standard_tests.addTests(test)
    
    return standard_tests

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", top_level_dir=None)
    runner = unittest.TextTestRunner()
    result = runner.run(suite)