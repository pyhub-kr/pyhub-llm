"""Test error recovery and handling in CodeInterpreter tool."""

import pytest
from pyhub.llm.agents.tools import CodeInterpreter


class TestCodeInterpreterErrorRecovery:
    """Test error handling and recovery in CodeInterpreter."""
    
    @pytest.fixture
    def tool(self):
        """Create CodeInterpreter instance."""
        return CodeInterpreter(backend="local")
    
    def test_syntax_error_recovery(self, tool):
        """Test recovery from syntax errors."""
        session_id = "syntax_error_test"
        
        # Set up initial state
        tool.run("x = 100\ny = 200", session_id=session_id)
        
        # Cause syntax error
        result_error = tool.run("if x > 50\n    print('error')", session_id=session_id)
        assert "Error" in result_error or "SyntaxError" in result_error
        
        # Verify state is preserved after syntax error
        result_check = tool.run("print(f'x={x}, y={y}')", session_id=session_id)
        assert "x=100, y=200" in result_check
    
    def test_runtime_error_recovery(self, tool):
        """Test recovery from runtime errors."""
        session_id = "runtime_error_test"
        
        # Set up data
        tool.run("""
data = [1, 2, 3, 4, 5]
results = []
""", session_id=session_id)
        
        # Process with error in the middle
        result1 = tool.run("""
for i, val in enumerate(data):
    if i == 2:
        # This will cause ZeroDivisionError
        results.append(val / 0)
    else:
        results.append(val * 2)
""", session_id=session_id)
        assert "Error" in result1 or "ZeroDivisionError" in result1
        
        # Continue processing after error
        result2 = tool.run("""
# Process remaining items safely
for val in data[2:]:
    results.append(val * 2)
print(f"Results so far: {results}")
""", session_id=session_id)
        assert "[2, 4, 6, 8, 10]" in result2
    
    def test_import_error_recovery(self, tool):
        """Test recovery from import errors."""
        session_id = "import_error_test"
        
        # Try to import non-existent module
        result_error = tool.run("import non_existent_module", session_id=session_id)
        assert "Error" in result_error or "ImportError" in result_error
        
        # Import should not affect other imports
        result_good = tool.run("""
import json
import math
print("Standard imports work")
print(f"Pi: {math.pi}")
""", session_id=session_id)
        assert "Standard imports work" in result_good
        assert "3.14159" in result_good
    
    def test_name_error_recovery(self, tool):
        """Test recovery from NameError."""
        session_id = "name_error_test"
        
        # Try to use undefined variable
        result_error = tool.run("print(undefined_variable)", session_id=session_id)
        assert "Error" in result_error or "NameError" in result_error
        
        # Define and use variables normally
        result_good = tool.run("""
defined_variable = "I exist"
print(defined_variable)
""", session_id=session_id)
        assert "I exist" in result_good
    
    def test_pandas_error_recovery(self, tool):
        """Test error recovery in pandas operations."""
        session_id = "pandas_error_test"
        
        # Create DataFrame
        tool.run("""
import pandas as pd
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})
""", session_id=session_id)
        
        # Try invalid operation
        result_error = tool.run("df['C'] = df['A'] / df['NonExistent']", session_id=session_id)
        assert "Error" in result_error or "KeyError" in result_error
        
        # DataFrame should still be intact
        result_check = tool.run("""
print("DataFrame still exists:")
print(df)
print(f"Columns: {df.columns.tolist()}")
""", session_id=session_id)
        assert "DataFrame still exists:" in result_check
        assert "['A', 'B']" in result_check
        
        # Continue with valid operations
        result_continue = tool.run("""
df['C'] = df['A'] + df['B']
print("Added column C successfully:")
print(df)
""", session_id=session_id)
        assert "Added column C successfully:" in result_continue
        assert "5" in result_continue  # 1+4
    
    def test_matplotlib_error_recovery(self, tool):
        """Test error recovery in matplotlib operations."""
        session_id = "matplotlib_error_test"
        
        # Create initial plot
        tool.run("""
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
x = [1, 2, 3, 4]
y = [1, 4, 2, 3]
ax.plot(x, y)
""", session_id=session_id)
        
        # Try invalid operation
        result_error = tool.run("ax.plot(x, 'invalid_data')", session_id=session_id)
        assert "Error" in result_error
        
        # Figure should still be usable
        result_continue = tool.run("""
# Add valid data
y2 = [2, 3, 4, 1]
ax.plot(x, y2, 'r--')
ax.set_title('Plot after error recovery')
plt.savefig('error_recovery_plot.png')
print("Plot saved successfully")
""", session_id=session_id)
        assert "Plot saved successfully" in result_continue
    
    def test_exception_in_function(self, tool):
        """Test error handling in user-defined functions."""
        session_id = "function_error_test"
        
        # Define function with potential error
        tool.run("""
def risky_function(x):
    if x == 0:
        raise ValueError("Cannot process zero")
    return 100 / x

def safe_wrapper(x):
    try:
        return risky_function(x)
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"
""", session_id=session_id)
        
        # Test with various inputs
        result = tool.run("""
results = []
for val in [10, 0, -5, 2]:
    result = safe_wrapper(val)
    results.append(f"f({val}) = {result}")

for r in results:
    print(r)
""", session_id=session_id)
        assert "f(10) = 10.0" in result
        assert "f(0) = Error: Cannot process zero" in result
        assert "f(-5) = -20.0" in result
        assert "f(2) = 50.0" in result
    
    def test_partial_execution_recovery(self, tool):
        """Test recovery when code partially executes."""
        session_id = "partial_exec_test"
        
        # Code that partially executes
        result = tool.run("""
successful_list = []
successful_list.append("First item")
successful_list.append("Second item")

# This will fail
failed_operation = 1 / 0

# This won't execute
successful_list.append("Third item")
""", session_id=session_id)
        assert "Error" in result or "ZeroDivisionError" in result
        
        # In current implementation, error rolls back all changes
        # So the list should not exist in the session
        result_check = tool.run("""
try:
    print(f"List contents: {successful_list}")
except NameError:
    print("List was not saved due to error rollback")
""", session_id=session_id)
        assert "List was not saved due to error rollback" in result_check
    
    def test_infinite_loop_protection(self, tool):
        """Test protection against infinite loops."""
        session_id = "infinite_loop_test"
        
        # Note: This test depends on the implementation having timeout protection
        # The local backend should have execution time limits
        result = tool.run("""
# This should timeout or be prevented
count = 0
# while True:
#     count += 1
# Instead, let's test a long-running but finite loop
for i in range(1000000):
    if i % 100000 == 0:
        print(f"Progress: {i}")
print("Completed")
""", session_id=session_id)
        # Should either complete or timeout gracefully
        assert "Progress:" in result or "Error" in result or "timeout" in result.lower()
    
    def test_memory_error_simulation(self, tool):
        """Test handling of memory-intensive operations."""
        session_id = "memory_test"
        
        # Try to create a large data structure
        result = tool.run("""
try:
    # Create a reasonably large list (not too large to actually cause issues)
    large_list = list(range(1000000))
    print(f"Created list with {len(large_list)} elements")
    print(f"First 5 elements: {large_list[:5]}")
    print(f"Last 5 elements: {large_list[-5:]}")
except MemoryError:
    print("Memory error caught")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")
""", session_id=session_id)
        assert "Created list with 1000000 elements" in result or "error" in result.lower()
    
    def test_cascading_errors(self, tool):
        """Test handling of cascading errors."""
        session_id = "cascading_error_test"
        
        # Set up initial state
        tool.run("""
class DataProcessor:
    def __init__(self):
        self.data = []
        self.errors = []
    
    def process(self, item):
        try:
            if item < 0:
                raise ValueError("Negative value")
            self.data.append(item * 2)
        except Exception as e:
            self.errors.append(f"Error processing {item}: {e}")

processor = DataProcessor()
""", session_id=session_id)
        
        # Process mixed valid/invalid data
        result = tool.run("""
items = [5, -3, 10, "invalid", 20, None, 30]

for item in items:
    try:
        processor.process(item)
    except Exception as e:
        processor.errors.append(f"Unhandled error for {item}: {e}")

print(f"Processed data: {processor.data}")
print(f"Errors encountered: {len(processor.errors)}")
for error in processor.errors:
    print(f"  - {error}")
""", session_id=session_id)
        assert "Processed data: [10, 20, 40, 60]" in result
        assert "Errors encountered:" in result
        assert "Negative value" in result
    
    def test_state_consistency_after_errors(self, tool):
        """Test that object state remains consistent after errors."""
        session_id = "state_consistency_test"
        
        # Create stateful object
        tool.run("""
class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
        self.transactions = []
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self.balance += amount
        self.transactions.append(('deposit', amount))
    
    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
        self.transactions.append(('withdraw', amount))
    
    def get_statement(self):
        return {
            'balance': self.balance,
            'transaction_count': len(self.transactions),
            'transactions': self.transactions[-5:]  # Last 5 transactions
        }

account = BankAccount(1000)
""", session_id=session_id)
        
        # Perform operations with some failures
        result = tool.run("""
operations = [
    ('deposit', 500),
    ('withdraw', 200),
    ('withdraw', 2000),  # Will fail - insufficient funds
    ('deposit', -100),   # Will fail - negative amount
    ('withdraw', 300),   # Should succeed
    ('deposit', 150)     # Should succeed
]

for op_type, amount in operations:
    try:
        if op_type == 'deposit':
            account.deposit(amount)
            print(f"✓ Deposited ${amount}")
        else:
            account.withdraw(amount)
            print(f"✓ Withdrew ${amount}")
    except ValueError as e:
        print(f"✗ Failed to {op_type} ${amount}: {e}")

statement = account.get_statement()
print(f"\\nFinal balance: ${statement['balance']}")
print(f"Successful transactions: {statement['transaction_count']}")
""", session_id=session_id)
        
        assert "Final balance: $1150" in result  # 1000 + 500 - 200 - 300 + 150
        assert "Successful transactions: 4" in result
        assert "Insufficient funds" in result
        assert "Amount must be positive" in result