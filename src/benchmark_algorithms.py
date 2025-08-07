# benchmark_algorithms.py
import random

# A collection of algorithms with different time complexities for benchmarking.

def algorithm_constant(n: int) -> int:
    """Constant time O(1) - Mathematical formula."""
    # Uses the arithmetic progression sum formula.
    return n * (n - 1) // 2


def algorithm_log_n_binary_search(n: int) -> int:
    """Logarithmic time O(log n) - Binary search."""
    # Create a sorted array for searching.
    data = list(range(n))
    # Search for an element that is definitely not in the list to ensure worst-case scenario.
    target = n + 1
    low, high = 0, len(data) - 1
    while low <= high:
        mid = (low + high) // 2
        if data[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return low


def algorithm_sqrt_n_primality_test(n: int) -> bool:
    """Time complexity O(sqrt(n)) - Simple primality test."""
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def algorithm_linear_sum(n: int) -> int:
    """Linear time O(n) - Summation. A basic example."""
    return sum(i for i in range(n))


def algorithm_linear_list_append(n: int) -> list:
    """Linear time O(n) - List creation using append."""
    result = []
    for i in range(n):
        result.append(i)
    return result


def algorithm_linear_string_concat(n: int) -> int:
    """Linear time O(n), but inefficient due to string immutability - String concatenation."""
    s = ""
    for i in range(n):
        # Inefficient operation, creates many new string objects.
        s += "*"
    return len(s)


def algorithm_linear_dict_creation(n: int) -> dict:
    """Linear time O(n) - Dictionary creation."""
    d = {}
    for i in range(n):
        d[i] = i * i
    return d


def algorithm_linear_factorial_iter(n: int) -> int:
    """Linear time O(n) - Iterative factorial (many multiplication operations)."""
    if n < 0: return 0
    res = 1
    for i in range(1, n + 1):
        res *= i
    return res


def algorithm_linear_recursive_power(n: int) -> int:
    """Linear time O(n) relative to the exponent - Recursive power calculation."""
    base = 2
    def power(b, p):
        if p == 0:
            return 1
        return b * power(b, p-1)
    # Using n as the exponent.
    return power(base, n)


def algorithm_n_log_n_sort(n: int) -> list:
    """Log-linear time O(n log n) - Sorting."""
    # Create a list of random numbers for demonstration.
    data = [random.randint(0, n) for _ in range(n)]
    return sorted(data)


def algorithm_quadratic_loops(n: int) -> int:
    """Quadratic time O(n^2) - Nested loops."""
    total = 0
    for i in range(n):
        for j in range(i):
            total += i * j
    return total


def algorithm_quadratic_list_search(n: int) -> int:
    """Quadratic time O(n^2) - Searching for common elements in lists."""
    list1 = list(range(n))
    list2 = list(range(n//2, n + n//2))
    count = 0
    for item1 in list1:
        for item2 in list2:
            if item1 == item2:
                count += 1
    return count


def algorithm_cubic_loops(n: int) -> int:
    """Cubic time O(n^3) - Triple nested loops.
    WARNING: Use n < 50, as it can be slow.
    """
    total = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total += 1
    return total


def algorithm_exponential_fib(n: int) -> int:
    """Exponential time O(2^n) - Naive Fibonacci recursion.
    WARNING: Use n < 15, otherwise it will be very slow.
    """
    if n <= 1:
        return n
    return algorithm_exponential_fib(n-1) + algorithm_exponential_fib(n-2)


def algorithm_factorial_permutations(n: int) -> int:
    """Factorial time O(n!) - Permutation generation.
    WARNING: Use n < 8, this is very resource-intensive.
    """
    count = 0
    def generate_permutations(arr, l, r):
        nonlocal count
        if l == r:
            count += 1
        else:
            for i in range(l, r + 1):
                arr[l], arr[i] = arr[i], arr[l]
                generate_permutations(arr, l + 1, r)
                arr[l], arr[i] = arr[i], arr[l] # backtrack

    elements = list(range(n))
    generate_permutations(elements, 0, len(elements) - 1)
    return count


# --- Dictionary to store all algorithms ---
# Key: human-readable name, Value: function reference
algorithms_collection = {
    "Constant_O(1)_Formula": algorithm_constant,
    "Logarithmic_O(log_n)_BinarySearch": algorithm_log_n_binary_search,
    "Sqrt_O(sqrt_n)_PrimalityTest": algorithm_sqrt_n_primality_test,
    "Linear_O(n)_Sum": algorithm_linear_sum,
    "Linear_O(n)_ListAppend": algorithm_linear_list_append,
    "Linear_O(n)_StringConcat": algorithm_linear_string_concat,
    "Linear_O(n)_DictCreation": algorithm_linear_dict_creation,
    "Linear_O(n)_FactorialIter": algorithm_linear_factorial_iter,
    "Linear_O(n)_RecursivePower": algorithm_linear_recursive_power,
    "N_Log_N_O(n_log_n)_Sort": algorithm_n_log_n_sort,
    "Quadratic_O(n^2)_NestedLoops": algorithm_quadratic_loops,
    "Quadratic_O(n^2)_ListSearch": algorithm_quadratic_list_search,
    "Cubic_O(n^3)_TripleLoops": algorithm_cubic_loops,
    "Exponential_O(2^n)_Fibonacci": algorithm_exponential_fib,
    "Factorial_O(n!)_Permutations": algorithm_factorial_permutations,
}
