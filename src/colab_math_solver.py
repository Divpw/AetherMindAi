import re
import numpy as np # Though direct use might be minimal, good for context
from sympy import (
    sympify, diff, integrate, solve, limit,
    sin, cos, tan, asin, acos, atan,
    pi, exp, log, sqrt, Abs, I, oo, zoo,
    symbols, Symbol, Function, Eq, Rational, Float, Integer,
    Derivative, Integral, Limit as SympyLimit, # SympyLimit to avoid conflict with built-in limit
    Matrix, det,
    latex, pretty,
    factorial, expand, factor, simplify,
    series,
    dsolve, Symbol as SympySymbol # dsolve for differential equations
)
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# Define common symbols that users might use without explicitly defining them.
x, y, z, t = symbols('x y z t')
a, b, c, d = symbols('a b c d') # More generic symbols
theta, phi = symbols('theta phi') # Common Greek letters for angles

# Global context for sympify, making common SymPy functions and constants available.
# This allows strings like "sin(x) + pi" to be parsed correctly.
SYMPY_GLOBALS = {
    # Symbols
    'x': x, 'y': y, 'z': z, 't': t,
    'a': a, 'b': b, 'c': c, 'd': d,
    'theta': theta, 'phi': phi,
    # Functions - Mathematical
    'sin': sin, 'cos': cos, 'tan': tan,
    'asin': asin, 'acos': acos, 'atan': atan,
    'exp': exp, 'log': log, 'sqrt': sqrt,
    'Abs': Abs, 'factorial': factorial,
    # Functions - SymPy specific operations that might be in a query string
    'integrate': integrate,
    'diff': diff,
    'solve': solve, # Note: solve is tricky with sympify if equations are not Eq()
    'limit': limit,
    'series': series,
    'expand': expand, 'factor': factor, 'simplify': simplify,
    'Derivative': Derivative, 'Integral': Integral, 'Limit': SympyLimit, 'Eq': Eq,
    'Matrix': Matrix, 'det': det,
    'dsolve': dsolve,
    # Constants
    'pi': pi, 'E': exp(1), 'I': I, # E for Euler's number, I for imaginary unit
    'oo': oo, 'zoo': zoo, # Infinity and complex infinity
    # Classes/Types (less likely to be typed directly in basic queries but good for completeness)
    'Symbol': SympySymbol, 'Function': Function,
    'Rational': Rational, 'Float': Float, 'Integer': Integer,
    # Common functions that might be used as unapplied functions
    'f': Function('f'), 'g': Function('g'), 'h': Function('h'),
}

# Transformations for parse_expr, allowing things like "2x" to be "2*x"
# and "sin x" to be "sin(x)"
PARSE_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


# To be populated further
print("colab_math_solver.py initialized with imports and SymPy context.")

def solve_math_query(query: str) -> str:
    """
    Parses and evaluates a mathematical query string using SymPy.

    Handles:
    - Basic arithmetic and direct SymPy expressions.
    - Calculus: diff, integrate, limit, series.
    - Algebra: solve, expand, factor, simplify.
    - Matrix operations: det.
    - Differential equations: dsolve.
    - Output formatting: latex, pretty.
    - Numerical evaluation: .evalf().

    Args:
        query (str): The mathematical query string.

    Returns:
        str: The result of the evaluation or an error message.
    """
    query = query.strip()
    original_query = query # For error messages or context

    try:
        # Phase 1: Direct evaluation for simple expressions or if query is a SymPy object method
        # e.g., "pi.evalf(10)", "sin(x)**2 + cos(x)**2", "simplify(sin(x)**2 + cos(x)**2)"
        # Using parse_expr for more robust parsing than sympify alone for initial string.
        # It handles transformations like "2x" to "2*x".

        # Check for specific commands first using regex to guide parsing
        # This allows for more structured argument handling for complex functions.

        # Command: diff(expression, var1, order1, var2, order2, ...)
        # Corrected regex: First arg is non-comma, rest is everything else for further parsing
        match = re.match(r"diff\s*\(([^,]+)\s*,(.+)\)", query, re.IGNORECASE)
        if match:
            expr_str = match.group(1).strip()
            if not expr_str: raise ValueError("Expression for diff cannot be empty.")
            args_str = match.group(2).strip()

            # DEBUG for Q22
            # if "f(x,y)" in expr_str and "x, y" in args_str :
            #     print(f"[DEBUG-DIFF] expr_str: '{expr_str}', args_str: '{args_str}'")

            expr = parse_expr(expr_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

            # Parse differentiation variables and orders
            # Example: "x" or "x, 2" or "x, y, 2" (for mixed partials)
            # SymPy's diff expects arguments like: diff(expr, symbol1, order1, symbol2, order2, ...)
            parsed_diff_args = []
            raw_diff_args = [s.strip() for s in args_str.split(',')]
            current_arg_idx = 0
            while current_arg_idx < len(raw_diff_args):
                var_name = raw_diff_args[current_arg_idx]
                # Ensure var_name is not empty, could happen with trailing commas
                if not var_name:
                    current_arg_idx +=1
                    continue
                var_sym = SYMPY_GLOBALS.get(var_name, symbols(var_name))
                parsed_diff_args.append(var_sym)
                current_arg_idx += 1
                # Check if next arg is an order for the current var
                if current_arg_idx < len(raw_diff_args) and raw_diff_args[current_arg_idx].isdigit():
                    parsed_diff_args.append(int(raw_diff_args[current_arg_idx]))
                    current_arg_idx += 1
                # else, it's a variable with an implicit order of 1, which is fine for sympy.diff

            result = diff(expr, *parsed_diff_args)
            return pretty(result)

        # Command: integrate(expression, var_or_tuple) or integrate(expression, (var, low, high))
        match = re.match(r"integrate\s*\((.+)\s*,\s*(.+)\)", query, re.IGNORECASE)
        if match:
            expr_str = match.group(1).strip()
            if not expr_str: raise ValueError("Expression for integrate cannot be empty.")
            var_spec_str = match.group(2).strip()
            if not var_spec_str: raise ValueError("Variable specification for integrate cannot be empty.")

            expr = parse_expr(expr_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

            # Parse the variable specification string (e.g., "x" or "(x,0,1)" or "(x,0,oo)")
            try:
                # Let parse_expr handle the structure, including tuples.
                var_spec_parsed = parse_expr(var_spec_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
            except Exception as e_parse_var:
                raise ValueError(f"Could not parse integration variable/bounds: '{var_spec_str}'. Error: {e_parse_var}")

            if isinstance(var_spec_parsed, SympySymbol):
                # Indefinite integral: var_spec_parsed is the symbol
                result = integrate(expr, var_spec_parsed)
            elif isinstance(var_spec_parsed, tuple) and len(var_spec_parsed) == 3:
                # Definite integral: var_spec_parsed is (symbol, low_bound, high_bound)
                # Ensure the first element of the tuple is a symbol
                if not isinstance(var_spec_parsed[0], SympySymbol):
                    raise ValueError(f"First element in integration tuple must be a symbol, got {type(var_spec_parsed[0])} for '{var_spec_str}'")
                # Ensure bounds are valid expressions (already handled by parse_expr for var_spec_parsed)
                result = integrate(expr, var_spec_parsed)
            else:
                # Not a symbol and not a 3-tuple, unsupported format for integration variable/bounds
                raise ValueError(f"Invalid format for integration variable/bounds: '{var_spec_str}'. Expected a symbol (e.g., x) or a tuple (e.g., (x, 0, 1)).")
            return pretty(result)

        # Command: solve(equations, variables)
        # equations can be "expr" (implies expr=0), "Eq(lhs, rhs)", "[Eq(..), Eq(..)]"
        # variables can be "x", "[x,y]"
        match = re.match(r"solve\s*\((.+)\s*,\s*(.+)\)", query, re.IGNORECASE)
        if match:
            eqs_str = match.group(1).strip()
            if not eqs_str: raise ValueError("Equations for solve cannot be empty.")
            vars_str = match.group(2).strip()
            if not vars_str: raise ValueError("Variables for solve cannot be empty.")

            # Parse variables first
            cleaned_vars_str = vars_str.strip()
            if cleaned_vars_str.startswith('[') and cleaned_vars_str.endswith(']'):
                vars_list_str = cleaned_vars_str[1:-1].split(',')
                # Filter out empty strings that might result from "[,]" or trailing commas
                parsed_vars = [SYMPY_GLOBALS.get(v.strip(), symbols(v.strip())) for v in vars_list_str if v.strip()]
                if not parsed_vars and vars_list_str: # Original list was not empty but all items were whitespace
                     raise ValueError("Variables list for solve is empty or contains only whitespace.")
            else:
                if not cleaned_vars_str: raise ValueError("Variable for solve cannot be empty.")
                parsed_vars = [SYMPY_GLOBALS.get(cleaned_vars_str, symbols(cleaned_vars_str))]

            # Parse equations
            # This is tricky because equations can be expressions (assumed =0) or Eq instances
            # If it's a list of equations for a system:
            if eqs_str.startswith('[') and eqs_str.endswith(']'):
                eq_list_str = eqs_str[1:-1]
                # Need a robust way to split equations in a list, respecting commas within Eq, etc.
                # For now, assume simple comma separation of individual Eq strings or expressions
                # Split by comma only if the comma is not inside parentheses.
                # Filter out empty strings that might result from splitting if there are trailing commas.
                individual_eq_strs = [s.strip() for s in re.split(r',(?![^()]*\))', eq_list_str) if s.strip()]

                parsed_eqs = []
                for eq_s in individual_eq_strs:
                    # If user types "x**2 = 4", parse it into an Eq object
                    if '=' in eq_s and not eq_s.lower().startswith("eq("):
                        parts = eq_s.split('=', 1)
                        if len(parts) == 2: # Ensure the split is valid
                            lhs, rhs = parts
                            lhs_strip = lhs.strip()
                            rhs_strip = rhs.strip()
                            if not lhs_strip: raise ValueError("LHS of equation cannot be empty.")
                            if not rhs_strip: raise ValueError("RHS of equation cannot be empty.")
                            lhs_expr = parse_expr(lhs_strip, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
                            rhs_expr = parse_expr(rhs_strip, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
                            parsed_eqs.append(Eq(lhs_expr, rhs_expr))
                        else: # Malformed equation string with '='
                            raise ValueError(f"Malformed equation string with '=': {eq_s}")
                    else: # Assumed to be an expression (e.g., x**2-4 for x**2-4=0) or an explicit Eq(..)
                        eq_s_strip = eq_s.strip()
                        if not eq_s_strip: raise ValueError("Equation string/expression in list cannot be empty.")
                        parsed_eqs.append(parse_expr(eq_s_strip, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS))
            else: # Single equation string
                eqs_str_strip = eqs_str.strip() # Use the already stripped and checked eqs_str
                if not eqs_str_strip: raise ValueError("Equation for solve cannot be empty.") # Should be caught by initial check on eqs_str
                if '=' in eqs_str_strip and not eqs_str_strip.lower().startswith("eq("):
                    parts = eqs_str_strip.split('=', 1)
                    if len(parts) == 2:
                        lhs, rhs = parts
                        lhs_strip = lhs.strip()
                        rhs_strip = rhs.strip()
                        if not lhs_strip: raise ValueError("LHS of equation cannot be empty.")
                        if not rhs_strip: raise ValueError("RHS of equation cannot be empty.")
                        lhs_expr = parse_expr(lhs_strip, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
                        rhs_expr = parse_expr(rhs_strip, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
                        parsed_eqs = Eq(lhs_expr, rhs_expr)
                    else:
                        raise ValueError(f"Malformed equation string with '=': {eqs_str_strip}")
                else:
                     parsed_eqs = parse_expr(eqs_str_strip, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

            result = solve(parsed_eqs, *parsed_vars)
            return pretty(result)

        # Command: limit(expression, var, point, dir) (dir is optional: "+", "-")
        # Corrected regex for point to be less greedy and correctly capture direction
        match = re.match(r"limit\s*\((.+)\s*,\s*(\w+)\s*,\s*([^,]+)(?:\s*,\s*['\"]([+\-])['\"])?\s*\)", query, re.IGNORECASE)
        if match:
            expr_str, var_str, point_str, dir_str = match.groups()
            expr_arg_str = expr_str.strip()
            if not expr_arg_str: raise ValueError("Expression for limit cannot be empty.")
            expr = parse_expr(expr_arg_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

            var_arg_str = var_str.strip() # \w+ ensures var_str is not empty if matched
            var_sym = SYMPY_GLOBALS.get(var_arg_str, symbols(var_arg_str))

            point_expr_str = point_str.strip() # [^,]+ ensures point_str is not empty if matched
            if not point_expr_str: raise ValueError("Limit point cannot be empty.") # Redundant due to [^,]+ but good practice
            point = parse_expr(point_expr_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

            if dir_str: # dir_str will be '+' or '-'
                result = limit(expr, var_sym, point, dir=str(dir_str)) # Explicitly cast to string
            else:
                result = limit(expr, var_sym, point) # Do not pass dir if None
            return pretty(result)

        # Command: series(expression, var, point, n) (point, n optional)
        match = re.match(r"series\s*\((.+?)(?:\s*,\s*(\w+))?(?:\s*,\s*([^,]+))?(?:\s*,\s*(\d+))?\s*\)", query, re.IGNORECASE)
        if match:
            expr_str, var_str, point_str, n_str = match.groups()

            expr_arg_str = expr_str.strip()
            if not expr_arg_str: raise ValueError("Expression for series cannot be empty.")
            expr = parse_expr(expr_arg_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

            args = [expr]
            if var_str:
                var_arg_str = var_str.strip()
                if not var_arg_str: raise ValueError("Variable for series cannot be empty if specified.")
                args.append(SYMPY_GLOBALS.get(var_arg_str, symbols(var_arg_str)))

            if point_str: # point_str could be None if var_str is also None
                point_arg_str = point_str.strip()
                if not point_arg_str: raise ValueError("Point for series cannot be empty if specified.")
                args.append(parse_expr(point_arg_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS))

            if n_str: # n_str is from \d+, so it's digits or None
                args.append(int(n_str.strip())) # .strip() is good practice though \d+ shouldn't have spaces
            result = series(*args)
            return pretty(result)

        # Command: latex(expression)
        match = re.match(r"latex\s*\((.+)\)", query, re.IGNORECASE)
        if match:
            expr_str = match.group(1).strip()
            if not expr_str: raise ValueError("Expression for latex cannot be empty.")
            expr = parse_expr(expr_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
            # For integrals, derivatives etc., apply doit() before latex for evaluated form
            if hasattr(expr, 'doit') and isinstance(expr, (Integral, Derivative, SympyLimit)):
                expr = expr.doit()
            return latex(expr)

        # Command: pretty(expression)
        match = re.match(r"pretty\s*\((.+)\)", query, re.IGNORECASE)
        if match:
            expr_str = match.group(1).strip()
            if not expr_str: raise ValueError("Expression for pretty cannot be empty.")
            expr = parse_expr(expr_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
            if hasattr(expr, 'doit') and isinstance(expr, (Integral, Derivative, SympyLimit)):
                expr = expr.doit()
            return pretty(expr)

        # Command: det(Matrix([...])) or det(matrix_var)
        match = re.match(r"det\s*\((.+)\)", query, re.IGNORECASE)
        if match:
            matrix_str = match.group(1).strip()
            if not matrix_str: raise ValueError("Argument for det cannot be empty.")
            # This assumes matrix_str is directly parsable by parse_expr into a Matrix object
            # e.g., "Matrix([[1,2],[3,4]])"
            matrix_expr = parse_expr(matrix_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
            if not isinstance(matrix_expr, Matrix):
                return "Error: Argument for 'det' must be a Matrix object (e.g., Matrix([[1,2],[3,4]]))."
            result = det(matrix_expr)
            return pretty(result)

        # Command: dsolve(equation, function)
        match = re.match(r"dsolve\s*\((.+)\s*,\s*(.+)\)", query, re.IGNORECASE)
        if match:
            eq_str = match.group(1).strip()
            if not eq_str: raise ValueError("Equation for dsolve cannot be empty.")
            func_str = match.group(2).strip()
            if not func_str: raise ValueError("Function for dsolve cannot be empty.")

            func_to_solve = parse_expr(func_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

            eq_strip = eq_str.strip() # Use the stripped version
            if not eq_strip: raise ValueError("Equation for dsolve (after strip) cannot be empty.") # Should be caught by initial check
            if '=' in eq_strip and not eq_strip.lower().startswith("eq("):
                parts = eq_strip.split('=', 1)
                if len(parts) == 2:
                    lhs, rhs = parts
                    lhs_s, rhs_s = lhs.strip(), rhs.strip()
                    if not lhs_s or not rhs_s: raise ValueError("LHS or RHS of dsolve equation is empty.")
                    lhs_expr = parse_expr(lhs_s, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
                    rhs_expr = parse_expr(rhs_s, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
                    eq = Eq(lhs_expr, rhs_expr)
                else:
                    raise ValueError(f"Malformed equation string with '=' for dsolve: {eq_strip}")
            else:
                eq = parse_expr(eq_strip, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

            result = dsolve(eq, func_to_solve)
            return pretty(result)


        # Phase 2: General expression evaluation or specific method calls like .evalf()
        # This handles queries like "sin(pi/4)", "x*y + 2*x", or "my_expr.evalf(10)"
        # If it's a specific method call like .evalf()
        evalf_match = re.match(r"(.+)\s*\.\s*evalf\s*(?:\(\s*(\d*)\s*\))?$", query, re.IGNORECASE)
        if evalf_match:
            expr_str = evalf_match.group(1).strip()
            if not expr_str: raise ValueError("Expression for .evalf() cannot be empty.")
            precision_str = evalf_match.group(2) # precision_str can be None or empty if no precision given

            expr = parse_expr(expr_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
            if hasattr(expr, 'doit'): # Evaluate things like Integral() or Derivative() before evalf
                expr = expr.doit()

            if precision_str and precision_str.strip():
                result = expr.evalf(int(precision_str))
            else:
                result = expr.evalf()
            return str(result) # evalf returns a SymPy Float, convert to string

        # Default: try to parse and evaluate the expression directly
        # This covers arithmetic, function calls like sin(pi/2), variable expressions like x+y
        # Query was already stripped at the beginning. If it's empty now, it means it was only whitespace.
        if not query: raise ValueError("Query cannot be empty or just whitespace.")
        expr = parse_expr(query, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)

        # If the expression is a type that should be "done" (like Derivative, Integral objects)
        # before potentially being returned or further processed.
        if hasattr(expr, 'doit') and isinstance(expr, (Integral, Derivative, SympyLimit)):
            evaluated_expr = expr.doit()
            return pretty(evaluated_expr)

        # Handle matrices specifically before number checks
        if isinstance(expr, Matrix):
            return pretty(expr)

        # Numeric results formatting
        if expr.is_Float: # Already a float, possibly from an evalf() call earlier
            return str(expr)
        if expr.is_Integer or expr.is_Rational: # Keep exact form for these
            return pretty(expr) # Or str(expr) if pretty is too verbose for simple numbers
        if expr.is_number: # For other types of numbers (e.g. ComplexFloat)
            return str(expr.evalf()) # Default to evalf for other numeric types

        # Fallback for other symbolic expressions
        return pretty(expr)

    except (SyntaxError, TypeError, AttributeError, ValueError, NotImplementedError, Exception) as e:
        error_name = type(e).__name__
        error_msg = str(e)
        # Provide more specific feedback for common parsing issues
        if isinstance(e, SyntaxError) and "invalid syntax" in error_msg.lower() and ("<string>" in error_msg or "EOL" in error_msg):
             return f"Error: Invalid syntax in query '{original_query}'. Please check your expression."
        if isinstance(e, (TypeError, ValueError)) and "sympy.parsing.sympy_parser" in str(e.__traceback__):
             return f"Error: Could not parse parts of the mathematical expression '{original_query}'. Details: {error_name}({error_msg})"
        if isinstance(e, NameError):
            return f"Error: Undefined symbol or function in query '{original_query}'. Details: {error_msg}"
        return f"Error processing query '{original_query}': {error_name}({error_msg})"

if __name__ == "__main__":
    print("--- SymPy Math Solver Test ---")

    test_queries = [
        # Basic arithmetic & direct evaluation
        "2 + 2",
        "pi",
        "pi.evalf()",
        "pi.evalf(50)",
        "(1+sqrt(5))/2",
        "(1+sqrt(5))/2 .evalf()", # Note the space before .evalf() is handled by strip
        "sin(pi/4)",
        "sin(pi/4).evalf(20)",
        "exp(1)",
        "log(E)", # E is exp(1) in SYMPY_GLOBALS
        "log(10)",
        "sqrt(x**2)", # Should give Abs(x)
        "Abs(x).subs(x, -5)",
        "Rational(1,3) + Rational(1,6)",
        "x + y + z", # Simple symbolic
        "expand((x+y)**3)",
        "factor(x**3 - y**3)",
        "simplify((sin(x)**2 + cos(x)**2))",

        # Calculus - Differentiation
        "diff(x**3 + 2*x, x)",
        "diff(sin(x)*exp(x), x)",
        "diff(x**3 * sin(y), x, 2, y, 1)", # diff(expr, x, 2, y, 1) -> d^3f / (dx^2 dy)
        "diff(f(x,y), x, y)", # Using predefined Function f

        # Calculus - Integration
        "integrate(sin(x)*cos(x), x)",
        "integrate(1/x, x)",
        "integrate(exp(-x), (x, 0, oo))", # Definite integral with SymPy's oo
        "integrate(x**2, (x, 0, 1))",
        "integrate(sin(x**2), x)", # Non-elementary integral

        # Calculus - Limits
        "limit(sin(x)/x, x, 0)",
        "limit((1+1/x)**x, x, oo)",
        "limit(1/x, x, 0, '+')", # One-sided limit
        "limit(1/x, x, 0, '-')",

        # Calculus - Series Expansion
        "series(cos(x), x, 0, 5)", # Series of cos(x) around x=0 up to O(x^5)
        "series(exp(x), x, 1, 3)",

        # Algebra - Solving Equations
        "solve(x**2 - 9, x)",
        "solve(Eq(x**2, 16), x)", # Using Eq explicitly
        "solve(x**2 = 25, x)",   # Using '=' directly in solve
        "solve(a*x**2 + b*x + c, x)", # Symbolic solve
        "solve([x + y - 5, x - y - 1], [x, y])", # System of linear equations using implicit Eq
        "solve([Eq(x + y, 5), Eq(x - y, 1)], [x, y])", # System using explicit Eq
        "solve(x**3 - 6*x**2 + 11*x - 6, x)", # Cubic equation

        # Differential Equations
        "dsolve(f(x).diff(x) - 2*f(x), f(x))",
        "dsolve(Eq(f(x).diff(x,x) + f(x), 0), f(x))", # y'' + y = 0

        # Matrix Operations
        "Matrix([[1,2],[3,4]])", # Direct matrix creation
        "det(Matrix([[1,2],[3,4]]))",
        "det(Matrix([[a,b],[c,d]]))",

        # Output Formatting
        "latex(integrate(1/x, x))",
        "pretty(integrate(1/x, x))",
        "latex(Matrix([[1,x],[y,2]]))",

        # Error Handling Test Cases
        "integrate(sin(x) بدون معنى, x)", # Invalid syntax within sympy parse
        "solve(x+*y=5,x)", # Syntax error before specific command parsing
        "diff(x, y, z)", # Mismatched args for diff if z is not int
        "unknown_function(x)",
        "1/0", # Should be caught by SymPy as zoo or error
        "limit(1/x, x, non_symbol)", # Error in point for limit
        "solve(x=5, non_existent_var)",
        "det(x+y)", # det on non-matrix
        "evalf(sin(x))" # evalf as a command (current regex expects .evalf())
    ]

    for i, query in enumerate(test_queries):
        print(f"Query {i+1}: \"{query}\"")
        result = solve_math_query(query)
        print(f"Result:\n{result}")
        print("-" * 40)
