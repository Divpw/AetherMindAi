import re
from sympy import (
    sympify, diff, integrate, solve, limit, series,
    sin, cos, tan, asin, acos, atan,
    pi, exp, log, sqrt, Abs, I, oo, zoo, S,
    symbols, Symbol, Function, Eq, Rational, Float, Integer,
    Derivative, Integral, Limit as SympyLimit,
    Matrix, det,
    latex, pretty,
    factorial, expand, factor, simplify,
    dsolve
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    function_exponentiation
)
from sympy.core.sympify import SympifyError

# Define common symbols that users might use without explicitly defining them.
# Using S.Infinity for oo, S.ComplexInfinity for zoo, S.ImaginaryUnit for I
x, y, z, t = symbols('x y z t')
a, b, c, d = symbols('a b c d')
theta, phi = symbols('theta phi')
f, g, h = symbols('f g h', cls=Function) # Pre-define f, g, h as functions

# Global context for parse_expr, making common SymPy functions and constants available.
SYMPY_GLOBALS = {
    # Symbols
    'x': x, 'y': y, 'z': z, 't': t,
    'a': a, 'b': b, 'c': c, 'd': d,
    'theta': theta, 'phi': phi,
    # Functions - Mathematical
    'sin': sin, 'cos': cos, 'tan': tan,
    'asin': asin, 'acos': acos, 'atan': atan,
    'exp': exp, 'ln': log, 'log': log, 'sqrt': sqrt, # Added ln as alias for log
    'abs': Abs, 'Abs': Abs, 'factorial': factorial, # Added abs as alias for Abs
    # Functions - SymPy specific operations
    'integrate': integrate, 'Integral': Integral,
    'diff': diff, 'Derivative': Derivative,
    'solve': solve, 'Eq': Eq,
    'limit': limit, 'Limit': SympyLimit,
    'series': series,
    'expand': expand, 'factor': factor, 'simplify': simplify,
    'Matrix': Matrix, 'det': det,
    'dsolve': dsolve,
    # Constants
    'pi': pi, 'E': exp(1), 'I': S.ImaginaryUnit,
    'oo': S.Infinity, 'zoo': S.ComplexInfinity,
    # Classes/Types (less likely to be typed directly but good for completeness)
    'Symbol': Symbol, 'Function': Function,
    'Rational': Rational, 'Float': Float, 'Integer': Integer,
    # Pre-defined functions
    'f': f, 'g': g, 'h': h,
}

# Transformations for parse_expr
PARSE_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, function_exponentiation)

print("colab_math_solver.py initialized with enhanced SymPy context.")

def _parse_arguments_str(args_str: str) -> list:
    """
    Parses a comma-separated string of arguments, respecting parentheses and tuples.
    Example: "x, (y, 0, 1), z" -> ["x", "(y, 0, 1)", "z"]
    """
    args = []
    current_arg = ""
    paren_level = 0
    for char in args_str:
        if char == ',' and paren_level == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char
            if char == '(':
                paren_level += 1
            elif char == ')':
                paren_level -= 1
    args.append(current_arg.strip())
    return [arg for arg in args if arg] # Filter out empty strings

def _handle_empty_arg(arg_str: str, arg_name: str, command: str):
    if not arg_str:
        raise ValueError(f"{arg_name} for {command} cannot be empty.")
    return arg_str

def _parse_expr_wrapper(expr_str: str, context_msg: str = "") -> any:
    """Wrapper for parse_expr to provide better error messages."""
    try:
        if not expr_str:
            raise ValueError(f"Expression string is empty {context_msg}.")
        return parse_expr(expr_str, local_dict=SYMPY_GLOBALS, transformations=PARSE_TRANSFORMATIONS)
    except SympifyError as e:
        raise SympifyError(f"Failed to parse expression '{expr_str}' {context_msg}. Details: {e}", e)
    except Exception as e: # Catch other potential errors during parsing
        raise ValueError(f"Unexpected error parsing expression '{expr_str}' {context_msg}. Details: {type(e).__name__}({e})")


def solve_math_query(query: str) -> str:
    """
    Parses and evaluates a mathematical query string using SymPy.

    Handles various commands like diff, integrate, solve, limit, series, etc.,
    with improved argument parsing and error handling.
    Uses pretty() for output where possible.
    """
    query = query.strip()
    original_query = query

    try:
        # Regex for commands: command_name whitespace? ( actual_args )
        # We capture command_name and actual_args.
        # actual_args will be further processed by _parse_arguments_str
        command_match = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", query, re.IGNORECASE)

        if command_match:
            command = command_match.group(1).lower()
            args_str = command_match.group(2).strip()

            # Split arguments string by top-level commas
            raw_args = _parse_arguments_str(args_str)

            if command == "diff":
                if len(raw_args) < 2: raise ValueError("diff requires at least 2 arguments: expression and variable(s).")
                expr_str = _handle_empty_arg(raw_args[0], "Expression", "diff")
                expr = _parse_expr_wrapper(expr_str, "for diff expression")

                diff_vars_orders = []
                for i in range(1, len(raw_args)):
                    arg_component = raw_args[i]
                    # Check if it's a symbol followed by an optional integer order
                    parts = [p.strip() for p in arg_component.split(',')] # e.g. "x,2" or just "x"
                    var_sym_str = _handle_empty_arg(parts[0], "Variable", "diff")
                    var_sym = _parse_expr_wrapper(var_sym_str, f"for diff variable '{var_sym_str}'")
                    if not isinstance(var_sym, Symbol):
                        raise ValueError(f"Expected a symbol for differentiation, got '{var_sym_str}'.")
                    diff_vars_orders.append(var_sym)
                    if len(parts) > 1:
                        order_str = _handle_empty_arg(parts[1], "Order", "diff")
                        try:
                            order = int(order_str)
                            if order < 0: raise ValueError("Order for diff must be non-negative.")
                            diff_vars_orders.append(order)
                        except ValueError:
                            raise ValueError(f"Invalid order '{order_str}' for diff, must be an integer.")
                result = diff(expr, *diff_vars_orders)
                return pretty(result)

            elif command == "integrate":
                if len(raw_args) < 2: raise ValueError("integrate requires at least 2 arguments: expression and variable/tuple.")
                expr_str = _handle_empty_arg(raw_args[0], "Expression", "integrate")
                expr = _parse_expr_wrapper(expr_str, "for integrate expression")

                integration_var_specs = []
                for i in range(1, len(raw_args)):
                    var_spec_str = _handle_empty_arg(raw_args[i], "Variable/Tuple", "integrate")
                    var_spec_parsed = _parse_expr_wrapper(var_spec_str, f"for integrate variable/bounds '{var_spec_str}'")

                    if isinstance(var_spec_parsed, Symbol): # Indefinite integral: integrate(expr, x)
                        integration_var_specs.append(var_spec_parsed)
                    elif isinstance(var_spec_parsed, tuple) and len(var_spec_parsed) == 3: # Definite: (x, low, high)
                        if not isinstance(var_spec_parsed[0], Symbol):
                            raise ValueError(f"First element in integration tuple must be a symbol, got {type(var_spec_parsed[0])} in '{var_spec_str}'.")
                        integration_var_specs.append(var_spec_parsed)
                    elif isinstance(var_spec_parsed, tuple) and len(var_spec_parsed) == 1 and isinstance(var_spec_parsed[0], Symbol): # integrate(expr, (x))
                         integration_var_specs.append(var_spec_parsed[0])
                    else:
                        raise ValueError(f"Invalid format for integration variable/bounds: '{var_spec_str}'. Expected symbol or (symbol, low, high) or (symbol).")

                result = integrate(expr, *integration_var_specs)
                return pretty(result)

            elif command == "solve":
                if len(raw_args) < 1: raise ValueError("solve requires at least 1 argument: equation(s) [and optionally variables].")
                eqs_str_arg = _handle_empty_arg(raw_args[0], "Equation(s)", "solve")

                # Determine if it's a list of equations or a single one
                parsed_eqs_arg = _parse_expr_wrapper(eqs_str_arg, "for solve equations")

                equations_to_solve = []
                if isinstance(parsed_eqs_arg, list) or isinstance(parsed_eqs_arg, tuple):
                    for item in parsed_eqs_arg:
                        if isinstance(item, Eq): equations_to_solve.append(item)
                        elif hasattr(item, 'is_Equality') and item.is_Equality: equations_to_solve.append(item) # For Eq objects parsed as such
                        else: equations_to_solve.append(Eq(item, 0)) # Assume expr = 0
                elif isinstance(parsed_eqs_arg, Eq) or (hasattr(parsed_eqs_arg, 'is_Equality') and parsed_eqs_arg.is_Equality):
                    equations_to_solve.append(parsed_eqs_arg)
                else: # Single expression, assumed to be expr = 0
                    equations_to_solve.append(Eq(parsed_eqs_arg, 0))

                vars_to_solve_for = []
                if len(raw_args) > 1:
                    vars_str_arg = _handle_empty_arg(raw_args[1], "Variables", "solve")
                    parsed_vars_arg = _parse_expr_wrapper(vars_str_arg, "for solve variables")
                    if isinstance(parsed_vars_arg, list) or isinstance(parsed_vars_arg, tuple):
                        for v in parsed_vars_arg:
                            if not isinstance(v, Symbol): raise ValueError(f"Expected symbol in variable list for solve, got {type(v)}: '{v}'.")
                            vars_to_solve_for.append(v)
                    elif isinstance(parsed_vars_arg, Symbol):
                        vars_to_solve_for.append(parsed_vars_arg)
                    else:
                        raise ValueError(f"Variables for solve must be a symbol or a list/tuple of symbols. Got: '{vars_str_arg}'")

                result = solve(equations_to_solve, *vars_to_solve_for)
                return pretty(result)

            elif command == "limit":
                if len(raw_args) < 3: raise ValueError("limit requires 3 arguments: expression, variable, point [and optionally direction].")
                expr_str = _handle_empty_arg(raw_args[0], "Expression", "limit")
                expr = _parse_expr_wrapper(expr_str, "for limit expression")

                var_str = _handle_empty_arg(raw_args[1], "Variable", "limit")
                var_sym = _parse_expr_wrapper(var_str, f"for limit variable '{var_str}'")
                if not isinstance(var_sym, Symbol): raise ValueError(f"Limit variable must be a symbol, got '{var_str}'.")

                point_str = _handle_empty_arg(raw_args[2], "Point", "limit")
                point = _parse_expr_wrapper(point_str, f"for limit point '{point_str}'")

                direction = '+' # Default direction
                if len(raw_args) > 3:
                    dir_str_raw = raw_args[3].strip()
                    # Remove quotes if present, e.g. "'+'" or "'-'"
                    dir_str = dir_str_raw.replace("'", "").replace('"', '')
                    if dir_str not in ['+', '-']: raise ValueError(f"Limit direction must be '+' or '-', got '{dir_str_raw}'.")
                    direction = dir_str

                result = limit(expr, var_sym, point, dir=direction)
                return pretty(result)

            elif command == "series":
                if len(raw_args) < 1: raise ValueError("series requires at least an expression argument.")
                expr_str = _handle_empty_arg(raw_args[0], "Expression", "series")
                expr = _parse_expr_wrapper(expr_str, "for series expression")

                series_args = [expr]
                # Optional args: x, x0, n
                if len(raw_args) > 1: # variable x
                    var_str = _handle_empty_arg(raw_args[1], "Variable", "series")
                    var_sym = _parse_expr_wrapper(var_str, f"for series variable '{var_str}'")
                    if not isinstance(var_sym, Symbol): raise ValueError(f"Series variable must be a symbol, got '{var_str}'.")
                    series_args.append(var_sym)
                if len(raw_args) > 2: # point x0
                    point_str = _handle_empty_arg(raw_args[2], "Point (x0)", "series")
                    series_args.append(_parse_expr_wrapper(point_str, f"for series point x0 '{point_str}'"))
                if len(raw_args) > 3: # number of terms n
                    n_str = _handle_empty_arg(raw_args[3], "Number of terms (n)", "series")
                    try:
                        n_val = int(n_str)
                        series_args.append(n_val)
                    except ValueError:
                        raise ValueError(f"Number of terms (n) for series must be an integer, got '{n_str}'.")

                result = series(*series_args)
                return pretty(result)

            elif command in ["latex", "pretty"]:
                if len(raw_args) != 1: raise ValueError(f"{command} requires 1 argument: expression.")
                expr_str = _handle_empty_arg(raw_args[0], "Expression", command)
                expr = _parse_expr_wrapper(expr_str, f"for {command} expression")
                if hasattr(expr, 'doit') and isinstance(expr, (Integral, Derivative, SympyLimit)):
                    expr = expr.doit()
                return latex(expr) if command == "latex" else pretty(expr)

            elif command == "det":
                if len(raw_args) != 1: raise ValueError("det requires 1 argument: matrix.")
                matrix_str = _handle_empty_arg(raw_args[0], "Matrix", "det")
                matrix_expr = _parse_expr_wrapper(matrix_str, "for det matrix")
                if not isinstance(matrix_expr, Matrix):
                    raise ValueError("Argument for 'det' must be a Matrix object (e.g., Matrix([[1,2],[3,4]])).")
                result = det(matrix_expr)
                return pretty(result)

            elif command == "dsolve":
                if len(raw_args) < 2: raise ValueError("dsolve requires at least 2 arguments: equation and function.")
                eq_str = _handle_empty_arg(raw_args[0], "Equation", "dsolve")
                # Equation can be an expression (assumed == 0) or an Eq object
                eq_parsed = _parse_expr_wrapper(eq_str, "for dsolve equation")
                if not isinstance(eq_parsed, Eq) and not (hasattr(eq_parsed, 'is_Equality') and eq_parsed.is_Equality) :
                    # if not an Eq object, assume it's an expression that should be equal to 0
                     eq_to_solve = Eq(eq_parsed, 0)
                else:
                     eq_to_solve = eq_parsed

                func_str = _handle_empty_arg(raw_args[1], "Function", "dsolve")
                func_to_solve = _parse_expr_wrapper(func_str, f"for dsolve function '{func_str}'")
                # Optional hint argument
                # hint_str = raw_args[2] if len(raw_args) > 2 else ""
                # result = dsolve(eq_to_solve, func_to_solve, hint=hint_str if hint_str else 'default')
                result = dsolve(eq_to_solve, func_to_solve)
                return pretty(result)

            # Simple commands that take one expression argument and call a SymPy function
            elif command in ["expand", "factor", "simplify"]:
                if len(raw_args) != 1: raise ValueError(f"{command} requires 1 argument: expression.")
                expr_str = _handle_empty_arg(raw_args[0], "Expression", command)
                expr = _parse_expr_wrapper(expr_str, f"for {command} expression")
                if command == "expand": result = expand(expr)
                elif command == "factor": result = factor(expr)
                elif command == "simplify": result = simplify(expr)
                return pretty(result)

            else: # Unrecognized command but matched command_match regex
                raise ValueError(f"Unrecognized math command: '{command}'. Supported commands include diff, integrate, solve, limit, series, expand, factor, simplify, det, dsolve, latex, pretty.")


        # Phase 2: General expression evaluation or specific method calls like .evalf()
        # This handles queries like "sin(pi/4)", "x*y + 2*x", or "my_expr.evalf(10)"
        evalf_match = re.match(r"(.+?)\s*\.\s*evalf\s*(?:\(\s*(\d*)\s*\))?$", query, re.IGNORECASE)
        if evalf_match:
            expr_str = _handle_empty_arg(evalf_match.group(1).strip(), "Expression", ".evalf()")
            precision_str = evalf_match.group(2)

            expr = _parse_expr_wrapper(expr_str, "for .evalf() expression")
            if hasattr(expr, 'doit'):
                expr = expr.doit()

            if precision_str and precision_str.strip():
                result = expr.evalf(int(precision_str))
            else:
                result = expr.evalf()
            return str(result)

        # Default: try to parse and evaluate the expression directly if no command structure found
        if not query: raise ValueError("Query cannot be empty or just whitespace.")
        expr = _parse_expr_wrapper(query, "for general expression")

        if hasattr(expr, 'doit') and isinstance(expr, (Integral, Derivative, SympyLimit)):
            evaluated_expr = expr.doit()
            return pretty(evaluated_expr)

        if isinstance(expr, Matrix): return pretty(expr)
        if expr.is_Float: return str(expr)
        if expr.is_Integer or expr.is_Rational: return pretty(expr)
        if expr.is_number: return str(expr.evalf()) # Default to evalf for other numeric types

        return pretty(expr) # Fallback for other symbolic expressions

    except (SyntaxError, TypeError, AttributeError, ValueError, NotImplementedError, SympifyError) as e:
        error_name = type(e).__name__
        error_msg = str(e)
        # Specific feedback for common parsing issues
        if isinstance(e, SympifyError) or ("parse_expr" in error_msg.lower() or "invalid syntax" in error_msg.lower()):
             return f"Error: Could not parse mathematical expression '{original_query}'. Please check syntax. Details: {error_name}({error_msg})"
        if isinstance(e, NameError): # Should be caught by parse_expr if symbol not in SYMPY_GLOBALS
            return f"Error: Undefined symbol or function in query '{original_query}'. Details: {error_msg}"
        if isinstance(e, ValueError) and "cannot be empty" in error_msg: # Custom empty arg errors
            return f"Error: {error_msg} in query '{original_query}'."
        # General error message
        return f"Error processing query '{original_query}': {error_name}({error_msg})"
    except Exception as e: # Catch any other unexpected errors
        return f"An unexpected error occurred while processing '{original_query}': {type(e).__name__}({str(e)})"


if __name__ == "__main__":
    print("--- SymPy Math Solver Test (Enhanced) ---")

    test_queries = [
        # Basic & Direct Eval
        "2 + 2*x", "pi.evalf(10)", "sin(pi/4)", "sqrt(x**2)", "Abs(x).subs(x, -5)",
        "expand((x+y)**2)", "factor(x**2 - y**2)", "simplify(sin(x)**2 + cos(x)**2)",
        # Diff
        "diff(x**3 + 2*x, x)", "diff(sin(x)*exp(x), x, x)", "diff(x**2*y**3, x, 2, y, 1)", "diff(f(x,y), x, y)",
        "diff(cos(x), (x))", # Test with tuple for var
        # Integrate
        "integrate(sin(x)*cos(x), x)", "integrate(1/x, (x, 1, exp(1)))", "integrate(exp(-x**2), (x, -oo, oo))",
        "integrate(x*y, x, y)", # Multiple indefinite integrals
        "integrate(x**2, (x, 0, 1), (y, 0, 2))", # Error: integrate with multiple tuples (SymPy needs one var or one tuple)
                                                  # Corrected: SymPy integrate handles one var or one (var, low, high) tuple at a time for its direct args.
                                                  # For multiple integration, it's nested: integrate(integrate(expr, var1_spec), var2_spec)
                                                  # This test will show current limitation if user tries this flat structure.
                                                  # A more robust parser might interpret this as nested. For now, it will likely error on second tuple.
        "integrate(x*y, (x,0,1))", # Indefinite on y, definite on x
        "integrate(1/(x**2+1), x)",
        "integrate(log(x), x)",
        "integrate(exp(-x), (x, 0, oo))",
        # Solve
        "solve(x**2 - 9, x)", "solve(Eq(x**2, 16), x)", "solve(x**2 = 4, x)", # Test implicit Eq
        "solve([x + y - 5, x - y - 1], [x, y])", "solve((x**2-1, x-1), x)",
        "solve(x**3 - 6*x**2 + 11*x - 6 = 0, x)",
        "solve(Eq(f(x), x**2), f(x))", # Solving for a function (general solution)
        # Limit
        "limit(sin(x)/x, x, 0)", "limit((1+1/x)**x, x, oo)", "limit(1/x, x, 0, '+')", "limit(1/x, x, 0, '-')",
        "limit( (x**2-1)/(x-1), x, 1)",
        # Series
        "series(cos(x), x, 0, 5)", "series(exp(x), x, 1, 3)", "series(log(1+x), x, 0, 4)",
        "series(1/(1-x))", # Default around x=0, default terms
        # Dsolve
        "dsolve(f(x).diff(x) - 2*f(x), f(x))", "dsolve(Eq(f(x).diff(x,x) + f(x), 0), f(x))",
        "dsolve(y*f(x).diff(x) + x*f(x) - x, f(x))", # More complex ODE
        # Matrix
        "Matrix([[1,2],[3,4]])", "det(Matrix([[1,x],[y,2]]))",
        # Output Formatting
        "latex(integrate(1/x,x))", "pretty(series(sin(x),x,0,4))",
        # Nested / Complex Tuple Inputs
        "integrate(x*sin(y), (x,0,1), (y,0,pi))", # This should be parsed as integrate(expr, (x,0,1), (y,0,pi))
                                                   # and then SymPy's integrate will take expr, then (x,0,1), then (y,0,pi)
                                                   # This is how SymPy's own integrate takes multiple limit tuples.
        "integrate(x*y*z, (x,0,1), (y,0,1), (z,0,1))",
        # Error Handling Cases
        "diff(x**2, x, y, 2, z)", # Valid if x,y,z are symbols
        "integrate(nonexistent_func(x), x)", # NameError
        "solve(x+*y=5,x)", # SyntaxError in expression
        "limit(1/x, x, non_symbol)", # SympifyError for point
        "series(sin(x), x, 0, non_integer_n)", # ValueError for n
        "det(x+y)", # Error: det on non-matrix
        "unsupported_command(x)", # ValueError for command
        "solve(x=2, )", # Error: empty variable for solve
        "integrate(x, )", # Error: empty integration var
        "diff( , x)", # Error: empty expression for diff
        "limit(sin(x)/x, x, 0, 'both')", # Error: invalid direction for limit
        "integrate(x, (x, 0))", # Error: incomplete tuple for integration
        "integrate(x, (x, 0, 1, 2))", # Error: tuple too long for integration
        "solve([x+y=5, x-y=1], x, y)" # Variables should be a list/tuple [x,y] or single var
    ]

    for i, t_query in enumerate(test_queries):
        print(f"\nQuery {i+1}: \"{t_query}\"")
        result = solve_math_query(t_query)
        # For latex, print raw string to see it better
        if t_query.startswith("latex("):
            print(f"Raw LaTeX Result: {result}")
        else:
            print(f"Result:\n{result}")
        print("-" * 40)

    print("\n--- Specific Error Case Example ---")
    err_query = "solve(x**2 = 4, [x, nonexistent_var])"
    print(f"Query: \"{err_query}\"")
    result = solve_math_query(err_query)
    print(f"Result:\n{result}")
    print("-" * 40)

    err_query_2 = "integrate(1/x, (x, 0, oo, oo))" # Tuple too long
    print(f"Query: \"{err_query_2}\"")
    result = solve_math_query(err_query_2)
    print(f"Result:\n{result}")
    print("-" * 40)
