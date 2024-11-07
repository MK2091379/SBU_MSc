class Polynomial:
    def __init__(self, coefficients):
        """
        Initializes the polynomial with a list of coefficients.
        The coefficients are in order from the highest degree to the constant term.
        Example: [3, 2, 1] represents 3x^2 + 2x + 1.
        """
        self.coefficients = coefficients

    def evaluate(self, x):
        """
        Evaluates the polynomial for a given value of x.
        """
        result = 0
        power = len(self.coefficients) - 1
        for coeff in self.coefficients:
            result += coeff * (x ** power)
            power -= 1
        return result

    def add(self, other):
        """
        Adds two polynomials and returns a new Polynomial object.
        """
        
        len_self = len(self.coefficients)
        len_other = len(other.coefficients)
        
        if len_self < len_other:
            self.coefficients = [0] * (len_other - len_self) + self.coefficients
        elif len_self > len_other:
            other.coefficients = [0] * (len_self - len_other) + other.coefficients
        
        new_coefficients = [a + b for a, b in zip(self.coefficients, other.coefficients)]
        return Polynomial(new_coefficients)

    def subtract(self, other):
        """
        Subtracts another polynomial from the current polynomial and returns a new Polynomial object.
        """

        len_self = len(self.coefficients)
        len_other = len(other.coefficients)
        
        if len_self < len_other:
            self.coefficients = [0] * (len_other - len_self) + self.coefficients
        elif len_self > len_other:
            other.coefficients = [0] * (len_self - len_other) + other.coefficients
        
        new_coefficients = [a - b for a, b in zip(self.coefficients, other.coefficients)]
        return Polynomial(new_coefficients)

    def __str__(self):
        """
        String representation of the polynomial.
        """
        result = ""
        power = len(self.coefficients) - 1
        for coeff in self.coefficients:
            if coeff != 0:
                if power > 0:
                    result += f"{coeff}x^{power} + "
                else:
                    result += f"{coeff} "
            power -= 1
        return result.strip().rstrip("+").strip()
