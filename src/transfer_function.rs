use super::iir::*;
use crate::image::*;
use crate::vector::*;

#[derive(Debug, Clone)]
pub struct TransferFunction {
    pub numerator: Polynomial,
    pub denominator: Polynomial,
}

impl TransferFunction {
    pub fn new(numerator: Polynomial, denominator: Polynomial) -> Self {
        TransferFunction { numerator, denominator }
    }
    // to push deeper with bilinear transform? lol.
    // pub fn z_to_s(&self) -> TransferFunction {

    // }
    // pub fn s_to_z(&self) -> TransferFunction {

    // }
    pub fn fb_comb(a: f32, d: usize) -> Self {
        let mut den_vec = vec![0.0; d];
        den_vec[0] = 1.0;
        den_vec[d-1] = -a;
        Self::new(Polynomial::from(vec![1.0]), Polynomial::from(den_vec))
    }
    pub fn ident() -> Self {
        Self::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0]))
    }
    // given H(z) and n, the number of samples to compute, returns the impulse response of the IIR
    pub fn impulse_response(&self, n: usize) -> Vec<Vec2> {
        let mut running = IIR::from(self.clone());
        let mut impulse_response = Vec::with_capacity(n);

        impulse_response.push(running.tick(vec2(1.0, 0.0)));
        for _ in 0..n-1 {
            impulse_response.push(running.tick(vec2(0.0, 0.0)));
        }

        impulse_response
    }
    pub fn plot_impulse_response(&self, outfile: &str, n: usize, xres: usize, yres: usize) {
        let ir = self.impulse_response(n);
        let mut buf = ImageBuffer::new(xres, yres);
        buf.fill(vec4(1.0, 1.0, 1.0, 1.0));
        buf.plot(&ir.iter().map(|z| z.x).collect::<Vec<f32>>(), vec4(0.5, 0.5, 1.0, 1.0), 1);
        buf.dump_to_file(&format!("{}.png", outfile));
    }
    // Returns a vector of pole locations and of zero locations
    pub fn pole_zero(&self) -> (Vec<Vec2>, Vec<Vec2>) {
        (self.numerator.roots(), self.denominator.roots())
    }
    pub fn plot_pole_zero(&self, outfile: &str, xres: usize, yres: usize) {
        // plot axis
        // plot red dots for poles and blue dots for zeros
        // ok this nearly works but i should use a transformation matrix for the plot orientating
        // our input points are in one space and this plotting is expecting 0..1 ... so we need to flip y axis, include negative quadrants, and scale
        // todo a test for u and v transformations. maybe inverse will be correct. also print out poles and zeros
        let (zeros, poles) = self.pole_zero();
        let mut buf = ImageBuffer::new(xres, yres);
        buf.fill(vec4(1.0, 1.0, 1.0, 1.0));
        let tu = vec3(0.4, 0.0, 0.5);
        let tv = vec3(0.0, -0.4, 0.5);
        let tw = vec3(0.0, 0.0, 1.0);

        buf.line_absolute(vec2(0.5, 0.0), vec2(0.5, 1.0), vec4(0.5, 0.5, 0.5, 1.0), 1);
        buf.line_absolute(vec2(0.0, 0.5), vec2(1.0, 0.5), vec4(0.5, 0.5, 0.5, 1.0), 1);

        buf.circle_absolute_transform(vec2(0.0, 0.0), tu, tv, tw, 1.0, 2, vec4(0.5, 0.5, 0.5, 1.0));
        for pole in poles {
            buf.set_square_absolute_transform(pole, tu, tv, tw, 3, vec4(1.0, 0.0, 0.0, 1.0));
        }
        for zero in zeros {
            buf.set_square_absolute_transform(zero, tu, tv, tw, 3, vec4(0.0, 0.0, 1.0, 1.0));
        }
        buf.dump_to_file(&format!("{}.png", outfile));
    }
}

impl std::ops::Add for TransferFunction {
    type Output = TransferFunction;

    fn add(self, other: TransferFunction) -> TransferFunction {
        let new_numerator = self.numerator.clone() * other.denominator.clone() + other.numerator.clone() * self.denominator.clone();
        let new_denominator = self.denominator.clone() * other.denominator.clone();

        TransferFunction {
            numerator: new_numerator,
            denominator: new_denominator,
        }
    }
}

impl std::ops::AddAssign for TransferFunction {
    fn add_assign(&mut self, other: TransferFunction) {
        self.numerator = self.numerator.clone() * other.denominator.clone() + other.numerator.clone() * self.denominator.clone();
        self.denominator = self.denominator.clone() * other.denominator.clone();
    }
}

impl std::ops::Mul for TransferFunction {
    type Output = TransferFunction;

    fn mul(self, other: TransferFunction) -> TransferFunction {
        let new_numerator = self.numerator.clone() * other.numerator.clone();
        let new_denominator = self.denominator.clone() * other.denominator.clone();

        TransferFunction {
            numerator: new_numerator,
            denominator: new_denominator,
        }
    }
}

impl std::ops::MulAssign for TransferFunction {
    fn mul_assign(&mut self, other: TransferFunction) {
        self.numerator = self.numerator.clone() * other.numerator.clone();
        self.denominator = self.denominator.clone() * other.denominator.clone();
    }
}

#[derive(Clone, PartialEq)]
pub struct Polynomial {
    pub coefficients: Vec<Vec2>,
}
impl Polynomial {
    // samples in a unit square radius
    pub fn roots(&self) -> Vec<Vec2> {
        let mut result = Vec::new();

        const X_SAMPLES: usize = 40;
        const Y_SAMPLES: usize = 40;
        let orig = vec2(-1.0, -1.0);
        let ivec = vec2(2.0 / X_SAMPLES as f32, 0.0);
        let jvec = vec2(0.0, 2.0 / Y_SAMPLES as f32);
        for i in 0..X_SAMPLES {
            for j in 0..Y_SAMPLES {
                let p = orig + i as f32 * ivec + j as f32 * jvec;
                if let Some(root) = self.newton_root(p) {
                    result.push(root);
                }
            }
        }

        // Deduplicate roots that are close together
        let mut deduplicated_result: Vec<Vec2> = Vec::new();
        for &root in &result {
            let mut is_duplicate = false;
            for &existing_root in &deduplicated_result {
                if (root.x - existing_root.x).abs() < 1E-8 && (root.y - existing_root.y).abs() < 1E-8 {
                    is_duplicate = true;
                    break;
                }
            }
            if !is_duplicate {
                deduplicated_result.push(root);
            }
        }

        deduplicated_result
    }
    pub fn newton_root(&self, initial_guess: Vec2) -> Option<Vec2> {
        const MAX_ITERATIONS: usize = 100;
        const TOLERANCE: f32 = 1e-10;

        let mut guess = initial_guess;
        for _ in 0..MAX_ITERATIONS {
            let f_val = self.evaluate_inv(guess);
            let f_prime_val = self.diff_inv().evaluate_inv(guess);

            let step = f_val / f_prime_val;

            guess = guess - step;

            if step.mag2() < TOLERANCE {
                return Some(guess);
            }
        }

        None // If convergence didn't happen within the allowed iterations
    }

    // evalutate and diff assume x0 x1 x2 x3 etc
    // but its acutally x0 x-1 x-2 x-3...
    // so i think thats why im seeing 'reciprocal' behaviour
    pub fn evaluate_inv(&self, x: Vec2) -> Vec2 {
        let mut result = Vec2::new(0.0, 0.0);
        let mut x_power = Vec2::new(1.0, 0.0); // x^0

        for coeff in &self.coefficients {
            result = result + (*coeff) * x_power;
            x_power = x_power / x;
        }

        result
    }
    pub fn evaluate(&self, x: Vec2) -> Vec2 {
        let mut result = Vec2::new(0.0, 0.0);
        let mut x_power = Vec2::new(1.0, 0.0); // x^0

        for coeff in &self.coefficients {
            result = result + (*coeff) * x_power;
            x_power = x_power * x;
        }

        result
    }
    pub fn diff_inv(&self) -> Polynomial {
        let mut res = Polynomial { coefficients: vec![vec2(0.0, 0.0); self.coefficients.len() - 1] };
        for i in 0..res.coefficients.len() {
            res.coefficients[i] = vec2(-(i as f32 + 1.0), 0.0) * self.coefficients[i+1];
        }
        res
    }
    pub fn diff(&self) -> Polynomial {
        let mut res = Polynomial { coefficients: vec![vec2(0.0, 0.0); self.coefficients.len() - 1] };
        for i in 0..res.coefficients.len() {
            res.coefficients[i] = vec2(i as f32 + 1.0, 0.0) * self.coefficients[i+1];
        }
        res
    }
}

// differentiating polynomial: what would you do, drop first term and multiply each thing by its position

impl std::fmt::Debug for Polynomial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut iter = self.coefficients.iter().peekable();
        while let Some(&coeff) = iter.next() {
            let mut count = 1;
            while let Some(&next_coeff) = iter.peek() {
                if *next_coeff == coeff {
                    count += 1;
                    iter.next();
                } else {
                    break;
                }
            }

            if count > 1 {
                write!(f, "...{} (x{})...", coeff, count)?;
            } else {
                write!(f, "{}", coeff)?;
            }

            if iter.peek().is_some() {
                write!(f, ", ")?;
            }
        }
        Ok(())
    }
}

impl std::ops::Add for Polynomial {
    type Output = Polynomial;

    fn add(self, other: Polynomial) -> Polynomial {
        let mut result_coefficients = vec![vec2(0.0, 0.0); std::cmp::max(self.coefficients.len(), other.coefficients.len())];

        for (i, &coeff) in self.coefficients.iter().enumerate() {
            result_coefficients[i] += coeff;
        }

        for (i, &coeff) in other.coefficients.iter().enumerate() {
            result_coefficients[i] += coeff;
        }

        Polynomial {
            coefficients: result_coefficients,
        }
    }
}

impl std::ops::AddAssign for Polynomial {
    fn add_assign(&mut self, other: Polynomial) {
        let max_len = std::cmp::max(self.coefficients.len(), other.coefficients.len());

        self.coefficients.resize(max_len, vec2(0.0, 0.0));

        for (i, &coeff) in other.coefficients.iter().enumerate() {
            self.coefficients[i] += coeff;
        }
    }
}

impl std::ops::Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: Polynomial) -> Polynomial {
        let mut result_coefficients = vec![vec2(0.0, 0.0); self.coefficients.len() + other.coefficients.len() - 1];

        for (i, &coeff1) in self.coefficients.iter().enumerate() {
            for (j, &coeff2) in other.coefficients.iter().enumerate() {
                result_coefficients[i + j] += coeff1 * coeff2;
            }
        }

        Polynomial {
            coefficients: result_coefficients,
        }
    }
}

impl std::ops::MulAssign for Polynomial {
    fn mul_assign(&mut self, other: Polynomial) {
        let mut result_coefficients = vec![vec2(0.0, 0.0); self.coefficients.len() + other.coefficients.len() - 1];

        for (i, &coeff1) in self.coefficients.iter().enumerate() {
            for (j, &coeff2) in other.coefficients.iter().enumerate() {
                result_coefficients[i + j] += coeff1 * coeff2;
            }
        }

        self.coefficients = result_coefficients;
    }
}

impl From<Vec<f32>> for Polynomial {
    fn from(coefficients: Vec<f32>) -> Self {
        Polynomial { coefficients: coefficients.iter().map(|x| vec2(*x, 0.0)).collect() }
    }
}

impl From<Vec<Vec2>> for Polynomial {
    fn from(coefficients: Vec<Vec2>) -> Self {
        Polynomial { coefficients }
    }
}

#[cfg(test)]
mod test_polynomial {
    use super::*;

    #[test]
    fn test_add() {
        let poly1 = Polynomial::from(vec![1.0, 2.0, 3.0]);
        let poly2 = Polynomial::from(vec![4.0, 5.0, 6.0, 7.0]);

        let result = poly1.clone() + poly2.clone();

        assert_eq!(result, Polynomial::from(vec![5.0, 7.0, 9.0, 7.0]));
    }

    #[test]
    fn test_add_assign() {
        let mut poly1 = Polynomial::from(vec![1.0, 2.0, 3.0]);
        let poly2 = Polynomial::from(vec![4.0, 5.0, 6.0, 7.0]);

        poly1 += poly2.clone();

        assert_eq!(poly1, Polynomial::from(vec![5.0, 7.0, 9.0, 7.0]));
    }

    #[test]
    fn test_mul() {
        let poly1 = Polynomial::from(vec![1.0, 2.0]);
        let poly2 = Polynomial::from(vec![3.0, 4.0, 5.0]);

        let result = poly1.clone() * poly2.clone();

        assert_eq!(result, Polynomial::from(vec![3.0, 10.0, 13.0, 10.0]));
    }

    #[test]
    fn test_mul_assign() {
        let mut poly1 = Polynomial::from(vec![1.0, 2.0]);
        let poly2 = Polynomial::from(vec![3.0, 4.0, 5.0]);

        poly1 *= poly2.clone();

        assert_eq!(poly1, Polynomial::from(vec![3.0, 10.0, 13.0, 10.0]));
    }
}

#[cfg(test)]
mod test_transfer_function {
    use super::*;

    #[test]
    fn test_transfer_function() {
        let p = TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0]));
        let q = TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0]));
        let r = p + q;
        assert_eq!(r.numerator, Polynomial::from(vec![2.0]));
        assert_eq!(r.denominator, Polynomial::from(vec![1.0]));

        let p = TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0]));
        let q = TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0]));
        let r = p * q;
        assert_eq!(r.numerator, Polynomial::from(vec![1.0]));
        assert_eq!(r.denominator, Polynomial::from(vec![1.0]));

        let p = TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0]));
        let q = TransferFunction::new(Polynomial::from(vec![1.0, 1.0]), Polynomial::from(vec![1.0]));
        let r = p + q;
        assert_eq!(r.numerator, Polynomial::from(vec![2.0, 1.0]));
        assert_eq!(r.denominator, Polynomial::from(vec![1.0]));

        let p = TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0]));
        let q = TransferFunction::new(Polynomial::from(vec![1.0, 1.0]), Polynomial::from(vec![1.0]));
        let r = p * q;
        assert_eq!(r.numerator, Polynomial::from(vec![1.0, 1.0]));
        assert_eq!(r.denominator, Polynomial::from(vec![1.0]));

        let p = TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0, 1.0]));
        let q = TransferFunction::new(Polynomial::from(vec![1.0, 1.0]), Polynomial::from(vec![1.0]));
        let r = p + q;
        assert_eq!(r.numerator, Polynomial::from(vec![2.0, 2.0, 1.0]));
        assert_eq!(r.denominator, Polynomial::from(vec![1.0, 1.0]));

        let p = TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0, 1.0]));
        let q = TransferFunction::new(Polynomial::from(vec![1.0, 1.0]), Polynomial::from(vec![1.0]));
        let r = p * q;
        assert_eq!(r.numerator, Polynomial::from(vec![1.0, 1.0]));
        assert_eq!(r.denominator, Polynomial::from(vec![1.0, 1.0]));
    }
}

#[test]
pub fn test_root() {
    let p = Polynomial::from(vec![-1.0, 0.0, 1.0]);
    dbg!(p.roots());
    let q = Polynomial::from(vec![1.0, 0.0, 1.0]);
    dbg!(q.roots());
    let p = Polynomial::from(vec![-4.0, 0.0, 1.0]);
    dbg!(p.roots());
    let p = Polynomial::from(vec![4.0, 0.0, 1.0]);
    dbg!(p.roots());
}

#[test]
pub fn test_polezero() {
    TransferFunction::new(Polynomial::from(vec![1.0, 0.1]), Polynomial::from(vec![1.0, -0.1])).plot_pole_zero("pz1", 500, 500);
    TransferFunction::fb_comb(0.5, 10).plot_pole_zero("pz2", 500, 500);
    TransferFunction::fb_comb(0.7, 10).plot_pole_zero("pz3", 500, 500);
    TransferFunction::fb_comb(0.9, 10).plot_pole_zero("pz4", 500, 500);
    TransferFunction::fb_comb(0.5, 100).plot_pole_zero("pz5", 500, 500);
    TransferFunction::fb_comb(0.7, 100).plot_pole_zero("pz6", 500, 500);
    TransferFunction::fb_comb(0.9, 100).plot_pole_zero("pz7", 500, 500);
}

// seems like my poles and zeros are reciprocated on here

// Why are the poles the reciprocal of the poles for this shit?
// so ok soi
// so like root finding seems correct
// maybe my representation doesn't handle decreasing powers of z properly?
// diff for polynomial of negative z: 