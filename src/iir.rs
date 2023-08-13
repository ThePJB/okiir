use std::collections::vec_deque::*;
use crate::sound_synthesis::*;
use super::transfer_function::*;
use crate::vector::*;

// todo probably normalize A(0) during implementation
// todo fir topology
// todo y unit tests failin bra

pub struct IIR {
    h: TransferFunction,
    rb_input: VecDeque<Vec2>,
    rb_output: VecDeque<Vec2>,
}

impl From<TransferFunction> for IIR {
    fn from(h: TransferFunction) -> Self {
        let len = h.numerator.coefficients.len().max(h.denominator.coefficients.len());
        let rb_input = VecDeque::from(vec![vec2(0.0, 0.0); len]);
        let rb_output = VecDeque::from(vec![vec2(0.0, 0.0); len]);
        let a0 = h.numerator.coefficients[0];
        let h = TransferFunction {
            numerator: Polynomial::from(h.numerator.coefficients.iter().map(|c| *c / a0).collect::<Vec<Vec2>>()),
            denominator: Polynomial::from(h.denominator.coefficients.iter().map(|c| *c / a0).collect::<Vec<Vec2>>()),
        };
        IIR {
            h,
            rb_input,
            rb_output,
        }
    }
}

impl IIR {
    pub fn tick(&mut self, input: Vec2)-> Vec2 {
        let mut input_sum = input;
        for i in 1..self.h.denominator.coefficients.len() {
            input_sum += self.h.denominator.coefficients[i]**self.rb_input.get(i-1).unwrap();
        }
        let mut output_sum = input_sum * self.h.numerator.coefficients[0];
        for i in 1..self.h.numerator.coefficients.len() {
            output_sum += self.h.numerator.coefficients[i]**self.rb_input.get(i-1).unwrap();

        }
        self.rb_input.push_front(input_sum);
        self.rb_input.pop_back();

        output_sum
    }    
    pub fn tick_old(&mut self, input: Vec2)-> Vec2 {
        let mut input_sum = vec2(0.0, 0.0);
        input_sum += input * self.h.numerator.coefficients[0];
        for i in 1..self.h.numerator.coefficients.len() {
            input_sum += self.h.numerator.coefficients[i]**self.rb_input.get(i-1).unwrap();
        }
        
        let mut output_sum = input_sum;
        for i in 1..self.h.denominator.coefficients.len() {
            output_sum += self.h.denominator.coefficients[i]**self.rb_output.get(i-1).unwrap();
        }
        self.rb_input.push_front(input);
        self.rb_input.pop_back();
        self.rb_output.push_front(output_sum);
        self.rb_output.pop_back();

        output_sum
    }    
}

pub fn run_and_output(name: &str, h: TransferFunction) {
    let mut sys = IIR::from(h);
    let samples = (0..SAMPLE_RATE*4).map(|n| {
        let n = n as usize;
        let x = 0.01*white(n)*pulse(5000, 40000, n);
        let y = sys.tick(vec2(x, 0.0));
        y
    });
    write_wav(&format!("{}.wav", name), samples.map(|x| x.x).collect());
}

#[test]
pub fn test_comb() {
    run_and_output("comb220-0.7", TransferFunction::fb_comb(0.7, 220));
    run_and_output("comb220-0.9", TransferFunction::fb_comb(0.9, 220));
    run_and_output("comb220-0.99", TransferFunction::fb_comb(0.99, 220));
    run_and_output("comb440-0.99", TransferFunction::fb_comb(0.99, 440));
    run_and_output("comb440-0.9", TransferFunction::fb_comb(0.9, 440));
    run_and_output("comb440-0.7", TransferFunction::fb_comb(0.7, 440));
}

#[test]
pub fn test_2comb() {

    // this is sorta violating my assumptions that it should stay bounded.
    // maybe need to inspect tf poles, zeros, stability, etc.
    // it would be good to plot the impulse response.
    // get min and max and just draw lines between the points on a plot.
    // ohh i guess comb filter gets many poles when its higher order... with no way of knowing if this shit is correct
    // i can do more tests lol
    // maybe look into it in matlab

    let q = TransferFunction::fb_comb(0.7, 220);
    q.plot_pole_zero("q_pz", 1000, 1000);
    q.plot_impulse_response("q", 10000, 1000, 1000);
    let p = TransferFunction::fb_comb(0.7, 500);
    p.plot_pole_zero("p_pz", 1000, 1000);
    p.plot_impulse_response("p", 10000, 1000, 1000);
    let s = q.clone() + p.clone();
    s.plot_pole_zero("s_pz", 1000, 1000);
    s.plot_impulse_response("q_plus_p", 10000, 1000, 1000);
    run_and_output("qplusp", s);
    let s = q * p;
    s.plot_impulse_response("q_times_p", 10000, 1000, 1000);
    run_and_output("qtimesp", s);
}



pub fn check_stability(case: &str, h: TransferFunction, n: usize) {
    let mut running = IIR::from(h.clone());
    running.tick(vec2(1.0, 0.0));
    for i in 0..n {
        running.tick(vec2(0.0, 0.0));
    }
    let val = running.tick(vec2(0.0, 0.0));
    assert!(val.magnitude() < 1.0, "{}: h: {:?}, y[{}]={}", case, h, n, val);
}


#[test]
pub fn test_iir() {
    check_iir("identity", TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0])), vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]);
    
    check_iir("onemorezero", TransferFunction::new(Polynomial::from(vec![1.0, 1.0]), Polynomial::from(vec![1.0])), vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 1.0, 0.0, 0.0]);
    check_iir("onemorezero_longer", TransferFunction::new(Polynomial::from(vec![1.0, 1.0]), Polynomial::from(vec![1.0])), vec![1.0, 1.0, 0.0, 0.0], vec![1.0, 2.0, 1.0, 0.0]);

    check_iir("p", TransferFunction::new(Polynomial::from(vec![1.0]), Polynomial::from(vec![1.0, 0.5])), vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.5, 0.25, 0.125]);
    check_iir("zp", TransferFunction::new(Polynomial::from(vec![1.0, 1.0]), Polynomial::from(vec![1.0, 0.5])), vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 1.5, 0.75, 0.375]);
    check_iir("zp2up", TransferFunction::new(Polynomial::from(vec![1.0, 1.0]), Polynomial::from(vec![1.0, 0.5])), vec![1.0, 1.0, 0.0, 0.0], vec![1.0, 2.5, 2.25, 1.125]);
}

pub fn check_iir(case: &str, h: TransferFunction, inputs: Vec<f32>, expected: Vec<f32>) {
    let mut running = IIR::from(h);
    for (i, x) in inputs.iter().enumerate() {
        let y = running.tick(vec2(*x, 0.0));
        assert_eq!(y.x, expected[i], "{} y[{}]={}, expected {}", case, i, y, expected[i]);
    }
}
