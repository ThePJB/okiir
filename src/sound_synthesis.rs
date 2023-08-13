use crate::vector::*;

use std::{fs::File, collections::VecDeque};
use std::io::BufWriter;

use riff_wave::WaveWriter;

use crate::rng::*;

pub const SAMPLE_RATE: u32 = 44100;

pub fn write_wav(outfile: &str, samples: Vec<f32>) {
    let file = File::create(outfile).unwrap();
	let writer = BufWriter::new(file);
	let mut wave_writer = WaveWriter::new(1, SAMPLE_RATE, 16, writer).unwrap();
    for s in samples {
        wave_writer.write_sample_i16((s * i16::MAX as f32) as i16).unwrap();
    }
}

pub fn pink_bad(n: usize) -> f32 {
    white(n) * white(n.wrapping_add(120947135137).wrapping_mul(928371241237))
}

pub fn pulse(hi: usize, lo: usize, n: usize) -> f32 {
    if n % (hi + lo) < hi {
        1.0
    } else {
        0.0
    }
}

pub fn white(n: usize) -> f32 {
    krand(n as u32)
}

pub struct TappedDelayLine {
    buf: VecDeque<f32>,
}

impl TappedDelayLine {
    pub fn new(len: usize) -> Self {
        let mut buf = VecDeque::new();
        for _ in 0..len {
            buf.push_front(0.0);
        }
        TappedDelayLine { buf }
    }

    pub fn tick(&mut self, in_sample: f32) -> f32 {
        let out_sample = self.buf.pop_back();
        self.buf.push_front(in_sample);

        out_sample.unwrap()
    }
}

pub struct FIR {
    line: TappedDelayLine,
    coefficients: Vec<f32>,
}

impl FIR {
    pub fn new(coefficients: Vec<f32>) -> Self {
        let len = coefficients.len();
        let line = TappedDelayLine::new(len);
        FIR {
            line,
            coefficients,
        }
    }

    pub fn tick(&mut self, in_sample: f32) -> f32 {
        // Feed the input sample to the delay line
        self.line.tick(in_sample);

        // Calculate the output sample using the FIR filter
        let mut output = 0.0;
        for i in 0..self.coefficients.len() {
            output += self.coefficients[i] * self.line.buf[i];
        }
        output
    }
}

pub fn lowpass_filter_coeffs(fs: f32, fc: f32, num_taps: usize) -> Vec<f32> {
    let nyquist = 0.5 * fs;
    let cutoff = fc / nyquist;

    // Calculate the filter impulse response (sinc function)
    let mut impulse_response = Vec::with_capacity(num_taps);
    let half_tap = (num_taps - 1) as f32 * 0.5;
    for n in 0..num_taps {
        let sinc_val = if n as f32 == half_tap {
            2.0 * cutoff
        } else {
            let t = (n as f32 - half_tap) * std::f32::consts::PI * cutoff;
            (t).sin() / t
        };
        impulse_response.push(sinc_val);
    }

    // Apply a window function (Hanning) to the impulse response
    let windowed_response: Vec<f32> = impulse_response
        .iter()
        .enumerate()
        .map(|(i, x)| x * 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (num_taps - 1) as f32).cos()))
        .collect();

    // Normalize the filter coefficients
    let sum: f32 = windowed_response.iter().sum();
    let normalized_response: Vec<f32> = windowed_response.iter().map(|x| x / sum).collect();

    normalized_response
}

// pub struct CombFilter {
//     delayline: TappedDelayLine,
// }

#[test]
pub fn test_comb() {
    let mut samples = vec![];
    let mut dc = TappedDelayLine::new(800);
    let mut y_prev = 0.0;
    for n in 0..SAMPLE_RATE * 10 {
        let n = n as usize;
        // let x = pink_bad(n) * pulse(40000, 40000, n);
        let x = white(n) * pulse(40000, 40000, n);
        let y = 0.1*x + 0.9*dc.tick(y_prev);
        y_prev = y;
        samples.push(0.5*y);
    }
    write_wav("comb1.wav", samples);
}

#[test]
pub fn test_comb_lp_fir1() {
    let mut samples = vec![];
    let mut dc = TappedDelayLine::new(1600);
    let mut fir = FIR::new(lowpass_filter_coeffs(44100.0, 200.0, 10)); // 10, 100, lol
    let mut y_prev = 0.0;
    for n in 0..SAMPLE_RATE * 10 {
        let n = n as usize;
        // let x = pink_bad(n) * pulse(40000, 40000, n);
        let x = white(n) * pulse(160000, 40000, n);
        let y = 0.1*x + 0.5*dc.tick(y_prev);
        y_prev = y;
        let y = fir.tick(y);
        samples.push(2.0*y);
    }
    write_wav("combfir1.wav", samples);
}

#[test]
pub fn test_comb_lp_fir2() {
    let mut samples = vec![];
    let mut dc = TappedDelayLine::new(1600);
    let mut fir = FIR::new(lowpass_filter_coeffs(44100.0, 400.0, 100)); // 10, 100, lol
    let mut y_prev = 0.0;
    for n in 0..SAMPLE_RATE * 10 {
        let n = n as usize;
        // let x = pink_bad(n) * pulse(40000, 40000, n);
        let x = white(n) * pulse(160000, 40000, n);
        let y = 0.1*x + 0.9*dc.tick(y_prev);
        y_prev = y;
        let y = fir.tick(y);
        samples.push(2.0*y);
    }
    write_wav("combfir2.wav", samples);
}

// and this is just filter on output, what about filter inside loop

#[test]
pub fn test_comb_lp_feedback_fir() {
    let mut samples = vec![];
    let mut dc = TappedDelayLine::new(1600);
    let mut fir = FIR::new(lowpass_filter_coeffs(44100.0, 800.0, 100)); // 10, 100, lol
    let mut y_prev = 0.0;
    for n in 0..SAMPLE_RATE * 10 {
        let n = n as usize;
        // let x = pink_bad(n) * pulse(40000, 40000, n);
        let x = white(n) * pulse(160000, 40000, n);
        let y = 0.1*x + 1.0*dc.tick(y_prev);
        let y = fir.tick(y);
        y_prev = y;
        samples.push(1.0*y);
    }
    write_wav("firfeedback.wav", samples);
}

// this is onto something but why it disappears


// going with explicit sizing, explicit code for the components here.
// distortion - just linear? or sampling multiple...

// Then imagine if it was really going crazy but you were enveloping to the envelope of a kick. Is that ring modulation, elementwise multiplication?
// or to a hard af bassline
// around 0dB for multiplication to be identity hey

// yea use delay based effects or echo effects. sick
// distortion that kicks in after a bit like shit just got HOT



// Todo
// write this
// take a break
// FIR coefficients calculator
// cone filter system
// basic systems that make nosie with cone filter
// exciter signal f(n)
// bruh delays are sick
// delays feedback
// flanges and echo/reverb
// 

// can I just impl all as traits of f32?
// or Sample which f32 is



// TODO FUCKING THE WARNINGS

// genetic algorithm trainer

// and evaluator

// just do it on IIR filters
// compressor fitness function:
// certain amount of gain
// certain amount of dynamics
// performance at certain frequencies
// allow a certain error kernel
// imagine if input was really quiet white noise and the fitness function was loudest / most saturated signal 
// + other methods of IIR filter design
// what about kalman filters


// in a way, making a certain IIR filter.
// could have genetic algorithms learning IIR coefficients to meet parameters.
// could just construct them. can make shit out of delay lines. TappedDelayLine is. FIR is.
// doing shit with multiple tapped delay lines is block diagram
// is there opportunity to specify better as ratio of two things?
// discrete logic: buffer + transfer function at the end
// could concatenate transfer functions, thats scuffed
// yea i mean comb filter is simplest z shit u can get
// H(z) = 1 + (gain * z^(-delay))
// thats fucked, you can combine the systems in parallel my +ing them
// and in series by *ing them
// so its like the ordering of parallel in serieses like an electronic circuit.
// computationally it must be convolution
// you can make infinite arbitrary architectures just multiplying the transfer functions together symbolically.
// then just collect z terms for buffer and implementation.
// thats so fucked it just reduces it down. synthesis just got easy AF
// power of the Z plane
// its equivalent to multiplying the H(w) frequency responses together maybe...


// but yea the genetic algorithm could be selecting for interesting things but then it gets more computationally expensive
// like it could select for distributions of dynamics temporally, movement, off a certain input.


// awesome_iir - genetic algos and infinite compositions (and primitives) eg comb
// pub struct LTI whatever whatever